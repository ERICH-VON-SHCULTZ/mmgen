"""Stage 2/3 of the MFD listwise reranker: train a shallow cross-encoder that rescores the
frozen MFD+PMQ+Q_f candidate pool, and evaluate it two-stage (frozen generator -> reranker
-> top-10).

The frozen main model is NOT loaded here. Stage 1 (mfd_dump_memory=true, an mfd_audit_dump
pass at K_f=200,K_t=200,K_i=50,K_pool=500) already wrote, per event, everything the reranker
needs into a compact .pt: the per-behavior candidate pools (f,t,i), their four component
log-probs (behavior, f, t|f, i|f), the frozen ENCODER MEMORY (full history hidden states) +
its mask, the ground-truth SID, and gt_in_catalog. So this script is a self-contained plain-
PyTorch trainer; the offline rescoring of the dumped pool over the dumped memory is bit-for-bit
the two-stage inference the deliverable asks for.

WHY (from the full 83,005-event G4 audit): GT is inside the pool 26.2% of the time, but t3
recall@10 is only 0.2041 and P(rank<10 | covered) = 0.781. Sweeping K_pool 500->2000 does not
move recall@10 -- the candidates are present, the hand-weighted log-prob sum just can't surface
them. Ordering headroom is +0.057. This reranker is the ordering fix.

Design (the three deliberate choices; earlier MSE/random-negative/pooled-query rerankers
degraded recall and are NOT reintroduced):
  * candidate geometry from the MD-RQ-VAE codebook, lifted through an orthogonal W_up into
    d_model: three view vectors [u_f, u_f+u_t, u_f+u_i]  (no new item embedding table);
  * the four component log-probs fed in as SCALAR features (learn the weighting, don't inherit
    w_b / tau);
  * a shallow cross-encoder: candidate tokens CROSS-ATTEND into the FULL encoder memory
    sequence (not a pooled summary), output one scalar per candidate;
  * listwise cross-entropy: softmax over the whole pool, GT as the positive, trained only on
    events where GT is in the pool, with the generator's own real-pool hard negatives.

Recall is computed with the SAME union_score / best_rank / recall_at primitives as
analyze_mfd_audit.py (the reranker only replaces the SCORE), so t3/t2 numbers stay directly
comparable to every prior audit. Baseline (union_score, the current system) is reported next
to the reranked numbers on every eval.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

# The MD-RQ-VAE Lightning checkpoint is pickled with references to src.* classes (its
# hyper_parameters / module classes), so unpickling it with weights_only=False needs the GRID
# package importable -- exactly why GRID's own codebook loader works only from the repo root.
# This standalone script runs from rec-tmall, so put the GRID root on sys.path up front.
sys.path.insert(0, os.environ.get("GRID_ROOT", "/scratch/yw8866/GRID"))

import torch
import torch.nn as nn
import torch.nn.functional as F


BEH = {0: "pv", 1: "fav", 2: "cart", 3: "buy"}
MD_K1, MD_K2 = 1024, 1024   # MD-RQ-VAE text/image codebook widths (for packing (f,t,i) -> key)
REPR_VARIANTS = ["md3", "shared", "mbgen1", "tiger1", "tiger3", "learned", "random"]


# ---------------------------------------------------------------------------
# ranking primitives -- copied verbatim from analyze_mfd_audit.py so the reranked
# numbers are computed on the identical denominators (only the SCORE differs).
# ---------------------------------------------------------------------------
def best_rank(score, match, eligible):
    """Per event: 0-based rank of the best matching+eligible candidate among eligible ones,
    or -1 if none. rank = #eligible candidates scoring strictly higher than that match."""
    score = score.masked_fill(~eligible, float("-inf"))
    m_score = torch.where(match & eligible, score, torch.full_like(score, float("-inf")))
    best = m_score.max(dim=1).values
    has = best > -1e30
    rank = (score > best.unsqueeze(1)).sum(1)
    rank = torch.where(has, rank, torch.full_like(rank, -1))
    return rank, has


def recall_at(rank, has, k, denom_mask=None):
    hit = has & (rank >= 0) & (rank < k)
    if denom_mask is not None:
        return hit[denom_mask].float().mean().item()
    return hit.float().mean().item()


def ndcg_at(rank, has, k, denom_mask=None):
    """NDCG@k for a single relevant item (the GT). At 0-based rank r<k: DCG=1/log2(r+2), IDCG=1."""
    hit = has & (rank >= 0) & (rank < k)
    gain = torch.where(hit, 1.0 / torch.log2(rank.clamp(min=0).float() + 2.0), torch.zeros_like(rank, dtype=torch.float))
    if denom_mask is not None:
        return gain[denom_mask].mean().item()
    return gain.mean().item()


# ---------------------------------------------------------------------------
# geometry: MD-RQ-VAE centroids -> orthogonal up-projection -> per-level view tables
# ---------------------------------------------------------------------------
_CENTROID_KEYS = {"shared": "shared_codebook.centroids",
                  "text": "text_codebook.centroids",
                  "image": "image_codebook.centroids"}


def load_geometry(mdrqvae_ckpt, cache, d_model, seed=0, device="cpu"):
    """Return (U_f, U_t, U_i) each (K_*, d_model): the codebook centroids lifted through a
    FIXED orthogonal W_up (isometry -> preserves the codebook cosine structure). Centroids are
    cached to a small .pt so the 50GB MD-RQ-VAE checkpoint is loaded at most once."""
    if os.path.exists(cache):
        g = torch.load(cache, map_location="cpu", weights_only=False)
        c_f, c_t, c_i = g["c_f"].float(), g["c_t"].float(), g["c_i"].float()
        print(f"[geom] loaded cached centroids {cache}", flush=True)
    else:
        print(f"[geom] extracting centroids from {mdrqvae_ckpt} (one-time; ~50GB load) ...", flush=True)
        # The checkpoint is a Lightning ckpt pickled with src.* class references (TokenizerConfig,
        # the tokenizer LightningModule, ...). Importing those COLD via the unpickler trips a
        # genuine interfaces<->utils circular import in GRID (only latent because normal app runs
        # import src.utils first). Warm src.utils.utils here so the package graph initialises in the
        # working order and interfaces is fully cached in sys.modules before torch.load needs it.
        try:
            import src.utils.utils  # noqa: F401
        except Exception as ex:
            print(f"[geom][warn] GRID warm-import failed ({ex}); attempting load anyway", flush=True)
        sd = torch.load(mdrqvae_ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        c_f = sd[_CENTROID_KEYS["shared"]].float()
        c_t = sd[_CENTROID_KEYS["text"]].float()
        c_i = sd[_CENTROID_KEYS["image"]].float()
        torch.save({"c_f": c_f, "c_t": c_t, "c_i": c_i}, cache)
        print(f"[geom] cached centroids -> {cache}  "
              f"(K_f={c_f.shape[0]} K_t={c_t.shape[0]} K_i={c_i.shape[0]} d_vae={c_f.shape[1]})",
              flush=True)
    d_vae = c_f.shape[1]
    torch.manual_seed(seed)                       # deterministic W_up across train+eval
    w_up = torch.empty(d_vae, d_model)
    nn.init.orthogonal_(w_up)
    with torch.no_grad():
        U_f = (c_f @ w_up).to(device)
        U_t = (c_t @ w_up).to(device)
        U_i = (c_i @ w_up).to(device)
    return U_f, U_t, U_i


# ---------------------------------------------------------------------------
# dump -> flat candidate tensors (identical layout to analyze_mfd_audit.build_candidates)
# ---------------------------------------------------------------------------
class RerankData:
    """Holds the flattened per-event candidate tensors for one split. All CPU; batches are
    moved to GPU on demand. Candidate axis C = nb * Kp, ordered [beh0 pool | beh1 pool | ...],
    exactly as analyze_mfd_audit flattens it."""

    def __init__(self, dump_path, wb, kpool=None):
        d = torch.load(dump_path, map_location="cpu", weights_only=False)
        c, meta = d["cols"], d["meta"]
        if not meta.get("has_memory", False) or "enc_mem" not in c:
            raise ValueError(f"{dump_path} has no encoder memory -- re-dump with mfd_dump_memory=true")
        self.meta = dict(meta)
        N, nb, Kp = c["pool_f"].shape
        # K_pool sweep from ONE dump: the per-behavior pool slots are score-sorted (topk output), so
        # keeping the first `kpool` slots IS the top-kpool pool. Sliced + made contiguous so the wide
        # tensors can be freed.
        self.meta["Kpool_stored"] = Kp
        if kpool is not None and kpool < Kp:
            for k in ("pool_f", "pool_t", "pool_i", "pool_sidscore", "pool_lpf", "pool_lpt", "pool_lpi"):
                c[k] = c[k][:, :, :kpool].contiguous()
            Kp = kpool
            self.meta["Kpool"] = kpool
        self.kpool = Kp
        self.N, self.nb, self.Kp = N, nb, Kp
        self.C = nb * Kp
        f32 = lambda x: x.float() if x.dtype == torch.float16 else x
        flat = lambda x: x.reshape(N, nb * Kp)

        self.user_id = c["user_id"].long() if "user_id" in c else None
        self.gt_b = c["gt_b"].long()
        self.gt_f = c["gt_f"].long()
        self.gt_t = c["gt_t"].long()
        self.gt_i = c["gt_i"].long()

        # kept int16 (f<2048, t/i<1024) -> .long() per batch. At K_pool=2000 (C=8000) this 4x's
        # the candidate RAM vs long, which is what keeps the wide dumps loadable.
        self.f = flat(c["pool_f"]).to(torch.int16)              # (N, C)
        self.t = flat(c["pool_t"]).to(torch.int16)
        self.i = flat(c["pool_i"]).to(torch.int16)
        # branch (behavior of each candidate) is the SAME (C,) vector for every event -> store 1-D
        # and broadcast at use time (saves ~N*C RAM).
        self.branch = torch.arange(nb).repeat_interleave(Kp)   # (C,)
        branch_NC = self.branch.view(1, -1).expand(N, -1)      # (N, C) view, no copy

        lp_f = f32(flat(c["pool_lpf"]))                         # (N, C)  log P(f|b)
        lp_t = f32(flat(c["pool_lpt"]))                         # (N, C)  log P(t|f,b)
        lp_i = f32(flat(c["pool_lpi"]))                         # (N, C)  log P(i|f,b)
        beh_logp = f32(c["beh_logp"])                          # (N, nb)
        lp_beh = torch.gather(beh_logp, 1, branch_NC)          # (N, C)  log P(b)
        self.lp = torch.stack([lp_beh, lp_f, lp_t, lp_i], dim=-1).to(torch.float16)  # (N,C,4) -> .float() per batch

        self.valid = f32(flat(c["pool_sidscore"])) > -1e30     # (N, C)
        # baseline union score = wb*logP(b) + logP(f) + logP(t|f) + logP(i|f) (analyze's union_score).
        # Kept FINITE for padding/invalid slots (their component logps are finite/zero); the -inf
        # mask is applied ONLY to the final score, so base_scale never multiplies -inf (no 0*-inf NaN).
        self.base = (wb * lp_beh + lp_f + lp_t + lp_i).to(torch.float16)  # (N,C) finite -> .float() per batch

        # GT candidate index in the union (branch==gt_b AND full SID match), -1 if not covered
        sid_match = ((self.f == self.gt_f[:, None]) & (self.t == self.gt_t[:, None])
                     & (self.i == self.gt_i[:, None]))
        beh_match = branch_NC == self.gt_b[:, None]
        gt_cand = sid_match & beh_match & self.valid           # (N, C), <=1 True/event
        self.has_gt = gt_cand.any(1)                           # GT covered by the union pool
        self.gt_idx = torch.where(self.has_gt, gt_cand.float().argmax(1),
                                  torch.full((N,), -1, dtype=torch.long))

        self.mem = c["enc_mem"]                                # (N, S, d) fp16 (kept fp16 on CPU)
        self.mem_pad = c["enc_mask"].to(torch.bool).logical_not()  # (N, S) True where PAD

        # ---- history for ENGAGEMENT features (present only in engagement dumps) ----
        # input_ids is the interleaved [b,f,t,i,...] history (oldest->newest at the FRONT, pad=-1,
        # right-padded per collate_fn_train). Reshape to per-item (N,H,4) and split the channels.
        self.has_hist = "hist_ids" in c
        if self.has_hist:
            hid = c["hist_ids"].long()                         # (N, S)
            H = hid.shape[1] // 4
            hid = hid[:, :H * 4].reshape(N, H, 4)              # (N,H,4): [b, f, t, i]
            hb = hid[:, :, 0]                                  # (N,H) behavior, -1 for pad items
            hvalid = hb >= 0                                   # (N,H)  valid history item
            num_valid = hvalid.sum(1, keepdim=True).clamp(min=1)          # (N,1)
            pos = torch.arange(H).view(1, H)                              # 0=oldest .. front-valid=newest
            # recency in (0,1]: most-recent valid item -> 1.0, oldest -> 1/num_valid, pad -> 0
            recency = ((pos + 1).float() / num_valid) * hvalid.float()    # (N,H)
            self.hist_b = hb.to(torch.int16)                  # keep -1 (used in ==behavior)
            self.hist_f = hid[:, :, 1].clamp(min=0).to(torch.int16)   # clamp pad(-1)->0; masked by hvalid
            self.hist_t = hid[:, :, 2].clamp(min=0).to(torch.int16)
            self.hist_i = hid[:, :, 3].clamp(min=0).to(torch.int16)
            self.hist_recency = recency.to(torch.float16)     # (N,H)
            self.H = H

        self.train_idx = torch.nonzero(self.has_gt, as_tuple=False).squeeze(1)
        print(f"[data] {os.path.basename(dump_path)}: N={N:,}  C={self.C}  "
              f"covered(GT in pool)={int(self.has_gt.sum()):,} ({self.has_gt.float().mean():.4f})  "
              f"S={self.mem.shape[1]} d={self.mem.shape[2]}  "
              f"hist={'yes(H=%d)' % self.H if self.has_hist else 'NO'}", flush=True)


# ---------------------------------------------------------------------------
# reranker: shallow cross-encoder (candidate tokens cross-attend into full memory)
# ---------------------------------------------------------------------------
class CrossBlock(nn.Module):
    def __init__(self, d, heads, drop):
        super().__init__()
        self.norm_q = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.norm_f = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x, mem, mem_pad):
        # x (B,C,d) candidate tokens; mem (B,S,d) frozen encoder memory; mem_pad (B,S) True=pad.
        # Candidates cross-attend into memory only (no candidate<->candidate attention: the
        # listwise coupling lives in the softmax, keeping the head permutation-invariant).
        a, _ = self.attn(self.norm_q(x), mem, mem, key_padding_mask=mem_pad, need_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.norm_f(x)))
        return x


class MFDReranker(nn.Module):
    def __init__(self, U_f, U_t, U_i, d_model, layers=2, heads=4, drop=0.1,
                 num_behaviors=4, use_hard=False, use_soft=False,
                 cand_repr="md3", repr_keys=None, repr_table=None, repr_M=None):
        super().__init__()
        self.register_buffer("U_f", U_f)   # frozen codebook geometry (not trained)
        self.register_buffer("U_t", U_t)
        self.register_buffer("U_i", U_i)
        self.nb = num_behaviors
        self.use_hard = use_hard
        self.use_soft = use_soft
        # ENGAGEMENT feature width: hard = nb*(indicator,log1p count,recency); soft = nb*3 views.
        self.n_hard = (3 * num_behaviors) if use_hard else 0
        self.n_soft = (3 * num_behaviors) if use_soft else 0
        n_scalar = 4 + self.n_hard + self.n_soft           # 4 component log-probs + engagement
        d = d_model

        # ---- CANDIDATE REPRESENTATION (the ablated axis; everything else identical) ----
        # md3   : [u_f, u_f+u_t, u_f+u_i]  (MD-RQ-VAE 3 modality views; current)   d_repr=3d
        # shared: u_f only                 (MD shared level alone)                  d_repr=d
        # mbgen1: MBGen level-1 QAE embedding (lifted)  -> frozen (M,d) table       d_repr=d
        # tiger1: TIGER chain level-1 centroid (lifted) -> frozen (M,d) table       d_repr=d
        # tiger3: TIGER chain residual sum c1+c2+c3 (lifted) -> frozen (M,d) table  d_repr=d
        # learned: TRAINABLE nn.Embedding(M,d) over unique tuples (the huge table)  d_repr=d
        # random : frozen random (M,d) per tuple (lower-bound control)              d_repr=d
        self.cand_repr = cand_repr
        self.repr_added_params = 0
        if cand_repr in ("mbgen1", "tiger1", "tiger3", "learned", "random"):
            self.register_buffer("repr_keys", repr_keys)     # (M,) sorted packed MD keys
            if cand_repr == "learned":
                self.repr_emb = nn.Embedding(repr_M, d)      # jointly trained item(=tuple) table
                nn.init.normal_(self.repr_emb.weight, std=0.02)
                self.repr_added_params = self.repr_emb.weight.numel()
            else:                                            # frozen (M,d) table (0 trainable params)
                self.register_buffer("repr_table", repr_table)
        d_repr = 3 * d if cand_repr == "md3" else d
        self.view_proj = nn.Linear(d_repr, d)
        self.scalar_proj = nn.Sequential(nn.Linear(n_scalar, d), nn.GELU(), nn.Linear(d, d))
        self.in_norm = nn.LayerNorm(d)
        self.in_drop = nn.Dropout(drop)                      # feature dropout on candidate tokens
        self.blocks = nn.ModuleList([CrossBlock(d, heads, drop) for _ in range(layers)])
        self.out_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)
        # zero-init head + base_scale=1  ->  at step 0 score == baseline union_score EXACTLY
        # (the reranker starts as an identity on the baseline ordering, then learns the delta).
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.base_scale = nn.Parameter(torch.tensor(1.0))

    def _cand_repr(self, f, t, i):
        """Candidate view vectors (Nc, d_repr) for the active representation variant."""
        if self.cand_repr == "md3":
            u_f = self.U_f[f]
            return torch.cat([u_f, u_f + self.U_t[t], u_f + self.U_i[i]], -1)   # (Nc, 3d)
        if self.cand_repr == "shared":
            return self.U_f[f]                                                  # (Nc, d)
        # table-indexed variants: MD tuple -> row via searchsorted into the unique catalog keys
        packed = (f.long() * MD_K1 + t.long()) * MD_K2 + i.long()
        row = torch.searchsorted(self.repr_keys, packed).clamp(max=self.repr_keys.numel() - 1)
        if self.cand_repr == "learned":
            return self.repr_emb(row)
        return self.repr_table[row]                                            # (Nc, d)

    def _engagement(self, f, t, i, hist):
        """Per-candidate engagement features vs the user's history, per behavior.
        f,t,i: (B,C) long. hist = (hf,ht,hi,hb,hvalid,recency), each (B,H).
        Returns (B, C, n_hard + n_soft)."""
        hf, ht, hi, hb, hvalid, recency = hist
        B, C = f.shape
        feats = []
        # HARD: exact SID-tuple match of the candidate against history items of each behavior.
        if self.use_hard:
            match = ((f[:, :, None] == hf[:, None, :])
                     & (t[:, :, None] == ht[:, None, :])
                     & (i[:, :, None] == hi[:, None, :]))         # (B,C,H)
            rec = recency[:, None, :]                             # (B,1,H)
            for b in range(self.nb):
                bm = ((hb == b) & hvalid)[:, None, :]             # (B,1,H)
                mb = match & bm                                  # (B,C,H)
                cnt = mb.sum(-1).float()                         # (B,C)
                ind = (cnt > 0).float()
                recb = (mb.float() * rec).amax(-1)               # most-recent match recency, 0 if none
                feats += [ind, torch.log1p(cnt), recb]
        # SOFT: max cosine of the candidate's view vectors vs history items of each behavior.
        if self.use_soft:
            cuf = self.U_f[f]; cvt = cuf + self.U_t[t]; cvi = cuf + self.U_i[i]      # (B,C,d)
            huf = self.U_f[hf]; hvt = huf + self.U_t[ht]; hvi = huf + self.U_i[hi]   # (B,H,d)
            for cv, hv in ((cuf, huf), (cvt, hvt), (cvi, hvi)):
                cvn = F.normalize(cv, dim=-1); hvn = F.normalize(hv, dim=-1)
                sim = torch.bmm(cvn, hvn.transpose(1, 2))        # (B,C,H) cosine in [-1,1]
                for b in range(self.nb):
                    bm = ((hb == b) & hvalid)[:, None, :]        # (B,1,H)
                    m = sim.masked_fill(~bm.expand_as(sim), -2.0).amax(-1)   # (B,C); -2 if no such item
                    feats.append(torch.where(m < -1.5, torch.zeros_like(m), m))
        return torch.stack(feats, dim=-1)                        # (B, C, n_hard+n_soft)

    def cand_tokens(self, f, t, i, scal):
        x = self.view_proj(self._cand_repr(f, t, i)) + self.scalar_proj(scal)
        return self.in_drop(self.in_norm(x))

    def forward(self, f, t, i, lp, base, valid, mem, mem_pad, hist=None):
        # f,t,i (B,C) long; lp (B,C,4); base,valid (B,C); mem (B,S,d); mem_pad (B,S) True=pad;
        # hist = (hf,ht,hi,hb,hvalid,recency) each (B,H), or None (views-only ablation).
        B, C = f.shape
        scal = lp
        if (self.use_hard or self.use_soft):
            scal = torch.cat([lp, self._engagement(f, t, i, hist)], dim=-1)   # (B,C,4+n_eng)
        x = self.cand_tokens(f.reshape(-1), t.reshape(-1), i.reshape(-1),
                             scal.reshape(-1, scal.size(-1))).reshape(B, C, -1)
        for blk in self.blocks:
            x = blk(x, mem, mem_pad)
        delta = self.head(self.out_norm(x)).squeeze(-1)      # (B, C)
        score = self.base_scale * base + delta
        return score.masked_fill(~valid, float("-inf"))


def hist_batch(data, idx, device):
    """Assemble the per-event history tensors for `idx` (or None if the dump has no history)."""
    if not data.has_hist:
        return None
    hb = data.hist_b[idx]
    return (data.hist_f[idx].long().to(device),
            data.hist_t[idx].long().to(device),
            data.hist_i[idx].long().to(device),
            hb.long().to(device),
            (hb >= 0).to(device),
            data.hist_recency[idx].float().to(device))


# ---------------------------------------------------------------------------
# eval: two-stage recall on a split, reranked vs baseline, on analyze's denominators
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, data: RerankData, device, batch=64, use_base_only=False):
    model.eval()
    N = data.N
    r3 = torch.empty(N, dtype=torch.long); h3 = torch.empty(N, dtype=torch.bool)
    r2 = torch.empty(N, dtype=torch.long); h2 = torch.empty(N, dtype=torch.bool)
    for s in range(0, N, batch):
        e = min(N, s + batch)
        idx = torch.arange(s, e)
        f = data.f[idx].to(device).long(); t = data.t[idx].to(device).long(); i = data.i[idx].to(device).long()
        branch = data.branch.to(device).view(1, -1).expand(e - s, -1)
        valid = data.valid[idx].to(device)
        base = data.base[idx].to(device).float()
        gt_f = data.gt_f[idx].to(device); gt_t = data.gt_t[idx].to(device)
        gt_i = data.gt_i[idx].to(device); gt_b = data.gt_b[idx].to(device)
        if use_base_only:
            score = base.masked_fill(~valid, float("-inf"))
        else:
            lp = data.lp[idx].to(device).float()
            mem = data.mem[idx].to(device).float()
            mem_pad = data.mem_pad[idx].to(device)
            h = hist_batch(data, idx, device) if (model.use_hard or model.use_soft) else None
            score = model(f, t, i, lp, base, valid, mem, mem_pad, h)
        sid_match = (f == gt_f[:, None]) & (t == gt_t[:, None]) & (i == gt_i[:, None])
        beh_match = branch == gt_b[:, None]
        # t3 union (marginalized): match = SID & behavior, eligible = all valid
        rr3, hh3 = best_rank(score, sid_match & beh_match, valid)
        # t2 routed: match = SID, eligible = valid & GT-behavior branch
        rr2, hh2 = best_rank(score, sid_match, valid & beh_match)
        r3[s:e], h3[s:e] = rr3.cpu(), hh3.cpu()
        r2[s:e], h2[s:e] = rr2.cpu(), hh2.cpu()
    return r3, h3, r2, h2


def report(tag, data, r3, h3, r2, h2):
    gt_b = data.gt_b
    covered = h3                                              # GT present in the union pool
    p_r10_cov = recall_at(r3, h3, 10, covered)               # P(rank<10 | covered)  (baseline 0.781)
    line = (f"  {tag:<10} "
            f"t3@5={recall_at(r3, h3, 5):.4f} t3@10={recall_at(r3, h3, 10):.4f} | "
            f"t2@5={recall_at(r2, h2, 5):.4f} t2@10={recall_at(r2, h2, 10):.4f} | "
            f"P(r<10|cov)={p_r10_cov:.4f} cov={covered.float().mean():.4f}")
    print(line, flush=True)
    # full metric set: recall@20 + NDCG@{5,10} for t3 (union) and t2 (routed)
    print(f"             t3 R@20={recall_at(r3, h3, 20):.4f} NDCG@5={ndcg_at(r3, h3, 5):.4f} "
          f"NDCG@10={ndcg_at(r3, h3, 10):.4f}  |  "
          f"t2 R@20={recall_at(r2, h2, 20):.4f} NDCG@5={ndcg_at(r2, h2, 5):.4f} "
          f"NDCG@10={ndcg_at(r2, h2, 10):.4f}", flush=True)
    # per-behavior t3@10 (pv dominance context) and t2@10
    p3 = [f"{BEH.get(b, b)}={recall_at(r3, h3, 10, gt_b == b):.4f}" for b in range(data.nb) if (gt_b == b).any()]
    p2 = [f"{BEH.get(b, b)}={recall_at(r2, h2, 10, gt_b == b):.4f}" for b in range(data.nb) if (gt_b == b).any()]
    print(f"             t3@10 by behavior: " + "  ".join(p3), flush=True)
    print(f"             t2@10 by behavior: " + "  ".join(p2), flush=True)
    return recall_at(r3, h3, 10)


# ---------------------------------------------------------------------------
# training: listwise cross-entropy over the pool, GT positive, real-pool negatives
# ---------------------------------------------------------------------------
def train(model, tr: RerankData, va: RerankData, te: RerankData, device, args):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    idx_all = tr.train_idx
    steps_per_epoch = math.ceil(len(idx_all) / args.batch)

    print(f"\n[train] {len(idx_all):,} GT-in-pool TRAIN events (of {tr.N:,})  batch={args.batch}  "
          f"{steps_per_epoch} steps/epoch  epochs<={args.epochs}", flush=True)
    print(f"[select] on VAL ({va.train_idx.numel():,} GT-in-pool of {va.N:,}); "
          f"[report] on TEST ({te.train_idx.numel():,} GT-in-pool of {te.N:,}). "
          f"No test peeking for selection.\n", flush=True)

    # baseline reference (no reranker) on BOTH val and test
    print("[eval] baseline (union_score, current system):", flush=True)
    base_test_arrays = evaluate(model, te, device, args.eval_batch, use_base_only=True)
    base_val = report("BASE val", va, *evaluate(model, va, device, args.eval_batch, use_base_only=True))
    base_test = report("BASE test", te, *base_test_arrays)

    best_val = -1.0; best_ep = 0; since = 0; best_test = float("nan"); best_test_arrays = None
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = idx_all[torch.randperm(len(idx_all))]
        t0 = time.time(); running = 0.0; nb = 0
        for s in range(0, len(perm), args.batch):
            bi = perm[s:s + args.batch]
            f = tr.f[bi].to(device).long(); t = tr.t[bi].to(device).long(); i = tr.i[bi].to(device).long()
            lp = tr.lp[bi].to(device).float(); base = tr.base[bi].to(device).float(); valid = tr.valid[bi].to(device)
            mem = tr.mem[bi].to(device).float(); mem_pad = tr.mem_pad[bi].to(device)
            gt_idx = tr.gt_idx[bi].to(device)
            h = hist_batch(tr, bi, device) if (model.use_hard or model.use_soft) else None
            score = model(f, t, i, lp, base, valid, mem, mem_pad, h)                    # (B,C)
            loss = F.cross_entropy(score, gt_idx)                   # listwise softmax over the pool
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item(); nb += 1
        print(f"[ep {ep:>2}] listwise CE={running / max(nb, 1):.4f}  "
              f"({time.time() - t0:.0f}s)  base_scale={model.base_scale.item():.3f}", flush=True)
        val_t3 = report(f"VAL  ep{ep}", va, *evaluate(model, va, device, args.eval_batch))
        test_arrays = evaluate(model, te, device, args.eval_batch)
        test_t3 = report(f"TEST ep{ep}", te, *test_arrays)
        # SELECT on val t3@10 only (test is printed for transparency, never used for selection or
        # stopping). Keep the best-by-val checkpoint; report the TEST number AT that epoch.
        if val_t3 > best_val + args.min_delta:
            best_val = val_t3; best_ep = ep; since = 0; best_test = test_t3; best_test_arrays = test_arrays
            if args.save:
                torch.save({"state_dict": model.state_dict(), "args": vars(args), "epoch": ep,
                            "val_t3_recall_10": val_t3, "test_t3_recall_10": test_t3}, args.save)
                print(f"             * new best VAL t3@10={best_val:.4f} (test {test_t3:.4f}) "
                      f"-> saved {args.save}", flush=True)
        else:
            since += 1
            if since >= args.patience:
                print(f"[early stop] no VAL t3@10 gain > {args.min_delta} for {args.patience} "
                      f"epochs -> stop at ep{ep}. best = ep{best_ep} (val={best_val:.4f}).", flush=True)
                break

    print(f"\n[done] model selected on VAL @ ep{best_ep} (val t3@10={best_val:.4f}).", flush=True)
    print(f"[done] TEST t3@10: baseline={base_test:.4f}  ->  reranked={best_test:.4f}  "
          f"(gain {best_test - base_test:+.4f}; target 0.233, coverage ceiling ~0.262)", flush=True)
    print(f"[done] (VAL baseline t3@10={base_val:.4f})", flush=True)
    if best_test_arrays is None:                              # never improved past init
        best_test_arrays = base_test_arrays
    return {"best_ep": best_ep, "best_val": best_val,
            "base_test_arrays": base_test_arrays, "best_test_arrays": best_test_arrays}


def orth_lift(raw, d_model, seed):
    """Lift raw centroid vectors (M, d_raw) into d_model via a FIXED orthogonal W_up (the same
    isometry mechanism as the MD-RQ-VAE views). Returns (M, d_model) float."""
    d_raw = raw.shape[1]
    torch.manual_seed(seed)
    w = torch.empty(d_raw, d_model)
    nn.init.orthogonal_(w)
    return raw.float() @ w


def make_repr(cand_repr, cache, d_model, seed, device):
    """Build (repr_keys, repr_table, repr_M) for a representation variant.
    md3/shared -> (None,None,None) (indexed from U_* directly). Table variants -> a (M,d_model)
    frozen buffer (or None for the learned/trainable table). All indexed by unique MD tuple."""
    if cand_repr in ("md3", "shared"):
        return None, None, None
    keys = cache["keys"]                                        # (M,) sorted packed MD keys
    M = keys.numel()
    if cand_repr == "learned":
        return keys.to(device), None, M                        # model builds the trainable Embedding
    if cand_repr == "random":
        torch.manual_seed(seed + 777)
        return keys.to(device), (torch.randn(M, d_model) * 0.02).to(device), M
    if cand_repr == "mbgen1":
        cb = cache["mbgen_cb"]                                 # MBGen QAE codebook
        mid = cache["mbgen1_id"].long()
        if cb.shape[0] <= int(mid.max()):                     # stored (d_qae, num_codes) -> orient
            cb = cb.t().contiguous()
        raw = cb[mid]                                          # (M, d_qae)
        return keys.to(device), orth_lift(raw, d_model, seed + 1).to(device), M
    if cand_repr in ("tiger1", "tiger3"):
        cb = cache["tiger_cb"]; l = cache["tiger_l"].long()    # 3x(1024,d_t), (M,3)
        raw = cb[0][l[:, 0]]
        if cand_repr == "tiger3":                              # additive residual reconstruction
            raw = raw + cb[1][l[:, 1]] + cb[2][l[:, 2]]
        return keys.to(device), orth_lift(raw, d_model, seed + 2).to(device), M
    raise ValueError(f"unknown cand_repr {cand_repr}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dump", required=True,
                    help="TRAINING-split Stage-1 dump (reranker training data)")
    ap.add_argument("--val_dump", required=True,
                    help="EVALUATION-split Stage-1 dump (model selection / early stopping)")
    ap.add_argument("--test_dump", required=True,
                    help="TESTING-split Stage-1 dump (final reported number only)")
    ap.add_argument("--mdrqvae_ckpt",
                    default="/scratch/yw8866/logs/train/runs/2026-05-06/2048_1024/checkpoints/checkpoint_000_010000.ckpt")
    ap.add_argument("--centroid_cache", default="/scratch/yw8866/rec-tmall/mdrqvae_centroids.pt")
    ap.add_argument("--wb", type=float, default=1.0, help="behavior weight in the baseline union score")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.3,
                    help="applied to candidate features, cross-attention, and FFN")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=40, help="upper bound; early stopping ends sooner")
    ap.add_argument("--patience", type=int, default=5,
                    help="stop after this many epochs with no test t3@10 improvement")
    ap.add_argument("--min_delta", type=float, default=0.0, help="min t3@10 gain to count as improvement")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval_batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="/scratch/yw8866/rec-tmall/mfd_reranker.pt")
    ap.add_argument("--use_hard", action="store_true", help="add hard tuple-match engagement features")
    ap.add_argument("--use_soft", action="store_true", help="add soft cosine-match engagement features")
    ap.add_argument("--ablation", action="store_true",
                    help="ignore --use_hard/--use_soft; run all 4 configs (views / +hard / +soft / +both)")
    ap.add_argument("--cand_repr", default="md3", choices=REPR_VARIANTS,
                    help="candidate representation variant (single-run)")
    ap.add_argument("--repr_cache", default="/scratch/yw8866/rec-tmall/cand_repr_tables.pt",
                    help="cache from build_cand_repr_tables.py (needed for non-md3/shared variants)")
    ap.add_argument("--geom", default="md", choices=["md", "random"],
                    help="codebook geometry for views+soft-match. random = fixed random tables sized to\n"
                         "this arm's own SID widths (prior ablation: random == MD three-view), which is what\n"
                         "makes an MFD-vs-chain comparison identical in construction across different SID spaces")
    ap.add_argument("--sid_path", default="", help="semantic_ids.pt -> per-level widths (read, never assumed)")
    ap.add_argument("--max_train_events", type=int, default=0,
                    help="cap the number of GT-in-pool TRAIN events (equal-training-data setting; 0 = all)")
    ap.add_argument("--save_ranks", default="", help="persist base/reranked TEST rank arrays for cross-arm analysis")
    ap.add_argument("--ablation_repr", action="store_true",
                    help="sweep candidate representations (all available in the cache), engagement=+both fixed")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[init] device={device}", flush=True)

    tr = RerankData(args.train_dump, args.wb)
    va = RerankData(args.val_dump, args.wb)
    te = RerankData(args.test_dump, args.wb)
    d_model = int(te.meta.get("d_model", tr.mem.shape[2]))
    nb = int(te.meta.get("nb", 4))
    if args.geom == "random":
        # widths READ from the SID file (never hard-coded: a prior chain analysis was invalidated
        # by assuming MD's (2048,1024,1024) layout), else inferred from the dumps.
        if args.sid_path:
            o = torch.load(args.sid_path, map_location="cpu", weights_only=False)
            sid = o if torch.is_tensor(o) else next(v for v in o.values() if torch.is_tensor(v) and v.dim() == 2)
            if sid.shape[0] > sid.shape[1]:
                sid = sid.t().contiguous()
            widths = [int(sid[l].max()) + 1 for l in range(sid.shape[0])]
        else:
            widths = [int(max(tr.f.max(), va.f.max(), te.f.max())) + 1,
                      int(max(tr.t.max(), va.t.max(), te.t.max())) + 1,
                      int(max(tr.i.max(), va.i.max(), te.i.max())) + 1]
        print(f"[geom] RANDOM tables; per-level SID widths read = {widths}", flush=True)
        g = torch.Generator().manual_seed(args.seed + 4242)   # identical construction on both arms
        U_f = (torch.randn(widths[0], d_model, generator=g) * 0.02).to(device)
        U_t = (torch.randn(widths[1], d_model, generator=g) * 0.02).to(device)
        U_i = (torch.randn(widths[2], d_model, generator=g) * 0.02).to(device)
    else:
        U_f, U_t, U_i = load_geometry(args.mdrqvae_ckpt, args.centroid_cache, d_model,
                                      seed=args.seed, device=device)
    # sanity: every dumped SID code must be below its level width
    for nm, col, tab in (("l0", "f", U_f), ("l1", "t", U_t), ("l2", "i", U_i)):
        mx = max(int(getattr(x, col).max()) for x in (tr, va, te))
        assert mx < tab.shape[0], f"SID code {mx} at {nm} exceeds width {tab.shape[0]}"
        print(f"[geom] {nm}: max code {mx} < width {tab.shape[0]}  OK", flush=True)

    if args.max_train_events and args.max_train_events < tr.train_idx.numel():
        gsub = torch.Generator().manual_seed(args.seed + 99)
        sel = torch.randperm(tr.train_idx.numel(), generator=gsub)[:args.max_train_events]
        tr.train_idx = tr.train_idx[sel.sort().values]
        print(f"[data] TRAIN capped to {tr.train_idx.numel():,} GT-in-pool events (equal-data setting)", flush=True)

    # ---- config list: (name, cand_repr, use_hard, use_soft) ----
    # --ablation_repr : sweep candidate REPRESENTATIONS, engagement fixed at +both (this experiment).
    # --ablation      : sweep ENGAGEMENT features (views/+hard/+soft/+both), representation = md3.
    # else            : single run at (--cand_repr, --use_hard, --use_soft).
    cache = None
    if args.ablation_repr or args.cand_repr not in ("md3", "shared"):
        cache = torch.load(args.repr_cache, map_location="cpu", weights_only=False)
        print(f"[repr] cache {args.repr_cache}: M={cache['keys'].numel():,} tuples; "
              f"available = md3 shared learned random"
              f"{' mbgen1' if 'mbgen_cb' in cache else ''}"
              f"{' tiger1 tiger3' if 'tiger_cb' in cache else ''}", flush=True)

    if args.ablation_repr:
        variants = ["md3", "shared", "learned", "random"]
        if "mbgen_cb" in cache: variants.insert(2, "mbgen1")
        if "tiger_cb" in cache: variants[3:3] = ["tiger1", "tiger3"]
        configs = [(v, v, True, True) for v in variants]       # engagement = +both fixed
    elif args.ablation:
        configs = [("views", "md3", False, False), ("+hard", "md3", True, False),
                   ("+soft", "md3", False, True), ("+both", "md3", True, True)]
    else:
        configs = [("run", args.cand_repr, args.use_hard, args.use_soft)]
    if any(uh or us for _, _, uh, us in configs):
        assert tr.has_hist and va.has_hist and te.has_hist, \
            "engagement features requested but dumps lack hist_ids -- re-dump with mfd_dump_memory=true"

    sweeping = args.ablation or args.ablation_repr
    results = {}; added = {}
    for name, cand_repr, use_hard, use_soft in configs:
        print("\n" + "#" * 78 + f"\n# CONFIG: {name}  (repr={cand_repr} hard={use_hard} soft={use_soft})\n"
              + "#" * 78, flush=True)
        rk, rt, rM = make_repr(cand_repr, cache, d_model, args.seed, device) if cache is not None \
            else (None, None, None)
        torch.manual_seed(args.seed)   # reseed AFTER make_repr (its orthogonal lift perturbs RNG) so
        #                                the base reranker init is identical across configs -> fair
        model = MFDReranker(U_f, U_t, U_i, d_model, layers=args.layers, heads=args.heads,
                            drop=args.dropout, num_behaviors=nb, use_hard=use_hard, use_soft=use_soft,
                            cand_repr=cand_repr, repr_keys=rk, repr_table=rt, repr_M=rM).to(device)
        n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
        added[name] = model.repr_added_params
        print(f"[init] reranker trainable params={n_par:,}  (added by repr={model.repr_added_params:,})  "
              f"d_model={d_model} layers={args.layers} heads={args.heads} scalar_in={4 + model.n_hard + model.n_soft}",
              flush=True)
        save = args.save if not sweeping else args.save.replace(".pt", f"_{name.strip('+')}.pt")
        run_args = argparse.Namespace(**{**vars(args), "save": save})
        results[name] = train(model, tr, va, te, device, run_args)
        if args.save_ranks:
            # persist TEST ranks (base + reranked) so the cross-arm breakdown can group GTs by
            # event id / GT tuple / baseline rank band without recomputing anything.
            br3, bh3, br2, bh2 = results[name]["base_test_arrays"]
            rr3, rh3, rr2, rh2 = results[name]["best_test_arrays"]
            torch.save({"base_r3": br3, "base_h3": bh3, "base_r2": br2, "base_h2": bh2,
                        "rr3": rr3, "rh3": rh3, "rr2": rr2, "rh2": rh2,
                        "gt_b": te.gt_b, "gt_f": te.gt_f, "gt_t": te.gt_t, "gt_i": te.gt_i,
                        "has_gt": te.has_gt, "config": name,
                        "n_train_events": int(tr.train_idx.numel())},
                       args.save_ranks if not sweeping else args.save_ranks.replace(".pt", f"_{name.strip(chr(43))}.pt"))
            print(f"[ranks] saved -> {args.save_ranks}", flush=True)

    # ---- final comparison table (TEST at each config's best-val epoch) ----
    if sweeping:
        print("\n" + "=" * 78 + "\n" + ("REPRESENTATION" if args.ablation_repr else "ENGAGEMENT")
              + " ABLATION SUMMARY  (TEST @ best-val epoch; baseline = union_score)\n" + "=" * 78, flush=True)
        first = configs[0][0]
        report("BASELINE", te, *results[first]["base_test_arrays"])
        for name, *_ in configs:
            print(f"  [repr {name}: +{added[name]:,} trainable params]", flush=True)
            report(name, te, *results[name]["best_test_arrays"])


if __name__ == "__main__":
    main()