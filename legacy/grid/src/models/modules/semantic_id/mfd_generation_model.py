"""MFD (Modality-Factorized Decoding) multi-behavior generative recommender.

Structural companion to the §12g count-level result: given the shared/fusion code L0=f,
the text code L1=t and image code L2=i are near-independent (CMI(t;i|f) small vs H(t|f),
H(i|f)). MFD is the DECODER that spends that license — it drops the chain edge t -> i:

    chain      P(b,s|x) = P(b|x) P(f|x,b) P(t|x,b,f) P(i|x,b,f,t)
    factorized P(b,s|x) = P(b|x) P(f|x,b) P(t|x,b,f) P(i|x,b,f)      <- removed edge = the 1.7%

Everything EXCEPT the decoder factorization is inherited unchanged from
`SemanticIDMultiBehaviorEncoderDecoder`, so t1/t2/t3 metrics stay directly comparable to the
chain baselines (this isolates the decoder contribution):

  * Embeddings: the same concatenated [L0|L1|L2|behavior] table, sid_level_offsets /
    behavior_offset, and MD-RQ-VAE codebook_init injection. No new item/catalog embeddings.
  * Encoder: unchanged (flat interleaved [b,f,t,i,...], stride 4, use_item_pe still available).
  * Evaluator / focal / behavior_weighted_sid / data builders / DDP: unchanged.

The DECODER change (the only change):
  Teacher-forced decoder input per item is [BOS, behavior, f] — 3 positions, not 4.
      position  p0=BOS        p1=behavior      p2=shared code f  (= h_f)
      predicts  b @ p0        f @ p1           t AND i @ p2  (two heads on the SAME h_f)
  t and i are never fed as decoder inputs — they exist only as targets. Both modality heads
  (decoder_mlp[1] text, decoder_mlp[2] image) read the same hidden state h_f; this shared-
  state contention is the whole point. The data pipeline is untouched: the label is still the
  full [b,f,t,i] 4-tuple (t and i are needed as targets); MFD simply feeds the decoder the
  first two columns and reads all three output positions. So NextKTokenMasking stays next_k=4
  and the encoder stays stride 4 — only the teacher-forced decoder slice is 3 wide.

  Loss  L = CE_beh (focal if enabled) + CE_f + CE_t|f + CE_i|f      (one term re-parented)

Generation is a compositional beam with agreement scoring (§5):
  1. beam over (behavior, f): keep top-K_f (b,f) roots.
  2. for each root, one forward to h_f, read BOTH heads -> top-K_t text + top-K_i image.
  3. compose (f,t,i) in K_t x K_i, filter by the catalog constraint (the precomputed set of
     valid SID tuples in self.codebooks — replaces the chain's implicit validity-by-
     construction), and score each survivor
         score(f,t,i) = logP(f|.) + (1/tau) * [ logP(t|f,.) + logP(i|f,.) ]
     tau is the agreement temperature (frozen, inference-swept: tau>1 softens the
     independence assumption where the branches disagree, tau<1 sharpens mutual verification;
     tau=1 is exact product-of-experts). A learnable tau is intentionally omitted so the
     TRAINING path is byte-for-byte the chain baseline's — the ablation isolates the decoder.
  4. global top-K_pool across all roots -> the evaluator takes top-10 by score, as today.

Ablations this file supports via config:
  (a) chain vs factorized      : run the chain PBA/tiger baseline vs this class, same everything
  (b) agreement vs single-branch: mfd_score_text / mfd_score_image (drop one logP term)
  (c) tau sweep                : mfd_tau in {0.7, 1.0, 1.5}
  (d) K_t / K_i asymmetry       : mfd_top_k_text / mfd_top_k_image
"""

from __future__ import annotations

from typing import Optional, Tuple

import time
from collections import defaultdict

import torch

from src.models.modules.semantic_id.tiger_generation_model import (
    SemanticIDMultiBehaviorEncoderDecoder,
)


class MFDMultiBehaviorEncoderDecoder(SemanticIDMultiBehaviorEncoderDecoder):
    """Modality-Factorized-Decoding variant of the multi-behavior TIGER model.

    Requires num_hierarchies == 3 (shared / text / image). Mutually exclusive with use_bmtv.
    """

    def __init__(
        self,
        use_modality_factorized: bool = True,
        mfd_top_k_shared: int = 10,
        mfd_top_k_text: int = 10,
        mfd_top_k_image: int = 10,
        mfd_pool_size: int = 50,
        mfd_tau: float = 1.0,
        mfd_catalog_constraint: bool = True,
        mfd_score_text: bool = True,
        mfd_score_image: bool = True,
        mfd_behavior_marginalized: bool = True,
        mfd_behavior_weight: float = 1.0,
        mfd_audit_dump: bool = False,
        mfd_audit_path: str = "/scratch/yw8866/rec-tmall/mfd_audit_dump.pt",
        mfd_audit_max_events: int = 200000,
        mfd_audit_pv_cap: int = 100000,
        mfd_audit_pv_id: int = 0,
        mfd_dump_memory: bool = False,
        mfd_wide_catalog_pool: bool = False,
        mfd_branch_contrastive: float = 0.0,   # >0: weight of the WITHIN-BRANCH contrastive loss (see
                                               # _branch_contrastive). 0 = off -> training path unchanged.
        mfd_branch_temp: float = 1.0,          # temperature on the summed log-probs in that loss
        mfd_force_history: bool = False,   # keep the user's history roots in the top-K_f and history tuples in
                                           # the pool (wide path) -- same decoder states; see _audit_forward
        mfd_compose_debug: int = 0,   # >0: print composition-stage internals (effective K_t/K_i, candidates
                                      # composed per f-branch, survivors after every truncation, missing-GT
                                      # ranks) for the first N audit batches. Read from the code path, not config.
        mfd_learned_fusion: bool = False,
        mfd_fusion_weights=None,   # list[4] or path to a .pt with {"weights": tensor(4,)}: learned
                                   # weights over [logP(b),logP(f),logP(t|f),logP(i|f)]. Off -> hand-set.
        use_pmq: bool = False,
        use_query_f: bool = False,
        decoder_slot_type_emb: bool = True,
        mfd_warmstart_path: Optional[str] = None,
        use_cmir: bool = False,
        cmir_num_rounds: int = 1,
        cmir_estimate_top_m: int = 3,
        cmir_loss_weight: float = 1.0,
        cmir_refine_top_m: int = -1,
        cmir_scheduled_sampling: bool = False,
        cmir_ss_p0: float = 0.25,
        cmir_ss_decay_steps: int = 20000,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if self.num_hierarchies != 3:
            raise ValueError(
                f"MFD requires num_hierarchies == 3 (shared/text/image), got {self.num_hierarchies}"
            )
        if getattr(self, "use_bmtv", False):
            raise ValueError("use_modality_factorized and use_bmtv are mutually exclusive")
        if getattr(self, "use_semantic_align", False) or getattr(self, "use_semantic_rerank", False):
            raise ValueError(
                "use_semantic_align / use_semantic_rerank are chain-decoder features and are "
                "not supported under MFD (the item-summary hidden state layout differs). "
                "Leave them false for MFD runs."
            )
        if not (mfd_score_text or mfd_score_image):
            raise ValueError("at least one of mfd_score_text / mfd_score_image must be True")

        self.use_modality_factorized = use_modality_factorized
        self.mfd_top_k_shared = mfd_top_k_shared
        self.mfd_top_k_text = mfd_top_k_text
        self.mfd_top_k_image = mfd_top_k_image
        self.mfd_pool_size = mfd_pool_size
        self.mfd_tau = float(mfd_tau)
        self.mfd_catalog_constraint = mfd_catalog_constraint
        self.mfd_score_text = mfd_score_text
        self.mfd_score_image = mfd_score_image
        self.mfd_behavior_marginalized = mfd_behavior_marginalized
        self.mfd_behavior_weight = float(mfd_behavior_weight)

        # ---- audit dump (§0 audits: one instrumented test pass, three offline analyses) ----
        self.mfd_audit_dump = mfd_audit_dump
        self.mfd_audit_path = mfd_audit_path
        self.mfd_audit_max_events = mfd_audit_max_events
        self.mfd_audit_pv_cap = mfd_audit_pv_cap
        self.mfd_audit_pv_id = mfd_audit_pv_id
        # Stage-1 reranker candidate dump: when true, the audit dump ALSO stores the frozen
        # encoder memory (full history hidden states) + its mask per event, so the offline
        # listwise reranker (train_mfd_reranker.py) can cross-attend into it with no T5 loaded.
        # The per-behavior pools + 4 component log-probs the reranker needs are ALREADY dumped
        # by _audit_forward; this flag only adds the memory. Default False -> audit path is
        # byte-for-byte unchanged (baseline bit-identical when disabled).
        self.mfd_dump_memory = mfd_dump_memory
        # Wide-catalog pool (G5): when true, the audit pool is the top-K_pool by SID score over the
        # UNIQUE CATALOG TUPLES directly (equivalent to K_t=K_i=full head width, no t/i pruning),
        # scored in one vectorized pass -- avoids the infeasible dense K_f*K_t*K_i grid at 1024^2.
        # Only affects _audit_forward's pool construction; requires the catalog constraint on.
        self.mfd_wide_catalog_pool = mfd_wide_catalog_pool
        self.mfd_compose_debug = int(mfd_compose_debug)
        # History forcing (2026-09-27). t3@10 is dominated by repeats, and the lead is capped by how many
        # predictable events MFD's POOL holds beyond chain's. With factorized decoding, keeping a history
        # shared code among the K_f roots costs ONE root, and then every catalog tuple under it -- including
        # the user's own history items -- is scored by the full t/i heads for free; chain would need beam
        # slots at two sequential levels per item. Roots stay K_f (decoder states unchanged); forced history
        # tuples enter the pool with their TRUE generator scores.
        self.mfd_force_history = bool(mfd_force_history)
        # Within-branch contrastive objective (2026-09-25). Measured motivation: for the ground truths
        # MFD alone covers, the shared code is already right (f* median rank 1) but the text/image heads
        # are diffuse (t* rank 6, i* rank 3); the reranker rescues exactly the items whose t and i are
        # rank 0. The heads are trained by independent CE over ALL 1024 codes, but inference only ever
        # ranks CATALOG-VALID (t,i) pairs inside the GT's branch -- a train/inference mismatch. This loss
        # trains the heads on exactly that: softmax over every catalog tuple under f*, target = the GT.
        # MFD-specific: both full heads come from one forward pass, so all ~3,000 tuples in a branch are
        # scored by a gather at no extra decoder cost; chain would need a sequential decode per negative.
        self.mfd_branch_contrastive = float(mfd_branch_contrastive)
        self.mfd_branch_temp = float(mfd_branch_temp)
        self._branch_off = None
        if self.mfd_branch_contrastive > 0 and not mfd_catalog_constraint:
            raise ValueError("mfd_branch_contrastive needs the catalog (mfd_catalog_constraint=true)")
        self._audit_dbg_left = int(mfd_compose_debug)
        self._audit_eff = {}          # what the composition stage ACTUALLY used (filled by _audit_forward)
        # learned score fusion (Task 1): re-rank the SELECTED pool with fitted weights over the four
        # component log-probs. Pool SELECTION stays hand-set, so coverage@pool is invariant.
        self.mfd_learned_fusion = mfd_learned_fusion
        fw = mfd_fusion_weights
        if isinstance(fw, str):
            fw = torch.load(fw, map_location="cpu", weights_only=False)["weights"]
        fw = torch.tensor([1.0, 1.0, 1.0, 1.0]) if fw is None else torch.as_tensor(fw, dtype=torch.float)
        # NON-persistent: these weights come from config, not training, so they must NOT enter
        # the state_dict -- otherwise frozen checkpoints predating this switch fail to load
        # with: Missing key(s) in state_dict: "mfd_fusion_w".
        self.register_buffer("mfd_fusion_w", fw, persistent=False)
        self._cat_ftii = None
        self._audit_cols = defaultdict(list)
        self._audit_n = 0
        self._audit_pv = 0
        # wall-clock generation timing (GPU-synced): seconds spent inside _audit_forward and the
        # number of events pushed through it -> reported as sec / 1000 events in the dump meta.
        self._audit_gen_time = 0.0
        self._audit_gen_events = 0

        # Radix for packing a (f,t,i) tuple into one int64 key. Use the per-level codebook
        # widths so keys never collide (text/image both 1024, shared 2048).
        self._k0 = int(self.sid_level_sizes[0])   # shared / f  head width
        self._k1 = int(self.sid_level_sizes[1])   # text  / t  head width
        self._k2 = int(self.sid_level_sizes[2])   # image / i  head width

        # ---- catalog constraint: the sorted set of valid (f,t,i) keys ----
        # self.codebooks is (N_items, 3) already ([:,0]=shared, [:,1]=text, [:,2]=image).
        # Non-persistent buffer: moves with the module (.to(device)) but is NOT written into
        # checkpoints (it is fully derivable from semantic_ids.pt). ~5.4M int64 ≈ 43MB.
        if mfd_catalog_constraint:
            cb = self.codebooks.long()
            keys = (cb[:, 0] * self._k1 + cb[:, 1]) * self._k2 + cb[:, 2]
            keys = torch.unique(keys)  # sorted, deduped
            self.register_buffer("_mfd_catalog_keys", keys, persistent=False)
        else:
            self._mfd_catalog_keys = None

        # ---- Query slots (PMQ / Query-f): learnable query positions in the decoder block ----
        # Per-item decoder block is a slot list. Ordering is load-bearing: q_f is BEFORE f so
        # causal masking already prevents q_f from seeing the teacher-forced f*; q_t/q_i are
        # AFTER f so they condition on it (P(t|f), P(i|f)). Encoder is UNTOUCHED (stride 4).
        #   both false  -> [BOS, beh, f]                (current MFD; uses the inherited path)
        #   pmq only    -> [BOS, beh, f, q_t, q_i]      (each modality its own hidden state)
        #   full        -> [BOS, beh, q_f, f, q_t, q_i] (shared head its own capacity too)
        self.use_pmq = use_pmq
        self.use_query_f = use_query_f
        self.decoder_slot_type_emb = decoder_slot_type_emb
        self._slot_mode = use_pmq or use_query_f

        slots = ["bos", "beh"]
        if use_query_f:
            slots.append("q_f")
        slots.append("f")
        if use_pmq:
            slots += ["q_t", "q_i"]
        self._slots = slots
        self._slot_pos = {n: i for i, n in enumerate(slots)}
        # head-read positions (which hidden state each head reads)
        self._pos_beh = self._slot_pos["bos"]                                    # behavior @ h_bos
        self._pos_f = self._slot_pos["q_f"] if use_query_f else self._slot_pos["beh"]
        # (t/i are read at the LAST slot of their own pass -> no fixed canonical index needed)

        d = self.embedding_dim
        if use_query_f:
            self.query_f = torch.nn.Parameter(torch.randn(d) * 0.02)
        if use_pmq:
            self.query_t = torch.nn.Parameter(torch.randn(d) * 0.02)
            self.query_i = torch.nn.Parameter(torch.randn(d) * 0.02)
        # Slot identity signal (T5 has no absolute PE, only relative bias). Zero-init -> a warm
        # start begins at parity with the current checkpoint (adds nothing until trained).
        if self._slot_mode and decoder_slot_type_emb:
            self.decoder_slot_type_embedding = torch.nn.Embedding(len(slots), d)
            torch.nn.init.zeros_(self.decoder_slot_type_embedding.weight)
        else:
            self.decoder_slot_type_embedding = None

        # ---- CMIR (Cross-Modal Iterative Refinement) ----
        # Round 0: q_t -> P0(t|f), q_i -> P0(i|f) (masked from each other, as today). From P0 we
        # build soft expected code embeddings (top-m, prob-weighted, stop-grad). Round 1 injects
        # the OTHER modality's estimate into the query slot: q_t += proj_i2t(i_est) -> P1(t|f,i_est),
        # q_i += proj_t2i(t_est) -> P1(i|f,t_est), still one pass each (mutually masked). proj_* are
        # ZERO-init so at step 0 Round 1 == Round 0 (query_add=0) and the Run-C warm start survives.
        self.use_cmir = use_cmir
        self.cmir_num_rounds = cmir_num_rounds
        self.cmir_estimate_top_m = cmir_estimate_top_m
        self.cmir_loss_weight = float(cmir_loss_weight)
        self.cmir_refine_top_m = cmir_refine_top_m
        self.cmir_scheduled_sampling = cmir_scheduled_sampling
        self.cmir_ss_p0 = float(cmir_ss_p0)
        self.cmir_ss_decay_steps = int(cmir_ss_decay_steps)
        if use_cmir:
            if not use_pmq:
                raise ValueError("use_cmir requires use_pmq (needs the separate q_t / q_i passes)")
            if cmir_num_rounds != 1:
                raise ValueError("this CMIR implementation supports exactly 1 refinement round")
            self.cmir_proj_i2t = torch.nn.Linear(d, d)  # image estimate -> text-query injection
            self.cmir_proj_t2i = torch.nn.Linear(d, d)  # text estimate  -> image-query injection
            for p in (self.cmir_proj_i2t, self.cmir_proj_t2i):
                torch.nn.init.zeros_(p.weight); torch.nn.init.zeros_(p.bias)

        # ---- Warm start: load a prior checkpoint's weights (strict=False so the new query
        # params / slot-type emb stay at init). Distinct from ckpt_path (which RESUMES). ----
        if mfd_warmstart_path:
            sd = torch.load(mfd_warmstart_path, map_location="cpu", weights_only=False)
            sd = sd.get("state_dict", sd)
            missing, unexpected = self.load_state_dict(sd, strict=False)
            print(f"[MFD warmstart] {mfd_warmstart_path}: loaded (missing={len(missing)} "
                  f"unexpected={len(unexpected)}); new query/slot params kept at init.", flush=True)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _valid_tuple_mask(self, f: torch.Tensor, t: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        """Boolean mask (same shape as inputs) — is (f,t,i) a catalog SID tuple?"""
        if self._mfd_catalog_keys is None:
            return torch.ones_like(f, dtype=torch.bool)
        keys = self._mfd_catalog_keys
        cand = (f.long() * self._k1 + t.long()) * self._k2 + i.long()
        pos = torch.searchsorted(keys, cand)
        pos = pos.clamp(max=keys.numel() - 1)
        return keys[pos] == cand

    def _catalog_triples(self):
        """Unique catalog SID tuples as (CF, CT, CI), each (M_unique,) long on the module device.
        Unpacked from the packed catalog keys; cached. Used by the wide-catalog pool path."""
        if self._cat_ftii is None:
            keys = self._mfd_catalog_keys
            if keys is None:
                cb = self.codebooks.long()
                keys = torch.unique((cb[:, 0] * self._k1 + cb[:, 1]) * self._k2 + cb[:, 2])
            ci = keys % self._k2
            ct = (keys // self._k2) % self._k1
            cf = keys // (self._k1 * self._k2)
            self._cat_ftii = (cf.long(), ct.long(), ci.long())
        return self._cat_ftii

    def _branch_contrastive(self, t_logits, i_logits, f, t, i):
        """Cross-entropy over EVERY catalog tuple in the GT's own branch f*, target = the GT tuple.

        Catalog keys are sorted by (f, t, i), so each branch is one contiguous slice [off[f], off[f+1]);
        scoring it is a gather from the two log-softmaxes the heads already produced -- no extra forward.
        score(t_j, i_j) = [log P(t_j|f*) + log P(i_j|f*)] / temp, which is exactly the within-branch part
        of the inference score, so the heads are trained on the comparison inference actually makes."""
        CF, CT, CI = self._catalog_triples()
        if CF.device != t_logits.device:                  # cache built before the module moved: re-home it
            self._cat_ftii = tuple(x.to(t_logits.device) for x in (CF, CT, CI))
            CF, CT, CI = self._cat_ftii
        if self._branch_off is None or self._branch_off.device != CF.device:
            self._branch_off = torch.searchsorted(CF, torch.arange(self._k0 + 1, device=CF.device))
        off = self._branch_off
        f = f.to(CF.device); t = t.to(CF.device); i = i.to(CF.device)
        lo = off[f]; L = off[f + 1] - lo                                   # (B,) branch start / size
        Lmax = int(L.max())
        if Lmax == 0:
            return t_logits.sum() * 0.0
        ar = torch.arange(Lmax, device=CF.device)
        valid = ar.unsqueeze(0) < L.unsqueeze(1)                           # (B, Lmax)
        idx = (lo.unsqueeze(1) + ar.unsqueeze(0)).clamp(max=CF.numel() - 1)
        tt, ii = CT[idx], CI[idx]
        tlp = torch.log_softmax(t_logits.float(), dim=-1)
        ilp = torch.log_softmax(i_logits.float(), dim=-1)
        sc = (tlp.gather(1, tt) + ilp.gather(1, ii)) / self.mfd_branch_temp
        sc = sc.masked_fill(~valid, float("-inf"))
        hit = (tt == t.unsqueeze(1)) & (ii == i.unsqueeze(1)) & valid      # <=1 True per row (keys unique)
        has = hit.any(1)
        if not bool(has.any()):
            return t_logits.sum() * 0.0
        tgt = hit.float().argmax(1)
        return torch.nn.functional.cross_entropy(sc[has], tgt[has])

    # ------------------------------------------------------------------
    # Slot-mode (PMQ / Query-f) decoder plumbing. Single target-item block; the encoder holds
    # the history and is untouched. Cache-free (blocks are <=5 slots).
    #
    # ROBUSTNESS CHOICE: q_t and q_i are decoded in SEPARATE passes (each the last slot of its
    # own block) rather than one block with a custom q_i-/->q_t mask. Since the two never coexist
    # in a pass, they are conditionally independent given f BY CONSTRUCTION (exact P(t|f)·P(i|f)),
    # and every pass uses only STANDARD CAUSAL masking (attention_mask=None -> T5's own causal),
    # the same path the current MFD relies on. Cost: one extra tiny modality forward under PMQ.
    # ------------------------------------------------------------------
    def _slot_names(self, include_f, extra):
        """Canonical slot-name list for a pass. include_f adds the f slot; extra in
        {None,'q_t','q_i'} appends a modality query as the last (read) slot."""
        names = ["bos", "beh"]
        if self.use_query_f:
            names.append("q_f")
        if include_f:
            names.append("f")
        if extra is not None:
            names.append(extra)
        return names

    def _slot_embed(self, name, b, f, M, dev):
        table = self.get_embedding_table("decoder")
        if name == "bos":
            e = self.decoder.bos_token.expand(M, -1)
        elif name == "beh":
            e = table(b.long() + self.behavior_offset)
        elif name == "q_f":
            e = self.query_f.unsqueeze(0).expand(M, -1)
        elif name == "f":
            e = table(f.long() + self.sid_level_offsets[0])
        elif name == "q_t":
            e = self.query_t.unsqueeze(0).expand(M, -1)
        elif name == "q_i":
            e = self.query_i.unsqueeze(0).expand(M, -1)
        if self.decoder_slot_type_embedding is not None:
            # index by the slot's CANONICAL position so identity is stable across passes
            tid = torch.tensor(self._slot_pos[name], device=dev)
            e = e + self.decoder_slot_type_embedding(tid).unsqueeze(0)
        return e

    def _slot_run(self, names, b, f, enc, enc_mask, query_add=None):
        """Build the block from `names` and run the decoder with STANDARD causal masking.
        `query_add` (M, d), if given, is added to the LAST slot's input embedding (CMIR uses this
        to inject the cross-modal estimate into the modality query). Returns (M, len(names), d)."""
        M = b.shape[0]; dev = b.device
        slots = [self._slot_embed(n, b, f, M, dev) for n in names]
        if query_add is not None:
            slots[-1] = slots[-1] + query_add   # inject into the modality query BEFORE stacking
        emb = torch.stack(slots, dim=1)
        return self.decoder(
            sequence_embedding=emb,
            attention_mask=None,               # None -> T5 applies its own causal mask
            encoder_output=enc,
            encoder_attention_mask=enc_mask,
            use_cache=False,
            past_key_values=None,
        )  # (M, len(names), d)

    def _slot_f_logp(self, enc, enc_mask, b_ids):
        """Stage A: block [BOS, beh, (q_f)] (no f slot) -> f logits @ last slot, behavior @ BOS.
        b_ids: (M,). Returns f_logits (M,k0), beh_logits (M,nb)."""
        names = self._slot_names(include_f=False, extra=None)
        h = self._slot_run(names, b_ids, None, enc, enc_mask)
        return self.decoder.decoder_mlp[0](h[:, -1]), self.behavior_mlp(h[:, 0])

    def _slot_modality(self, enc, enc_mask, b_sel, f_sel):
        """Stage B: log P(t|f), log P(i|f). PMQ -> two passes ([...,f,q_t] and [...,f,q_i]),
        else one pass ([...,f]) whose h_f feeds both heads. Returns (B,Kf,k1),(B,Kf,k2)."""
        B, Kf = f_sel.shape
        b = b_sel.reshape(B * Kf); f = f_sel.reshape(B * Kf)
        enc_rep = enc.repeat_interleave(Kf, 0); mask_rep = enc_mask.repeat_interleave(Kf, 0)
        if self.use_pmq:
            h_t = self._slot_run(self._slot_names(True, "q_t"), b, f, enc_rep, mask_rep)[:, -1]
            h_i = self._slot_run(self._slot_names(True, "q_i"), b, f, enc_rep, mask_rep)[:, -1]
        else:
            h = self._slot_run(self._slot_names(True, None), b, f, enc_rep, mask_rep)[:, -1]
            h_t = h_i = h
        t_logp = torch.log_softmax(self.decoder.decoder_mlp[1](h_t), dim=-1)
        i_logp = torch.log_softmax(self.decoder.decoder_mlp[2](h_i), dim=-1)
        return t_logp.reshape(B, Kf, -1), i_logp.reshape(B, Kf, -1)

    def _stageA_all_behaviors(self, enc, enc_mask, B):
        """Behavior log-probs (B,nb) + f log-probs under every behavior (B,nb,k0)."""
        nb = self.num_behaviors; dev = enc.device
        beh_ids = torch.arange(nb, device=dev)
        if self._slot_mode:
            b_rep = beh_ids.view(1, nb).expand(B, nb).reshape(B * nb)
            f_logits, beh_logits = self._slot_f_logp(
                enc.repeat_interleave(nb, 0), enc_mask.repeat_interleave(nb, 0), b_rep
            )
            f_logp = torch.log_softmax(f_logits, dim=-1).reshape(B, nb, self._k0)
            # h_bos is identical across the behavior branches (BOS precedes the beh slot) -> take branch 0
            beh_logp = torch.log_softmax(beh_logits.reshape(B, nb, -1)[:, 0], dim=-1)
            return beh_logp, f_logp
        dec0 = self.decoder_forward_pass(
            future_ids=None, encoder_output=enc, attention_mask_for_encoder=enc_mask, use_cache=False
        )
        beh_logp = torch.log_softmax(self.behavior_mlp(dec0[:, -1, :]), dim=-1)
        fut_b = beh_ids.view(1, nb, 1).expand(B, nb, 1).reshape(B * nb, 1)
        dec1 = self.decoder_forward_pass(
            future_ids=fut_b, encoder_output=enc.repeat_interleave(nb, 0),
            attention_mask_for_encoder=enc_mask.repeat_interleave(nb, 0), use_cache=False,
        )
        f_logp = torch.log_softmax(
            self.decoder.decoder_mlp[0](dec1[:, -1, :]), dim=-1
        ).reshape(B, nb, self._k0)
        return beh_logp, f_logp

    # ------------------------------------------------------------------
    # CMIR (Cross-Modal Iterative Refinement)
    # ------------------------------------------------------------------
    def _soft_estimate(self, logp, level, gt=None, ss_mask=None):
        """Soft expected code embedding from a modality log-prob (top-m, prob-weighted), at the
        given SID level (1=text, 2=image). Stop-gradient by default. If scheduled sampling is
        active (ss_mask), substitute the GT code embedding where masked. Returns (M, d)."""
        p = logp.exp()
        m = min(self.cmir_estimate_top_m, p.size(-1))
        vals, idx = p.topk(m, dim=-1)                                  # (M, m)
        w = vals / vals.sum(-1, keepdim=True).clamp(min=1e-9)          # renormalise over top-m
        table = self.get_embedding_table("decoder")
        off = self.sid_level_offsets[level]
        est = (w.unsqueeze(-1) * table(idx + off)).sum(1).detach()     # (M, d), stop-grad
        if gt is not None and ss_mask is not None and bool(ss_mask.any()):
            gt_emb = table(gt.long() + off).detach()
            est = torch.where(ss_mask.unsqueeze(-1), gt_emb, est)
        return est

    def _cmir_modality(self, enc, enc_mask, b_sel, f_sel, score_root=None):
        """Inference Stage B under CMIR: Round 0 -> estimates -> Round 1. Returns the ROUND-1
        (B,Kf,k1)/(B,Kf,k2) log-probs used for composition/scoring."""
        B, Kf = f_sel.shape
        b = b_sel.reshape(-1); f = f_sel.reshape(-1)
        enc_rep = enc.repeat_interleave(Kf, 0); mask_rep = enc_mask.repeat_interleave(Kf, 0)
        nt = self._slot_names(True, "q_t"); ni = self._slot_names(True, "q_i")
        # Round 0 (masked from each other)
        h_t0 = self._slot_run(nt, b, f, enc_rep, mask_rep)[:, -1]
        h_i0 = self._slot_run(ni, b, f, enc_rep, mask_rep)[:, -1]
        t_lp0 = torch.log_softmax(self.decoder.decoder_mlp[1](h_t0), -1)
        i_lp0 = torch.log_softmax(self.decoder.decoder_mlp[2](h_i0), -1)
        i_est = self._soft_estimate(i_lp0, 2)   # image estimate -> conditions q_t
        t_est = self._soft_estimate(t_lp0, 1)   # text estimate  -> conditions q_i
        # Round 1 (each still one pass, mutually masked; only cross-round info flows)
        h_t1 = self._slot_run(nt, b, f, enc_rep, mask_rep, query_add=self.cmir_proj_i2t(i_est))[:, -1]
        h_i1 = self._slot_run(ni, b, f, enc_rep, mask_rep, query_add=self.cmir_proj_t2i(t_est))[:, -1]
        t_lp1 = torch.log_softmax(self.decoder.decoder_mlp[1](h_t1), -1)
        i_lp1 = torch.log_softmax(self.decoder.decoder_mlp[2](h_i1), -1)
        # optional cost cap: only the top-M f branches (by Round-0 score) keep Round 1
        M = self.cmir_refine_top_m
        if score_root is not None and 0 < M < Kf:
            sel = score_root.topk(M, dim=-1).indices
            refine = torch.zeros(B, Kf, dtype=torch.bool, device=f.device)
            refine.scatter_(1, sel, True)
            refine = refine.reshape(-1, 1)
            t_lp1 = torch.where(refine, t_lp1, t_lp0)
            i_lp1 = torch.where(refine, i_lp1, i_lp0)
        return t_lp1.reshape(B, Kf, -1), i_lp1.reshape(B, Kf, -1)

    def _modality_heads(
        self,
        encoder_output: torch.Tensor,
        encoder_mask: torch.Tensor,
        b_sel: torch.Tensor,   # (B, Kf)
        f_sel: torch.Tensor,   # (B, Kf)
        score_root: torch.Tensor = None,   # (B, Kf) for the CMIR top-M-branch cap; None -> refine all
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One cache-free forward per (b,f) root -> (text_logp, img_logp), (B,Kf,k1)/(B,Kf,k2).
        CMIR -> Round-1 two-round refinement; slot mode -> query-slot block; else plain [BOS,b,f]."""
        if self.use_cmir:
            return self._cmir_modality(encoder_output, encoder_mask, b_sel, f_sel, score_root)
        if self._slot_mode:
            return self._slot_modality(encoder_output, encoder_mask, b_sel, f_sel)
        B, Kf = f_sel.shape
        fut_bf = torch.stack([b_sel, f_sel], dim=-1).reshape(B * Kf, 2)  # (B*Kf, 2)
        enc_rep = encoder_output.repeat_interleave(Kf, dim=0)
        mask_rep = encoder_mask.repeat_interleave(Kf, dim=0)
        dec = self.decoder_forward_pass(
            future_ids=fut_bf,
            encoder_output=enc_rep,
            attention_mask_for_encoder=mask_rep,
            use_cache=False,
        )  # (B*Kf, 3, d)
        h_f = dec[:, -1, :]  # after f -> the shared item-summary hidden state
        t_logp = torch.log_softmax(self.decoder.decoder_mlp[1](h_f), dim=-1)  # (B*Kf, k1)
        i_logp = torch.log_softmax(self.decoder.decoder_mlp[2](h_f), dim=-1)  # (B*Kf, k2)
        return t_logp.reshape(B, Kf, -1), i_logp.reshape(B, Kf, -1)

    def _compose_and_pool(
        self,
        b_sel: torch.Tensor,     # (B, Kf)
        f_sel: torch.Tensor,     # (B, Kf)
        score_root: torch.Tensor,  # (B, Kf)  logP(b)+logP(f|b)  (t2: logP(f|b_gt))
        t_logp: torch.Tensor,    # (B, Kf, k1)
        i_logp: torch.Tensor,    # (B, Kf, k2)
        lp_beh_root: torch.Tensor = None,   # (B, Kf) logP(b) alone -- only for learned-fusion re-rank
        lp_f_root: torch.Tensor = None,     # (B, Kf) logP(f|b) alone -- only for learned-fusion re-rank
    ):
        """Compose top-Kt x top-Ki modality candidates per root, catalog-filter, score, and
        return the global top-K_pool (b, f, t, i) tuples + their scores per example.

        Pool SELECTION always uses the hand-set additive score `comp`. When mfd_learned_fusion is on
        and the split behavior/f components are provided, the RETURNED score of the selected pool is
        re-computed as w.[logP(b),logP(f),logP(t|f),logP(i|f)] -- selection unchanged, so coverage
        is invariant; only the ordering of the fixed pool changes."""
        B, Kf = f_sel.shape
        Kt = min(self.mfd_top_k_text, t_logp.size(-1))
        Ki = min(self.mfd_top_k_image, i_logp.size(-1))
        inv_tau = 1.0 / self.mfd_tau

        tv, ti = t_logp.topk(Kt, dim=-1)   # (B, Kf, Kt)
        iv, ii = i_logp.topk(Ki, dim=-1)   # (B, Kf, Ki)

        # additive log-scores over the Kt x Ki grid per root
        w_t = 1.0 if self.mfd_score_text else 0.0
        w_i = 1.0 if self.mfd_score_image else 0.0
        comp = (
            score_root.view(B, Kf, 1, 1)
            + inv_tau * w_t * tv.view(B, Kf, Kt, 1)
            + inv_tau * w_i * iv.view(B, Kf, 1, Ki)
        )  # (B, Kf, Kt, Ki)

        b_cand = b_sel.view(B, Kf, 1, 1).expand(B, Kf, Kt, Ki)
        f_cand = f_sel.view(B, Kf, 1, 1).expand(B, Kf, Kt, Ki)
        t_cand = ti.view(B, Kf, Kt, 1).expand(B, Kf, Kt, Ki)
        i_cand = ii.view(B, Kf, 1, Ki).expand(B, Kf, Kt, Ki)

        M = Kf * Kt * Ki
        comp = comp.reshape(B, M)
        b_cand = b_cand.reshape(B, M)
        f_cand = f_cand.reshape(B, M)
        t_cand = t_cand.reshape(B, M)
        i_cand = i_cand.reshape(B, M)

        # catalog constraint: invalid compositions are discarded (score -> -inf)
        if self.mfd_catalog_constraint:
            valid = self._valid_tuple_mask(f_cand, t_cand, i_cand)
            comp = comp.masked_fill(~valid, float("-inf"))

        Kpool = min(self.mfd_pool_size, M)
        score_pool, pidx = comp.topk(Kpool, dim=-1)  # (B, Kpool)  (hand-set selection -- unchanged)
        gather = lambda x: torch.gather(x, 1, pidx)

        # learned-fusion RE-RANK of the fixed pool (coverage-invariant): needs the split components.
        if self.mfd_learned_fusion and lp_beh_root is not None and lp_f_root is not None:
            lb = gather(lp_beh_root.view(B, Kf, 1, 1).expand(B, Kf, Kt, Ki).reshape(B, M))
            lf = gather(lp_f_root.view(B, Kf, 1, 1).expand(B, Kf, Kt, Ki).reshape(B, M))
            lt = gather(tv.view(B, Kf, Kt, 1).expand(B, Kf, Kt, Ki).reshape(B, M))
            li = gather(iv.view(B, Kf, 1, Ki).expand(B, Kf, Kt, Ki).reshape(B, M))
            w = self.mfd_fusion_w.to(lb.device)
            fused = w[0] * lb + w[1] * lf + w[2] * lt + w[3] * li
            score_pool = fused.masked_fill(score_pool == float("-inf"), float("-inf"))  # keep invalid at -inf
        return gather(b_cand), gather(f_cand), gather(t_cand), gather(i_cand), score_pool

    def _cmir_model_step(self, model_input, fut_ids):
        """Teacher-forced CMIR training. Round 0 gives beh/f/t0/i0 and the model's OWN estimates;
        Round 1 conditions each modality query on the other's estimate (stop-grad, optional
        scheduled sampling to GT). Loss = CE_beh + CE_f + CE_t0 + CE_i0 + lambda*(CE_t1 + CE_i1)."""
        inputs = {
            self.feature_to_model_input_map.get(k, k): v
            for k, v in model_input.transformed_sequences.items()
        }
        enc, enc_mask = self.encoder_forward_pass(
            attention_mask=model_input.mask, input_ids=inputs["input_ids"], user_id=inputs.get("user_id")
        )
        b = fut_ids[:, 0].long(); f = fut_ids[:, 1].long()
        t = fut_ids[:, 2].long(); i = fut_ids[:, 3].long()
        nt = self._slot_names(True, "q_t"); ni = self._slot_names(True, "q_i")
        # ---- Round 0 ----
        h_t0 = self._slot_run(nt, b, f, enc, enc_mask)
        h_i0 = self._slot_run(ni, b, f, enc, enc_mask)
        beh_logits = self.behavior_mlp(h_t0[:, self._pos_beh])
        f_logits = self.decoder.decoder_mlp[0](h_t0[:, self._pos_f])
        t0_logits = self.decoder.decoder_mlp[1](h_t0[:, -1])
        i0_logits = self.decoder.decoder_mlp[2](h_i0[:, -1])
        # ---- estimates (own Round-0 predictions; optional scheduled sampling to GT) ----
        ss = None
        if self.cmir_scheduled_sampling and self.training:
            p = self.cmir_ss_p0 * max(0.0, 1.0 - self.global_step / max(1, self.cmir_ss_decay_steps))
            ss = torch.rand(b.shape[0], device=b.device) < p
        i_est = self._soft_estimate(torch.log_softmax(i0_logits, -1), 2, gt=i, ss_mask=ss)
        t_est = self._soft_estimate(torch.log_softmax(t0_logits, -1), 1, gt=t, ss_mask=ss)
        # ---- Round 1 ----
        h_t1 = self._slot_run(nt, b, f, enc, enc_mask, query_add=self.cmir_proj_i2t(i_est))
        h_i1 = self._slot_run(ni, b, f, enc, enc_mask, query_add=self.cmir_proj_t2i(t_est))
        t1_logits = self.decoder.decoder_mlp[1](h_t1[:, -1])
        i1_logits = self.decoder.decoder_mlp[2](h_i1[:, -1])
        # ---- loss ----
        beh_loss_fn = self.behavior_loss_function or self.loss_function
        loss = beh_loss_fn(input=beh_logits, target=b)
        lw = self.cmir_loss_weight
        if self.behavior_weighted_sid:
            w = self.sid_behavior_weights.to(b.device)[b]; denom = w.sum().clamp(min=1e-8)
            ce = lambda lg, tg: (
                w * torch.nn.functional.cross_entropy(lg, tg, reduction="none")
            ).sum() / denom
            loss = (loss + ce(f_logits, f) + ce(t0_logits, t) + ce(i0_logits, i)
                    + lw * (ce(t1_logits, t) + ce(i1_logits, i)))
        else:
            L = self.loss_function
            loss = (loss + L(input=f_logits, target=f)
                    + L(input=t0_logits, target=t) + L(input=i0_logits, target=i)
                    + lw * (L(input=t1_logits, target=t) + L(input=i1_logits, target=i)))
        return h_t1, loss

    # ------------------------------------------------------------------
    # Training: teacher-forced 3-position decoder, two modality heads on h_f
    # ------------------------------------------------------------------
    def _slot_model_step(self, model_input, fut_ids):
        """Teacher-forced training for the query-slot decoder. Full block [BOS, beh, (q_f), f,
        (q_t), (q_i)] with f fed at the f slot; each head reads its own slot's hidden state.
        Preserves the MFD factorization P(f|hist,b)·P(t|hist,b,f)·P(i|hist,b,f)."""
        inputs = {
            self.feature_to_model_input_map.get(k, k): v
            for k, v in model_input.transformed_sequences.items()
        }
        enc, enc_mask = self.encoder_forward_pass(
            attention_mask=model_input.mask, input_ids=inputs["input_ids"], user_id=inputs.get("user_id")
        )
        b = fut_ids[:, 0].long(); f = fut_ids[:, 1].long()
        t = fut_ids[:, 2].long(); i = fut_ids[:, 3].long()
        # f-head reads q_f (or beh) at its canonical index; the block prefix preserves those
        # indices, and the modality head reads the LAST slot (q_t/q_i, or f when not PMQ).
        if self.use_pmq:
            h = self._slot_run(self._slot_names(True, "q_t"), b, f, enc, enc_mask)
            h_i = self._slot_run(self._slot_names(True, "q_i"), b, f, enc, enc_mask)
            beh_logits = self.behavior_mlp(h[:, self._pos_beh])
            f_logits = self.decoder.decoder_mlp[0](h[:, self._pos_f])
            t_logits = self.decoder.decoder_mlp[1](h[:, -1])       # q_t pass
            i_logits = self.decoder.decoder_mlp[2](h_i[:, -1])     # q_i pass
        else:
            h = self._slot_run(self._slot_names(True, None), b, f, enc, enc_mask)
            beh_logits = self.behavior_mlp(h[:, self._pos_beh])
            f_logits = self.decoder.decoder_mlp[0](h[:, self._pos_f])
            t_logits = self.decoder.decoder_mlp[1](h[:, -1])  # h_f
            i_logits = self.decoder.decoder_mlp[2](h[:, -1])
        beh_loss_fn = self.behavior_loss_function or self.loss_function
        loss = beh_loss_fn(input=beh_logits, target=b)
        if self.behavior_weighted_sid:
            w = self.sid_behavior_weights.to(b.device)[b]; denom = w.sum().clamp(min=1e-8)
            ce = lambda lg, tg: (
                w * torch.nn.functional.cross_entropy(lg, tg, reduction="none")
            ).sum() / denom
            loss = loss + ce(f_logits, f) + ce(t_logits, t) + ce(i_logits, i)
        else:
            loss = (loss + self.loss_function(input=f_logits, target=f)
                    + self.loss_function(input=t_logits, target=t)
                    + self.loss_function(input=i_logits, target=i))
        if self.mfd_branch_contrastive > 0:
            loss = loss + self.mfd_branch_contrastive * self._branch_contrastive(t_logits, i_logits, f, t, i)
        return h, loss

    def model_step(self, model_input, label_data=None):
        if label_data is None:
            generated_ids, marginal_probs = self.generate_multibehavior(
                attention_mask=model_input.mask,
                **{
                    self.feature_to_model_input_map.get(k, k): v
                    for k, v in model_input.transformed_sequences.items()
                },
            )
            return generated_ids, 0

        fut_ids = None
        for label in label_data.labels:
            fut_ids = label_data.labels[label].reshape(model_input.mask.size(0), -1)
        # fut_ids: (B, 1+H) = [behavior, f, t, i]

        if self.use_cmir:
            if self.mfd_branch_contrastive > 0:
                raise ValueError("mfd_branch_contrastive is not wired into the CMIR training path")
            return self._cmir_model_step(model_input, fut_ids)
        if self._slot_mode:
            return self._slot_model_step(model_input, fut_ids)

        # Decoder INPUT is only [behavior, f]; BOS is prepended inside decoder_forward_pass,
        # giving decoder positions [BOS, b, f] -> output (B, 3, d). Read ALL three positions
        # (do NOT drop the last, unlike the chain model): p0->behavior, p1->f, p2=h_f->t & i.
        dec_in = fut_ids[:, :2]
        model_output = self.forward(
            attention_mask_encoder=model_input.mask,
            future_ids=dec_in,
            **{
                self.feature_to_model_input_map.get(k, k): v
                for k, v in model_input.transformed_sequences.items()
            },
        )  # (B, 3, d)

        h_bos, h_b, h_f = model_output[:, 0], model_output[:, 1], model_output[:, 2]

        # behavior loss @ p0 (focal if enabled)
        behavior_target = fut_ids[:, 0].long()
        behavior_loss_fn = self.behavior_loss_function or self.loss_function
        loss = behavior_loss_fn(input=self.behavior_mlp(h_bos), target=behavior_target)

        # SID losses: f @ p1 (from h_b), t & i @ p2 (both from the SAME h_f)
        f_logits = self.decoder.decoder_mlp[0](h_b)
        t_logits = self.decoder.decoder_mlp[1](h_f)
        i_logits = self.decoder.decoder_mlp[2](h_f)
        f_tgt = fut_ids[:, 1].long()
        t_tgt = fut_ids[:, 2].long()
        i_tgt = fut_ids[:, 3].long()

        if self.behavior_weighted_sid:
            w = self.sid_behavior_weights.to(behavior_target.device)[behavior_target]  # (B,)
            denom = w.sum().clamp(min=1e-8)
            ce = lambda logit, tgt: (
                w * torch.nn.functional.cross_entropy(logit, tgt, reduction="none")
            ).sum() / denom
            loss = loss + ce(f_logits, f_tgt) + ce(t_logits, t_tgt) + ce(i_logits, i_tgt)
        else:
            loss = (
                loss
                + self.loss_function(input=f_logits, target=f_tgt)
                + self.loss_function(input=t_logits, target=t_tgt)
                + self.loss_function(input=i_logits, target=i_tgt)
            )
        if self.mfd_branch_contrastive > 0:
            loss = loss + self.mfd_branch_contrastive * self._branch_contrastive(
                t_logits, i_logits, f_tgt, t_tgt, i_tgt)

        return model_output, loss

    # ------------------------------------------------------------------
    # Free generation (t1 / t3): compositional beam with agreement scoring
    # ------------------------------------------------------------------
    def generate_multibehavior(self, attention_mask, input_ids, user_id=None):
        encoder_output, encoder_mask = self.encoder_forward_pass(
            attention_mask=attention_mask, input_ids=input_ids, user_id=user_id
        )
        B = input_ids.size(0)
        device = encoder_output.device
        nb = self.num_behaviors

        # ---- Stage A: behavior log-probs (B,nb) + f log-probs under every behavior (B,nb,k0) ----
        # (routes to the query-slot decoder when use_pmq/use_query_f, else the plain [BOS,b,f] path)
        beh_logp, f_logp = self._stageA_all_behaviors(encoder_output, encoder_mask, B)

        # ---- roots: pick (b, f) roots and their w_b*logP(b)+logP(f|b) score ----
        w_b = self.mfd_behavior_weight
        if self.mfd_behavior_marginalized:
            # Behavior-marginalized pooling (the stronger version): DO NOT commit one behavior
            # in the beam. Give EACH of the nb behaviors its OWN top-Kf f-roots, so the per-
            # behavior SID retrieval that makes t2 strong runs for every behavior; then rank the
            # union of all nb pools by w_b*logP(b) + SID_score(.|b). Guarantees every behavior a
            # slice of the beam instead of letting one behavior crowd out the shared top-Kf.
            # Cost: nb x the f-branch modality forward (heads are cheap).
            kf = min(self.mfd_top_k_shared, self._k0)
            sc_f, f_idx = f_logp.topk(kf, dim=-1)                # (B, nb, kf)  logP(f|b)
            score_root = w_b * beh_logp.unsqueeze(-1) + sc_f     # (B, nb, kf)
            b_sel = (
                torch.arange(nb, device=device).view(1, nb, 1).expand(B, nb, kf)
            )                                                    # (B, nb, kf)
            f_sel = f_idx                                        # (B, nb, kf)
            Kf_tot = nb * kf
            b_sel = b_sel.reshape(B, Kf_tot)
            f_sel = f_sel.reshape(B, Kf_tot)
            score_root = score_root.reshape(B, Kf_tot)
        else:
            # Behavior-committed pooling: one shared beam over (b,f) -> global top-Kf roots.
            kf = min(self.mfd_top_k_shared, self._k0 * nb)
            joint = w_b * beh_logp.unsqueeze(-1) + f_logp        # (B, nb, k0)
            score_root, idx = joint.reshape(B, nb * self._k0).topk(kf, dim=-1)  # (B, kf)
            b_sel = idx // self._k0
            f_sel = idx % self._k0

        # split logP(b) / logP(f|b) per root -- only needed for the learned-fusion re-rank
        lp_beh_root = lp_f_root = None
        if self.mfd_learned_fusion:
            lp_beh_root = torch.gather(beh_logp, 1, b_sel)          # (B, Kf_tot) logP(b)
            lp_f_root = score_root - w_b * lp_beh_root              # (B, Kf_tot) logP(f|b)

        # ---- steps 2/2': both modality heads on h_f, compose, filter, pool ----
        t_logp, i_logp = self._modality_heads(encoder_output, encoder_mask, b_sel, f_sel, score_root)
        b, f, t, i, score = self._compose_and_pool(b_sel, f_sel, score_root, t_logp, i_logp,
                                                   lp_beh_root=lp_beh_root, lp_f_root=lp_f_root)
        generated_ids = torch.stack([b, f, t, i], dim=-1)  # (B, Kpool, 4)
        return generated_ids, score  # (B, Kpool, 1+H), (B, Kpool)

    # ------------------------------------------------------------------
    # Conditioned generation (t2): behavior forced to GT, SID-only output
    # ------------------------------------------------------------------
    def generate_conditioned(self, attention_mask, input_ids, gt_behavior, user_id=None):
        encoder_output, encoder_mask = self.encoder_forward_pass(
            attention_mask=attention_mask, input_ids=input_ids, user_id=user_id
        )
        B = input_ids.size(0)
        Kf = min(self.mfd_top_k_shared, self._k0)

        # f log-probs conditioned on the GT behavior (slot decoder when in slot mode)
        if self._slot_mode:
            f_logits, _ = self._slot_f_logp(encoder_output, encoder_mask, gt_behavior.long())
            f_logp = torch.log_softmax(f_logits, dim=-1)  # (B, k0)
        else:
            dec1 = self.decoder_forward_pass(
                future_ids=gt_behavior.long().view(B, 1),
                encoder_output=encoder_output,
                attention_mask_for_encoder=encoder_mask,
                use_cache=False,
            )  # (B, 2, d)
            f_logp = torch.log_softmax(self.decoder.decoder_mlp[0](dec1[:, -1, :]), dim=-1)  # (B, k0)
        score_root, f_sel = f_logp.topk(Kf, dim=-1)  # (B, Kf)
        b_sel = gt_behavior.long().view(B, 1).expand(B, Kf)

        t_logp, i_logp = self._modality_heads(encoder_output, encoder_mask, b_sel, f_sel, score_root)
        _, f, t, i, score = self._compose_and_pool(b_sel, f_sel, score_root, t_logp, i_logp)
        generated_ids = torch.stack([f, t, i], dim=-1)  # (B, Kpool, H)  SIDs only
        return generated_ids, score

    # ==================================================================
    # AUDIT DUMP (mfd_audit_dump=true): one instrumented test pass captures, per event,
    # everything the three §0 audits need. Nothing is recomputed later.
    #   0a pool routing : per-behavior pools (SID_score sans behavior term) + beh_logp -> any
    #                     w_b union/routed re-ranking offline (committed is the cited baseline).
    #   0b contention   : teacher-forced GT-f branch reads (rank of t*,i* in the head softmax).
    #   0c coverage/rank: f* full rank (GT behavior), t*/i* ranks at GT f, and the pools.
    # ==================================================================
    @torch.no_grad()
    def _audit_forward(self, mask, input_ids, user_id, labels):
        enc, emask = self.encoder_forward_pass(
            attention_mask=mask, input_ids=input_ids, user_id=user_id
        )
        B = input_ids.size(0)
        dev = enc.device
        nb = self.num_behaviors
        ar = torch.arange(B, device=dev)
        gt_b = labels[:, 0].long(); gt_f = labels[:, 1].long()
        gt_t = labels[:, 2].long(); gt_i = labels[:, 3].long()

        # DIRECT catalog-validity of the GT tuple (the unit test): GT is a real tokenized item,
        # so it MUST be in the catalog. If any GT is absent, the constraint hash is wrong. This
        # separates a true catalog bug from mere pool truncation (composed+valid but rank>Kpool).
        gt_in_catalog = self._valid_tuple_mask(gt_f, gt_t, gt_i)  # (B,) bool

        # behavior + f log-probs under every behavior (slot-aware Stage A)
        beh_logp, f_logp = self._stageA_all_behaviors(enc, emask, B)  # (B,nb), (B,nb,k0)

        # f* rank within the GT-behavior shared softmax (full rank, for 0c near-misses)
        f_logp_gt = f_logp[ar, gt_b]                       # (B, k0)
        f_star_logp = f_logp_gt[ar, gt_f]                  # (B,)
        f_star_rank = (f_logp_gt > f_star_logp.unsqueeze(1)).sum(1)  # (B,) 0-based

        # teacher-forced GT-f branch reads (t & i read at their slots given the GT f) -> 0b/0c.
        # Reuse _modality_heads (slot-aware) with a single (b*, f*) root.
        t_lp1, i_lp1 = self._modality_heads(enc, emask, gt_b.view(B, 1), gt_f.view(B, 1))
        t_lp_gt = t_lp1[:, 0, :]  # (B, k1)  already log-softmax
        i_lp_gt = i_lp1[:, 0, :]  # (B, k2)
        tf_t_logp = t_lp_gt[ar, gt_t]; tf_t_rank = (t_lp_gt > tf_t_logp.unsqueeze(1)).sum(1)
        tf_i_logp = i_lp_gt[ar, gt_i]; tf_i_rank = (i_lp_gt > tf_i_logp.unsqueeze(1)).sum(1)

        # per-behavior pools (top-Kpool per behavior, catalog-filtered). SID_score EXCLUDES the
        # behavior term so 0a can add w_b*logP(b) for ANY w_b offline.
        Kf = min(self.mfd_top_k_shared, self._k0)
        Kt = min(self.mfd_top_k_text, self._k1)
        Ki = min(self.mfd_top_k_image, self._k2)
        Kp = self.mfd_pool_size
        inv_tau = 1.0 / self.mfd_tau
        w_t = 1.0 if self.mfd_score_text else 0.0
        w_i = 1.0 if self.mfd_score_image else 0.0
        pf = torch.zeros(B, nb, Kp, dtype=torch.long, device=dev)
        pt = torch.zeros_like(pf); pi = torch.zeros_like(pf)
        psc = torch.full((B, nb, Kp), float("-inf"), device=dev)
        plpf = torch.zeros(B, nb, Kp, device=dev)
        plpt = torch.zeros_like(plpf); plpi = torch.zeros_like(plpf)
        v_cnt = torch.zeros(B, nb, device=dev)   # catalog-valid compositions per behavior (budget accounting)
        n_comp = torch.zeros(B, nb, device=dev)  # candidates COMPOSED per behavior, before any catalog filter
        # GT's rank among ALL scored candidates of its own behavior (before the top-K_pool cut):
        #   -1 = GT never composed (its f not in the top-K_f roots, or -- dense path -- t/i pruned by K_t/K_i)
        # This is the number that says whether a missing GT was dropped by BRANCH PRUNING (-1) or by
        # the K_POOL cut (rank >= K_pool). It is dumped as a column so coverage@K for ANY K is offline.
        gt_comp_rank = torch.full((B,), -1, dtype=torch.long, device=dev)
        gt_comp_score = torch.full((B,), float("-inf"), device=dev)
        gt_in_pool = torch.zeros(B, dtype=torch.bool, device=dev)     # direct membership test in the emitted pool
        dbg = self._audit_dbg_left > 0
        if dbg:
            self._audit_dbg_left -= 1
            print("\n[compose-debug] ===== composition stage internals (read from the running code, not config) =====", flush=True)
            print(f"[compose-debug] path={'WIDE-CATALOG' if self.mfd_wide_catalog_pool else 'DENSE K_f x K_t x K_i grid'}"
                  f"  locals: Kf={Kf} Kt={Kt} Ki={Ki} Kp={Kp}  head widths: k0={self._k0} k1={self._k1} k2={self._k2}"
                  f"  catalog_constraint={self.mfd_catalog_constraint}", flush=True)
        for b in range(nb):
            if self.mfd_force_history:
                # history shared codes first (a +1e4 selection bonus), then the usual top by logP(f|b);
                # sc_f keeps the TRUE log-probs so scores are unchanged.
                _h = input_ids.long()[:, :(input_ids.shape[1] // 4) * 4].reshape(B, -1, 4)
                _hv = _h[:, :, 0] >= 0
                _bonus = torch.zeros(B, self._k0, device=dev)
                _bonus.scatter_(1, _h[:, :, 1].clamp(min=0, max=self._k0 - 1), _hv.float() * 1e4)
                f_idx = (f_logp[:, b, :] + _bonus).topk(Kf, dim=-1).indices
                sc_f = torch.gather(f_logp[:, b, :], 1, f_idx)
            else:
                sc_f, f_idx = f_logp[:, b, :].topk(Kf, dim=-1)        # (B, Kf)
            b_sel = torch.full((B, Kf), b, dtype=torch.long, device=dev)
            t_lp, i_lp = self._modality_heads(enc, emask, b_sel, f_idx)  # (B,Kf,k1),(B,Kf,k2)
            is_gt_b = gt_b == b
            # GT's root index among this behavior's top-Kf roots (-1 if pruned by K_f)
            gt_root = torch.full((B,), -1, dtype=torch.long, device=dev)
            hit_r = (f_idx == gt_f.view(B, 1))                     # (B, Kf)
            gt_root = torch.where(hit_r.any(1), hit_r.float().argmax(1), gt_root)

            # ---- G5 wide-catalog pool: score the unique catalog tuples directly (= K_t=K_i=full,
            # no dense grid). Per event, map each catalog tuple's shared code to its top-Kf root (or
            # drop it), gather the root's per-modality log-probs at that tuple's t/i, score, top-Kp.
            if self.mfd_wide_catalog_pool:
                CF, CT, CI = self._catalog_triples()              # (Mc,) each
                Mc = CF.numel()
                code2root = torch.full((B, self._k0), -1, dtype=torch.long, device=dev)
                code2root.scatter_(1, f_idx, torch.arange(Kf, device=dev).view(1, Kf).expand(B, Kf))
                root = code2root[:, CF]                            # (B, Mc) root index or -1
                valid = root >= 0
                rc = root.clamp(min=0)
                scf_i = torch.gather(sc_f, 1, rc)                 # (B, Mc)  logP(f|b)
                t_at = torch.gather(t_lp.reshape(B, Kf * self._k1), 1, rc * self._k1 + CT.view(1, Mc))
                i_at = torch.gather(i_lp.reshape(B, Kf * self._k2), 1, rc * self._k2 + CI.view(1, Mc))
                score = scf_i + inv_tau * (w_t * t_at + w_i * i_at)
                score = score.masked_fill(~valid, float("-inf"))
                kk = min(Kp, Mc)
                if self.mfd_force_history:
                    # history tuples (whose root is kept) enter the pool; stored scores stay the true ones
                    _keys = self._mfd_catalog_keys
                    _hk = (_h[:, :, 1].clamp(min=0) * self._k1 + _h[:, :, 2].clamp(min=0)) * self._k2 + _h[:, :, 3].clamp(min=0)
                    _pos = torch.searchsorted(_keys, _hk).clamp(max=Mc - 1)
                    _ok = (_keys[_pos] == _hk) & _hv
                    _sel = score.clone()
                    _bon = torch.zeros_like(score).scatter_(1, _pos, _ok.float() * 1e4)
                    _sel = torch.where(valid, _sel + _bon, _sel)
                    pidx = _sel.topk(kk, dim=-1).indices
                    sc_pool = torch.gather(score, 1, pidx)
                else:
                    sc_pool, pidx = score.topk(kk, dim=-1)            # (B, kk)   <-- the ONLY truncation on this path
                pf[:, b, :kk] = CF[pidx]; pt[:, b, :kk] = CT[pidx]; pi[:, b, :kk] = CI[pidx]
                psc[:, b, :kk] = sc_pool
                plpf[:, b, :kk] = torch.gather(scf_i, 1, pidx)
                plpt[:, b, :kk] = torch.gather(t_at, 1, pidx)
                plpi[:, b, :kk] = torch.gather(i_at, 1, pidx)
                n_comp[:, b] = valid.float().sum(1); v_cnt[:, b] = n_comp[:, b]   # composition == catalog here
                # GT bookkeeping (events whose GT behavior is b): catalog index of the GT tuple, its score,
                # its rank among all valid candidates, and direct membership in the emitted top-kk.
                keys = self._mfd_catalog_keys
                if keys is None:                                   # constraint off: rebuild the (sorted) keys
                    keys = (CF * self._k1 + CT) * self._k2 + CI
                gkey = (gt_f * self._k1 + gt_t) * self._k2 + gt_i
                j = torch.searchsorted(keys, gkey).clamp(max=Mc - 1)
                j_ok = (keys[j] == gkey) & valid[ar, j] & is_gt_b
                gsc = score[ar, j]
                grank = (score > gsc.view(B, 1)).sum(1)
                gt_comp_rank = torch.where(j_ok, grank, gt_comp_rank)
                gt_comp_score = torch.where(j_ok, gsc, gt_comp_score)
                gt_in_pool = gt_in_pool | (is_gt_b & (pidx == j.view(B, 1)).any(1))
                if dbg:
                    per_root = torch.zeros(B, Kf, device=dev).scatter_add_(1, rc, valid.float())   # tuples per f-branch
                    print(f"[compose-debug] b={b}: catalog tuples Mc={Mc:,}; t/i log-probs gathered from FULL heads "
                          f"t_lp{tuple(t_lp.shape)} i_lp{tuple(i_lp.shape)} -> K_t_eff={t_lp.shape[-1]} K_i_eff={i_lp.shape[-1]} "
                          f"(no t/i top-k on this path; K_t={Kt}/K_i={Ki} are NOT referenced)", flush=True)
                    print(f"[compose-debug]      candidates composed per event (tuples whose f-root is kept) = "
                          f"mean {n_comp[:, b].mean():,.0f} min {n_comp[:, b].min():,.0f} max {n_comp[:, b].max():,.0f}; "
                          f"per f-branch: mean {per_root.mean():.1f} max {per_root.max():.0f} (this is the catalog's tuples "
                          f"under that f, not K_t x K_i); catalog filter: none needed (composition = catalog); "
                          f"survivors after topk(min(Kp,Mc)) = {kk}", flush=True)
                continue

            tv, ti = t_lp.topk(Kt, dim=-1)                         # (B, Kf, Kt)
            iv, ii = i_lp.topk(Ki, dim=-1)                         # (B, Kf, Ki)
            lpf = sc_f.view(B, Kf, 1, 1).expand(B, Kf, Kt, Ki)
            comp = lpf + inv_tau * (w_t * tv.view(B, Kf, Kt, 1) + w_i * iv.view(B, Kf, 1, Ki))
            f_c = f_idx.view(B, Kf, 1, 1).expand(B, Kf, Kt, Ki)
            t_c = ti.view(B, Kf, Kt, 1).expand(B, Kf, Kt, Ki)
            i_c = ii.view(B, Kf, 1, Ki).expand(B, Kf, Kt, Ki)
            lpt_c = tv.view(B, Kf, Kt, 1).expand(B, Kf, Kt, Ki)
            lpi_c = iv.view(B, Kf, 1, Ki).expand(B, Kf, Kt, Ki)
            M = Kf * Kt * Ki
            comp = comp.reshape(B, M); f_c = f_c.reshape(B, M); t_c = t_c.reshape(B, M)
            i_c = i_c.reshape(B, M); lpf_c = lpf.reshape(B, M)
            lpt_c = lpt_c.reshape(B, M); lpi_c = lpi_c.reshape(B, M)
            n_comp[:, b] = float(M)
            if self.mfd_catalog_constraint:
                vmask = self._valid_tuple_mask(f_c, t_c, i_c)
                v_cnt[:, b] = vmask.reshape(B, -1).float().sum(1)
                comp = comp.masked_fill(~vmask, float("-inf"))
            else:
                v_cnt[:, b] = float(M)
            kk = min(Kp, M)
            sc_pool, pidx = comp.topk(kk, dim=-1)                  # <-- the ONLY truncation after the K_t/K_i top-k
            g = lambda x: torch.gather(x, 1, pidx)
            pf[:, b, :kk] = g(f_c); pt[:, b, :kk] = g(t_c); pi[:, b, :kk] = g(i_c)
            psc[:, b, :kk] = sc_pool
            plpf[:, b, :kk] = g(lpf_c); plpt[:, b, :kk] = g(lpt_c); plpi[:, b, :kk] = g(lpi_c)
            # GT bookkeeping: composed iff f kept AND t* within top-Kt of its root AND i* within top-Ki.
            rr = gt_root.clamp(min=0)
            t_row = t_lp[ar, rr]; i_row = i_lp[ar, rr]           # (B,k1),(B,k2) at the GT's root
            gt_tl = t_row[ar, gt_t]; gt_il = i_row[ar, gt_i]
            t_in = (t_row > gt_tl.view(B, 1)).sum(1) < Kt
            i_in = (i_row > gt_il.view(B, 1)).sum(1) < Ki
            gsc = sc_f[ar, rr] + inv_tau * (w_t * gt_tl + w_i * gt_il)
            composed = is_gt_b & (gt_root >= 0) & t_in & i_in
            grank = (comp > gsc.view(B, 1)).sum(1)
            gt_comp_rank = torch.where(composed, grank, gt_comp_rank)
            gt_comp_score = torch.where(composed, gsc, gt_comp_score)
            inpool_b = ((g(f_c) == gt_f.view(B, 1)) & (g(t_c) == gt_t.view(B, 1)) & (g(i_c) == gt_i.view(B, 1))
                        & (sc_pool > -1e30)).any(1)
            gt_in_pool = gt_in_pool | (is_gt_b & inpool_b)
            if dbg:
                print(f"[compose-debug] b={b}: t_lp{tuple(t_lp.shape)} i_lp{tuple(i_lp.shape)} -> topk(Kt={Kt}) topk(Ki={Ki}); "
                      f"pairs composed per f-branch = Kt*Ki = {Kt*Ki:,}; total per event M = Kf*Kt*Ki = {M:,}; "
                      f"catalog-valid survivors per event: mean {v_cnt[:, b].mean():,.0f} min {v_cnt[:, b].min():,.0f} "
                      f"max {v_cnt[:, b].max():,.0f}; survivors after topk(min(Kp,M)) = {kk}", flush=True)

        self._audit_eff = {
            "path": "wide_catalog" if self.mfd_wide_catalog_pool else "dense",
            "Kf_eff": Kf, "Kt_eff": (self._k1 if self.mfd_wide_catalog_pool else Kt),
            "Ki_eff": (self._k2 if self.mfd_wide_catalog_pool else Ki), "Kpool_eff": Kp,
            "Mc_catalog": int(self._catalog_triples()[0].numel()) if self.mfd_wide_catalog_pool else None,
            "truncations": (["score.topk(min(Kp,Mc)) per behavior"] if self.mfd_wide_catalog_pool
                            else ["t_lp.topk(Kt)", "i_lp.topk(Ki)", "catalog mask", "comp.topk(min(Kp,M)) per behavior"]),
        }
        if dbg:
            # per-event verdict for the GTs of this batch: where exactly did each missing GT go?
            print(f"[compose-debug] effective: {self._audit_eff}", flush=True)
            print(f"[compose-debug] {'ev':>3} {'b':>2} {'f*rk':>6} {'t*rk':>6} {'i*rk':>6} {'inpool':>6} {'comprk':>8} "
                  f"{'gt_score':>9} {'cutoff':>9}  verdict", flush=True)
            cut = psc[ar, gt_b].masked_fill(psc[ar, gt_b] <= -1e30, float("inf")).min(1).values   # kk-th score in GT's behavior
            for e in range(B):
                rk = int(gt_comp_rank[e]); inp = bool(gt_in_pool[e])
                if inp:
                    verdict = "in pool"
                elif rk < 0:
                    verdict = ("NOT COMPOSED: f* pruned by K_f" if int(f_star_rank[e]) >= Kf
                               else "NOT COMPOSED: t*/i* pruned by K_t/K_i (dense path)")
                elif float(gt_comp_score[e]) < float(cut[e]):
                    verdict = f"TRUNCATED BY K_pool (rank {rk} >= {Kp})"
                else:
                    verdict = "** BUG: score >= cutoff yet absent **"
                print(f"[compose-debug] {e:>3} {int(gt_b[e]):>2} {int(f_star_rank[e]):>6} {int(tf_t_rank[e]):>6} "
                      f"{int(tf_i_rank[e]):>6} {str(inp):>6} {rk:>8} {float(gt_comp_score[e]):>9.2f} {float(cut[e]):>9.2f}  {verdict}",
                      flush=True)
        i16 = lambda x: x.to(torch.int16).cpu()
        i32 = lambda x: x.to(torch.int32).cpu()
        f16 = lambda x: x.to(torch.float16).cpu()
        if user_id is None:
            uid = torch.zeros(B, dtype=torch.int32)
        else:
            uid = i32(user_id[:, 0] if user_id.dim() > 1 else user_id)
        rec = {
            "user_id": uid,
            "gt_b": i16(gt_b), "gt_f": i16(gt_f), "gt_t": i16(gt_t), "gt_i": i16(gt_i),
            "beh_logp": f16(beh_logp),
            "gt_in_catalog": gt_in_catalog.to(torch.int8).cpu(),
            "f_star_rank": i32(f_star_rank), "f_star_logp": f16(f_star_logp),
            "tf_t_rank": i32(tf_t_rank), "tf_i_rank": i32(tf_i_rank),
            "tf_t_logp": f16(tf_t_logp), "tf_i_logp": f16(tf_i_logp),
            "pool_f": i16(pf), "pool_t": i16(pt), "pool_i": i16(pi),
            "pool_sidscore": f16(psc), "pool_lpf": f16(plpf),
            "pool_lpt": f16(plpt), "pool_lpi": f16(plpi),
            "v_cnt": v_cnt.to(torch.float32).cpu(), "n_comp": n_comp.to(torch.float32).cpu(),   # ~6e5 overflows fp16 (printed inf)
            "gt_comp_rank": i32(gt_comp_rank), "gt_comp_score": f16(gt_comp_score),
            "gt_in_pool": gt_in_pool.to(torch.int8).cpu(),
        }
        # Stage-1 reranker dump: attach the frozen encoder memory (B,S,d) + mask (B,S). fp16
        # keeps it compact (~40KB/event at S=160,d=128); the reranker cross-attends into it.
        # Also store the raw history token sequence (input_ids: interleaved [b,f,t,i,...], pad=-1)
        # so the reranker can build per-candidate ENGAGEMENT features (hard tuple-match + soft
        # cosine-match against the user's history, per behavior). ~S int16 = negligible size.
        if self.mfd_dump_memory:
            rec["enc_mem"] = f16(enc)
            rec["enc_mask"] = emask.to(torch.int8).cpu()
            rec["hist_ids"] = input_ids.to(torch.int16).cpu()
        return rec

    def eval_step(self, batch, loss_to_aggregate):
        if not self.mfd_audit_dump:
            return super().eval_step(batch, loss_to_aggregate)
        if self._audit_n >= self.mfd_audit_max_events:
            return  # dump full — skip the (heavy) generation entirely
        model_input, label_data = batch
        inputs = {
            self.feature_to_model_input_map.get(k, k): v
            for k, v in model_input.transformed_sequences.items()
        }
        labels = list(label_data.labels.values())[0].reshape(model_input.mask.size(0), -1)
        _dev = model_input.mask.device
        if _dev.type == "cuda":
            torch.cuda.synchronize(_dev)
        _t0 = time.perf_counter()
        rec = self._audit_forward(
            mask=model_input.mask, input_ids=inputs["input_ids"],
            user_id=inputs.get("user_id"), labels=labels.to(_dev),
        )
        if _dev.type == "cuda":
            torch.cuda.synchronize(_dev)
        self._audit_gen_time += time.perf_counter() - _t0
        self._audit_gen_events += int(model_input.mask.size(0))
        # stratified keep: every non-pv event; pv only up to pv_cap; overall cap max_events.
        gt_b = rec["gt_b"].long()
        keep = torch.zeros(gt_b.shape[0], dtype=torch.bool)
        for idx in range(gt_b.shape[0]):
            if self._audit_n >= self.mfd_audit_max_events:
                break
            if gt_b[idx].item() == self.mfd_audit_pv_id:
                if self._audit_pv >= self.mfd_audit_pv_cap:
                    continue
                self._audit_pv += 1
            keep[idx] = True
            self._audit_n += 1
        if keep.any():
            for k, v in rec.items():
                self._audit_cols[k].append(v[keep])

    def on_test_epoch_end(self):
        # Non-audit runs MUST fall through to the base hook that logs test/loss + metrics.
        if not self.mfd_audit_dump:
            return super().on_test_epoch_end()
        if not self._audit_cols:
            return
        cols = {k: torch.cat(v, dim=0) for k, v in self._audit_cols.items()}
        meta = {
            "nb": self.num_behaviors, "pv_id": self.mfd_audit_pv_id,
            "k0": self._k0, "k1": self._k1, "k2": self._k2,
            "Kf": min(self.mfd_top_k_shared, self._k0),
            "Kt": min(self.mfd_top_k_text, self._k1),
            "Ki": min(self.mfd_top_k_image, self._k2),
            "Kpool": self.mfd_pool_size, "tau": self.mfd_tau,
            "w_b_run": self.mfd_behavior_weight, "buy_id": self.buy_behavior_id,
            "d_model": self.embedding_dim, "has_memory": bool(self.mfd_dump_memory),
            "wide_catalog": bool(self.mfd_wide_catalog_pool), "force_history": bool(self.mfd_force_history),
            # what the composition stage ACTUALLY used (recorded inside _audit_forward, not from config)
            "eff": dict(self._audit_eff),
            # decoder-state evaluations per item for factorized decoding:
            #   Stage A  : nb * (behavior slot + f slot) = nb * 2
            #   modality : nb * Kf forwards at the kept (b,f) roots (t/i are free head-reads;
            #              K_t / K_i do NOT add decoder states -- that is the factorization).
            "fwd_states_per_item": self.num_behaviors * (2 + min(self.mfd_top_k_shared, self._k0)),
            # peak GPU memory during generation -- reported so the wide-catalog (P3) pool, which
            # scores catalog tuples branch-by-branch instead of materialising K_f*K_t*K_i, is
            # reproducible.
            "peak_mem_gb": (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0,
            "gen_seconds": float(self._audit_gen_time),
            "gen_events": int(self._audit_gen_events),
        }
        out = {"cols": cols, "meta": meta}
        torch.save(out, self.mfd_audit_path)
        n = cols["gt_b"].shape[0]
        n_invalid = int((cols["gt_in_catalog"] == 0).sum())
        print(f"[MFD audit] wrote {n:,} events -> {self.mfd_audit_path}", flush=True)
        print(f"[MFD audit] CATALOG UNIT TEST: GT tuples NOT in catalog = {n_invalid} / {n} "
              f"({100.0 * n_invalid / max(n, 1):.4f}%) -- expected 0 if the constraint hash is correct.",
              flush=True)
        if n_invalid > 0:
            print("[MFD audit] ** catalog hash is WRONG (a real GT tuple is missing). **", flush=True)
        else:
            print("[MFD audit] catalog OK -> any 'GT not in pool' is POOL TRUNCATION (rank>Kpool), "
                  "not a catalog bug.", flush=True)