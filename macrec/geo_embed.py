"""mmgen addition: codebook-geometry-aware T5 for MD SIDs (docs/MFD_REPORT.md §14).

A. Codebook-initialised code embeddings. Every SID code token <s0_k> / <t0_k> / <t1_k> / <v0_k> / <v1_k> (level
   key = token prefix) gets
       emb(code) = P_level c(code) + delta(code)
   c(code) = row k of that level's codebook in the MD-RQ-VAE tokenizer checkpoint: <prefix>_rq.vq_layers.<l>.
   embedding.weight (MACRec's VectorQuantizer: a plain nn.Embedding trained by the VQ loss after k-means init; there
   is no EMA / centroid buffer; the SIDs are argmins over exactly these rows, cross_index/reassign_md.py).
   P_level (d_model x d_code) = alpha_level * Q_level, Q_level with orthonormal columns (an isometry R^32 -> R^128,
   so cosine similarities are preserved exactly at init), alpha_level = mean norm of T5's default token embeddings
   (original vocabulary rows at init) / mean norm of that level's c(code). delta = 0 at init. Both learnable.
   Every other token (prompts, <T>, <I>, specials, original vocabulary) keeps its normal embedding row.
   This T5 ties input embeddings and output head (config has no tie_word_embeddings -> True; checked at install),
   so the same table also gives the output logits: geometry enters both.
   geo="shuffled": the control; codebook vectors are permuted among the code tokens of each level (fixed seed).
B. Soft labels for f: target q(f') = (1 - eps) 1[f' = f*] + eps softmax_{f' != f*}(-||c(f') - c(f*)||^2 / T),
   T = median over f codes of the squared distance to their 10th-nearest f code, or the T that puts a given share of
   the eps mass on the 10 nearest f codes (§15 V4). soft_mode="uniform": plain label smoothing on f (§15 V3). Applied to every label that is an
   f token (chain: position 0; MFD: text row position 0); all other labels stay one-hot. eps = 0: HF's own loss.
C. Further training terms on top of A (§17): f_weight multiplies the cross-entropy of every f label (training only; the
   denominator stays the number of targets, so all other terms are unchanged); offset_lambda adds lambda * mean over
   code tokens of ||delta||^2 (training only). Validation loss (early stopping) stays the plain token cross-entropy.
sid="c" (§17 Part 3): the same construction for baseline C's tokens <a_k>..<d_k> (text) and <A_k>..<D_k> (image), from
   rq.vq_layers.<l>.embedding.weight of C's two tokenizers; the level key is the token's letter.
After training, bake() turns the model back into a plain T5 whose embedding table is the learned P c + delta, so all
evaluation code loads it unchanged.
"""
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

CODE_TOKEN = re.compile(r"^<([stv])(\d+)_(\d+)>$")
C_TOKEN = re.compile(r"^<([a-dA-D])_(\d+)>$")
RQ = {"s": "shared_rq", "t": "text_rq", "v": "image_rq"}


def code_tokens(tokenizer, sid="md"):
    """[(token id, level key e.g. 't0' (sid c: 'a'..'D'), code)] for every SID code token in the tokenizer."""
    out = []
    for tok, i in tokenizer.get_vocab().items():
        m = (C_TOKEN if sid == "c" else CODE_TOKEN).match(tok)
        if m:
            out.append((i, m.group(1), int(m.group(2))) if sid == "c" else (i, m.group(1) + m.group(2), int(m.group(3))))
    return sorted(out)


def load_c_codebooks(text_ckpt, image_ckpt):
    """C's two single-modality tokenizers: level key 'a'..'d' (text), 'A'..'D' (image) -> (K, d_code) codebook tensor."""
    out = {}
    for path, letters in ((text_ckpt, "abcd"), (image_ckpt, "ABCD")):
        sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        for j, c in enumerate(letters):
            out[c] = sd[f"rq.vq_layers.{j}.embedding.weight"].float()
    return out


def load_codebooks(ckpt_path):
    """level key -> (K, d_code) codebook tensor."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    out = {}
    for k, v in sd.items():
        m = re.match(r"^(shared_rq|text_rq|image_rq)\.vq_layers\.(\d+)\.embedding\.weight$", k)
        if m:
            out[{b: a for a, b in RQ.items()}[m.group(1)] + m.group(2)] = v.float()
    return out


class GeoEmbedding(nn.Module):
    """Embedding table whose SID code rows are P_level c + delta (shared by encoder, decoder and the tied head)."""

    def __init__(self, base, toks, books, perm_seed=None):
        super().__init__()
        self.base = base                                         # the model's nn.Embedding (all other rows)
        self.levels = sorted({lv for _, lv, _ in toks})
        d_model = base.weight.shape[1]
        target = base.weight[:32100].norm(dim=1).mean().item()  # T5's default token embeddings (original vocabulary)
        g = torch.Generator().manual_seed(perm_seed) if perm_seed is not None else None
        self.P = nn.ParameterDict()
        self.info = {"target_mean_norm": target, "levels": {}}
        for lv in self.levels:
            sel = [(i, c) for i, l, c in toks if l == lv]
            ids = torch.tensor([i for i, _ in sel])
            cv = books[lv][torch.tensor([c for _, c in sel])]
            perm = torch.randperm(len(sel), generator=g) if g is not None else torch.arange(len(sel))
            cv = cv[perm]
            q = torch.empty(d_model, cv.shape[1])
            nn.init.orthogonal_(q)                               # orthonormal columns: q^T q = I
            alpha = target / cv.norm(dim=1).mean().item()
            self.P[lv] = nn.Parameter(alpha * q)
            self.register_buffer(f"ids_{lv}", ids)
            self.register_buffer(f"cv_{lv}", cv)
            self.info["levels"][lv] = {"tokens": len(sel), "alpha": alpha, "code_mean_norm": cv.norm(dim=1).mean().item(),
                                       "permuted": g is not None}
        n = sum(len(getattr(self, f"ids_{lv}")) for lv in self.levels)
        self.delta = nn.Parameter(torch.zeros(n, d_model))

    def code_rows(self):
        rows = torch.cat([getattr(self, f"cv_{lv}") @ self.P[lv].t() for lv in self.levels])
        return torch.cat([getattr(self, f"ids_{lv}") for lv in self.levels]), rows + self.delta

    def full_weight(self):
        ids, rows = self.code_rows()
        return self.base.weight.index_copy(0, ids, rows.to(self.base.weight.dtype))

    def forward(self, input_ids):
        return F.embedding(input_ids, self.full_weight())


class GeoHead(nn.Module):
    """Tied output head reading the GeoEmbedding table (not registered as a submodule: no duplicate parameters)."""

    def __init__(self, geo):
        super().__init__()
        self.__dict__["geo"] = geo

    def forward(self, x):
        return F.linear(x, self.geo.full_weight())


def nn_mass(d2, T, k=10):
    """Mean (over true f) share of the eps mass on the k nearest other f codes, for temperature T."""
    lg = -d2 / T
    lg.fill_diagonal_(float("-inf"))
    q = lg.softmax(dim=1)
    off = d2.clone()
    off.fill_diagonal_(float("inf"))
    order = off.argsort(dim=1)
    return q.gather(1, order[:, :k]).sum(1).mean().item()


def soft_f_targets(tokenizer, books, eps, mode="geometric", nn_target=None):
    """(f token ids (K,), Q (K, K) target rows, T, info) for the f level 's0'.
    mode geometric: q(f') = (1 - eps) 1[f' = f*] + eps softmax_{f' != f*}(-||c(f') - c(f*)||^2 / T), T = median squared
      distance to the 10th-nearest f code, or (nn_target) the T whose eps mass on the 10 nearest codes is nn_target.
    mode uniform: plain label smoothing on f: eps / (K - 1) on every other f code."""
    toks = [(i, c) for i, lv, c in code_tokens(tokenizer) if lv == "s0"]
    ids = torch.tensor([i for i, _ in toks])
    K = len(toks)
    info = {"eps": eps, "f_tokens": K, "mode": mode}
    if mode == "uniform":
        Q = torch.full((K, K), eps / (K - 1))
        Q.fill_diagonal_(1 - eps)
        return ids, Q, None, info | {"T": None}
    c = books["s0"][torch.tensor([k for _, k in toks])]
    d2 = torch.cdist(c, c).pow(2)
    T = d2.sort(dim=1).values[:, 10].median().item()             # column 0 is the code itself
    if nn_target is not None:                                     # bisection on log T (mass decreases with T)
        lo, hi = 1e-3, T
        for _ in range(100):
            mid = (lo * hi) ** 0.5
            lo, hi = (mid, hi) if nn_mass(d2, mid) > nn_target else (lo, mid)
        T = (lo * hi) ** 0.5
    logits = -d2 / T
    logits.fill_diagonal_(float("-inf"))
    Q = eps * logits.softmax(dim=1)
    Q += (1 - eps) * torch.eye(K)
    return ids, Q, T, info | {"T": T, "nn_target": nn_target, "eps_mass_5nn": nn_mass(d2, T, 5), "eps_mass_10nn": nn_mass(d2, T, 10)}


class GeoMixin:
    """install() / soft-label loss / bake() for PlainT5 and MFDT5 subclasses."""
    _tied_weights_keys = ["encoder.embed_tokens", "decoder.embed_tokens", "lm_head"]
    geo_installed = False
    f_weight, offset_lambda = 1.0, 0.0

    def install_geometry(self, tokenizer, ckpt_path, geo, eps=0.0, perm_seed=0, soft_mode="geometric", nn_target=None,
                         f_weight=1.0, offset_lambda=0.0, sid="md", image_ckpt=None):
        assert self.config.tie_word_embeddings and self.lm_head.weight is self.shared.weight, "expected a tied T5"
        books = load_c_codebooks(ckpt_path, image_ckpt) if sid == "c" else load_codebooks(ckpt_path)
        toks = code_tokens(tokenizer, sid)
        info = {"geo": geo, "tied_head": True, "codebook_tensor": "<chain>_rq.vq_layers.<l>.embedding.weight",
                "ckpt": ckpt_path, "code_tokens": len(toks), "sid": sid, "image_ckpt": image_ckpt,
                "f_weight": f_weight, "offset_lambda": offset_lambda}
        self.f_weight, self.offset_lambda = f_weight, offset_lambda
        if geo in ("codebook", "shuffled"):
            g = GeoEmbedding(self.shared, toks, books, perm_seed if geo == "shuffled" else None)
            self.shared = g
            self.encoder.embed_tokens = g
            self.decoder.embed_tokens = g
            self.lm_head = GeoHead(g)
            self.geo_installed = True
            info.update(g.info)
            info["perm_seed"] = perm_seed if geo == "shuffled" else None
        if eps > 0:
            ids, Q, T, sinfo = soft_f_targets(tokenizer, books, eps, soft_mode, nn_target)
            fmap = torch.full((len(tokenizer) + 1,), -1, dtype=torch.long)
            fmap[ids] = torch.arange(len(ids))
            self.register_buffer("soft_ids", ids, persistent=False)
            self.register_buffer("soft_Q", Q, persistent=False)
            self.register_buffer("soft_map", fmap, persistent=False)
            info["soft_f"] = sinfo
        elif f_weight != 1.0:
            ids = torch.tensor([i for i, lv, _ in toks if lv == "s0"])
            fmap = torch.full((len(tokenizer) + 1,), -1, dtype=torch.long)
            fmap[ids] = torch.arange(len(ids))
            self.register_buffer("soft_map", fmap, persistent=False)
        self.geo_info = info
        return info

    def tie_weights(self):
        if not self.geo_installed:
            super().tie_weights()

    def custom_active(self):
        return getattr(self, "soft_Q", None) is not None or (self.training and (self.f_weight != 1.0 or self.offset_lambda > 0))

    def offset_penalty(self):
        return self.shared.delta.pow(2).sum(1).mean()

    def soft_loss(self, logits, labels):
        """Token cross-entropy summed over labels != -100 / their number; f labels use the soft target if there is one
        and (training) weight f_weight; + offset penalty (training)."""
        lp = logits.float().log_softmax(-1).reshape(-1, logits.size(-1))
        y = labels.reshape(-1)
        valid = y != -100
        yc = y.clamp(min=0)
        nll = -lp.gather(1, yc[:, None]).squeeze(1)
        Q = getattr(self, "soft_Q", None)
        if Q is not None or self.f_weight != 1.0:
            r = self.soft_map[yc.clamp(max=len(self.soft_map) - 1)]
            m = valid & (r >= 0)
            if Q is not None and m.any():
                nll = nll.clone()
                nll[m] = -(lp[m][:, self.soft_ids] * Q[r[m]]).sum(1)
            if self.training and self.f_weight != 1.0:
                nll = nll * torch.where(m, torch.full_like(nll, self.f_weight), torch.ones_like(nll))
        loss = nll[valid].sum() / valid.sum()
        if self.training and self.offset_lambda > 0:
            loss = loss + self.offset_lambda * self.offset_penalty()
        return loss

    @torch.no_grad()
    def bake(self):
        """Replace the geometry modules by a plain tied nn.Embedding holding the learned table."""
        if not self.geo_installed:
            return
        W = self.shared.full_weight().detach().clone()
        emb = nn.Embedding(W.shape[0], W.shape[1]).to(W.device)
        emb.weight.data.copy_(W)
        self.shared = emb
        self.encoder.embed_tokens = emb
        self.decoder.embed_tokens = emb
        self.lm_head = nn.Linear(W.shape[1], W.shape[0], bias=False).to(W.device)
        self.geo_installed = False
        super().tie_weights()


def make_geo_class(base_cls):
    """GeoMixin on top of finetune_ext.PlainT5 or mfd_model.MFDT5; the soft-label loss replaces the model's own."""

    class Geo(GeoMixin, base_cls):

        # Explicit argument names (HF Trainer keeps only batch keys in forward()'s signature), as PlainT5 / MFDT5.
        def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, labels=None,
                    decoder_attention_mask=None, encoder_outputs=None, task_flag=None, **kwargs):
            if decoder_attention_mask is not None:
                kwargs["decoder_attention_mask"] = decoder_attention_mask
            if encoder_outputs is not None:
                kwargs["encoder_outputs"] = encoder_outputs
            if labels is None or not self.custom_active():
                return base_cls.forward(self, input_ids=input_ids, attention_mask=attention_mask,
                                        decoder_input_ids=decoder_input_ids, labels=labels, **kwargs)
            if decoder_input_ids is None:                       # chain: HF builds them from the labels
                decoder_input_ids = self._shift_right(labels)
            out = base_cls.forward(self, input_ids=input_ids, attention_mask=attention_mask,
                                   decoder_input_ids=decoder_input_ids, labels=None, **kwargs)
            # a new output object: ModelOutput only exposes keys set at construction (out.loss = ... is not seen by the
            # Trainer, which reads outputs["loss"])
            return type(out)(loss=self.soft_loss(out.logits, labels), **{k: v for k, v in out.items() if k != "loss"})

    Geo.__name__ = "Geo" + base_cls.__name__
    return Geo
