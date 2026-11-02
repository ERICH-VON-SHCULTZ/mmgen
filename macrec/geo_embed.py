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
   T = median over f codes of the squared distance to their 10th-nearest f code. Applied to every label that is an
   f token (chain: position 0; MFD: text row position 0); all other labels stay one-hot. eps = 0: HF's own loss.
After training, bake() turns the model back into a plain T5 whose embedding table is the learned P c + delta, so all
evaluation code loads it unchanged.
"""
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

CODE_TOKEN = re.compile(r"^<([stv])(\d+)_(\d+)>$")
RQ = {"s": "shared_rq", "t": "text_rq", "v": "image_rq"}


def code_tokens(tokenizer):
    """[(token id, level key e.g. 't0', code)] for every SID code token in the tokenizer."""
    out = []
    for tok, i in tokenizer.get_vocab().items():
        m = CODE_TOKEN.match(tok)
        if m:
            out.append((i, m.group(1) + m.group(2), int(m.group(3))))
    return sorted(out)


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


def soft_f_targets(tokenizer, books, eps):
    """(f token ids (K,), Q (K, K) target rows, T, info) for the f level 's0'."""
    toks = [(i, c) for i, lv, c in code_tokens(tokenizer) if lv == "s0"]
    ids = torch.tensor([i for i, _ in toks])
    c = books["s0"][torch.tensor([k for _, k in toks])]
    d2 = torch.cdist(c, c).pow(2)
    T = d2.sort(dim=1).values[:, 10].median().item()             # column 0 is the code itself
    logits = -d2 / T
    logits.fill_diagonal_(float("-inf"))
    Q = eps * logits.softmax(dim=1)
    Q += (1 - eps) * torch.eye(len(toks))
    return ids, Q, T, {"T": T, "eps": eps, "f_tokens": len(toks)}


class GeoMixin:
    """install() / soft-label loss / bake() for PlainT5 and MFDT5 subclasses."""
    _tied_weights_keys = ["encoder.embed_tokens", "decoder.embed_tokens", "lm_head"]
    geo_installed = False

    def install_geometry(self, tokenizer, ckpt_path, geo, eps=0.0, perm_seed=0):
        assert self.config.tie_word_embeddings and self.lm_head.weight is self.shared.weight, "expected a tied T5"
        books = load_codebooks(ckpt_path)
        toks = code_tokens(tokenizer)
        info = {"geo": geo, "tied_head": True, "codebook_tensor": "<chain>_rq.vq_layers.<l>.embedding.weight",
                "ckpt": ckpt_path, "code_tokens": len(toks)}
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
            ids, Q, T, sinfo = soft_f_targets(tokenizer, books, eps)
            fmap = torch.full((len(tokenizer) + 1,), -1, dtype=torch.long)
            fmap[ids] = torch.arange(len(ids))
            self.register_buffer("soft_ids", ids, persistent=False)
            self.register_buffer("soft_Q", Q, persistent=False)
            self.register_buffer("soft_map", fmap, persistent=False)
            info["soft_f"] = sinfo
        self.geo_info = info
        return info

    def tie_weights(self):
        if not self.geo_installed:
            super().tie_weights()

    def soft_loss(self, logits, labels):
        """Mean token cross-entropy over labels != -100; labels that are f tokens use the soft target."""
        lp = logits.float().log_softmax(-1).reshape(-1, logits.size(-1))
        y = labels.reshape(-1)
        valid = y != -100
        yc = y.clamp(min=0)
        nll = -lp.gather(1, yc[:, None]).squeeze(1)
        r = self.soft_map[yc.clamp(max=len(self.soft_map) - 1)]
        m = valid & (r >= 0)
        if m.any():
            nll = nll.clone()
            nll[m] = -(lp[m][:, self.soft_ids] * self.soft_Q[r[m]]).sum(1)
        return nll[valid].mean()

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
            if getattr(self, "soft_Q", None) is None or labels is None:
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
