"""mmgen addition (docs/MFD_REPORT.md §22.8, E8): f embedding composed from its children in the MD SID tree, on A01.

For every shared code f (an <s0_*> token):
  c(f)  = mean over the catalog items under f of the sum of their child-code embeddings
          E8-a: E(t1) + E(t2) + E(i1) + E(i2)          E8-b: E(t1) + E(i1)
  E'(f) = E(f) + W c(f),   W = Linear(d -> d, no bias), initialised to ZERO (step 0 = A01 exactly)
The SID tree comes from the locked v1 1+2+2 index of all catalog items (no interaction data). c(f) is recomputed from the
CURRENT embedding table on every forward pass (gradients flow into the child embeddings as well as W).
E'(f) replaces E(f) in the whole tied table: the item-vector sum of the encoder input (enc_struct E2), the tied output layer
(f-step logits; f rows of every softmax) and the decoder's input f token (the spec names the first two; the decoder input is
the remaining use of E(f) and is switched too so that one table is used everywhere).
Implementation: E8Embedding wraps the model's embedding module (the GeoEmbedding during training) and exposes full_weight(),
exactly like geo_embed.GeoEmbedding, so finetune_geo.bake_and_save folds E'(f) (with the final W and c) into the plain table
that is saved. The saved checkpoint is therefore a plain A01-layout model (no W; config.r22_e8 records the variant only) and
every evaluation tool loads it unchanged. f codes with no catalog item (if any) get c(f) = 0.
"""
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from geo_embed import GeoHead
from r21_views import GroupedItemMFDT5

CHILD_POS = {"a": (1, 2, 3, 4), "b": (1, 3)}          # 1+2+2 SID = [s0, t0, t1, v0, v1]: t1 t2 i1 i2 / t1 i1


def table_of(emb):
    return emb.full_weight() if hasattr(emb, "full_weight") else emb.weight


class E8Embedding(nn.Module):
    def __init__(self, inner, f_ids, child_ids, M):
        super().__init__()
        self.inner = inner
        d = table_of(inner).shape[1]
        self.W = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.W.weight)
        self.register_buffer("f_ids", f_ids)
        self.register_buffer("child_ids", child_ids)
        self.register_buffer("M", M)                      # (n_f, n_child): count / items under f

    def children_mean(self, T=None):
        T = table_of(self.inner) if T is None else T
        return self.M @ T[self.child_ids]

    def full_weight(self):
        T = table_of(self.inner)
        rows = T[self.f_ids] + self.W(self.children_mean(T))
        return T.index_copy(0, self.f_ids, rows.to(T.dtype))

    def forward(self, input_ids):
        return F.embedding(input_ids, self.full_weight())


def tree(tokenizer, index_file, mode):
    """f token ids, child token ids, M (n_f x n_child) averaging matrix, bucket-size info."""
    vocab = tokenizer.get_vocab()
    f_ids = sorted(i for t, i in vocab.items() if t.startswith("<s0_"))
    frow = {i: r for r, i in enumerate(f_ids)}
    idx = json.load(open(index_file))
    pos = CHILD_POS[mode]
    child = sorted({vocab[toks[p]] for toks in idx.values() for p in pos})
    ccol = {i: c for c, i in enumerate(child)}
    M = torch.zeros(len(f_ids), len(child))
    n = torch.zeros(len(f_ids))
    for toks in idx.values():
        assert len(toks) == 5 and toks[0].startswith("<s0_"), "E8 is defined for the 1+2+2 SIDs"
        r = frow[vocab[toks[0]]]
        n[r] += 1
        for p in pos:
            M[r, ccol[vocab[toks[p]]]] += 1
    M = M / n.clamp(min=1)[:, None]
    nz = n[n > 0]
    info = {"f_tokens": len(f_ids), "f_with_items": int((n > 0).sum()), "f_without_items": int((n == 0).sum()),
            "items": len(idx), "items_per_f": {"min": int(nz.min()), "median": float(nz.median()), "mean": float(nz.mean()),
                                               "max": int(nz.max())},
            "child_tokens": len(child), "children_per_item": len(pos)}
    return torch.tensor(f_ids), torch.tensor(child), M, info


class E8Mixin:
    r22_e8 = None

    def install_e8(self, tokenizer, index_file, mode):
        assert mode in CHILD_POS, mode
        assert getattr(self, "struct_mode", None) == "E2", "E8 is defined on A01 (item-level encoder)"
        dev = next(self.parameters()).device
        f_ids, child, M, info = tree(tokenizer, index_file, mode)
        e = E8Embedding(self.shared, f_ids, child, M).to(dev)
        self.shared = e
        self.encoder.embed_tokens = e
        self.decoder.embed_tokens = e
        self.lm_head = GeoHead(e)
        self.r22_e8 = mode
        self.config.r22_e8 = mode
        with torch.no_grad():
            c = e.children_mean()
        info.update({"r22_e8": mode, "e8_params": e.W.weight.numel(), "e8_zero_init": True,
                     "c_finite": bool(torch.isfinite(c).all()), "c_norm_mean": float(c.norm(dim=1).mean())})
        return info

    def e8_w_norm(self):
        W = getattr(self.shared, "W", None)
        return None if W is None else float(W.weight.norm())


class E8GroupedItemMFDT5(E8Mixin, GroupedItemMFDT5):
    pass
