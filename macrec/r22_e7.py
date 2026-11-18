"""mmgen addition (docs/MFD_REPORT.md §22.7, E7): structured item vector that keeps f separable, on A01.

A01's item vector (enc_struct E2, one encoder position per history item): s_j = sum of the code embeddings present in the
item's view (5 in the full view, f + 2 text or f + 2 image codes in the text / image views) + recency r_j. E7 adds a linear map
of the PARTS of that sum, zero-initialised, so at step 0 the model is exactly A01:
  E7-a  e_j = s_j + W [E(f) ; E(t1) + E(t2) ; E(i1) + E(i2)] + r_j      W = Linear(3d -> d, no bias), init 0
  E7-b  e_j = s_j + W_f E(f) + r_j                                      W_f = Linear(d -> d, no bias), init 0
A part absent from a view (the image codes in the text view, ...) is a zero vector; there is no view marker (as in A01).
1+2+2 SIDs only (slots s0 t0 t1 v0 v1 of enc_struct; t0 / t1 = the item's first / second text code). The class is
GroupedItemMFDT5 (A01's training loss) + this encoder; config.r22_e7 = "a" | "b" rebuilds it at load (mfd_eval.load_mfd_model).
"""
import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput

from enc_struct import MAX_ITEMS, OTHER
from r21_views import GroupedItemMFDT5

PARTS = {"f": (0,), "t": (1, 2), "i": (3, 4)}        # enc_struct.SLOTS = (s0, t0, t1, v0, v1)


class E7Mixin:
    def __init__(self, config):
        super().__init__(config)
        self.r22_e7 = getattr(config, "r22_e7", None)
        if self.r22_e7:
            self._build_e7()

    def _build_e7(self):
        d = self.config.d_model
        self.e7 = nn.Linear(3 * d if self.r22_e7 == "a" else d, d, bias=False)
        nn.init.zeros_(self.e7.weight)

    def install_e7(self, mode):
        assert mode in ("a", "b"), mode
        assert getattr(self, "struct_mode", None) == "E2", "E7 needs the item-level encoder (A01 / P1a)"
        dev = next(self.parameters()).device
        self.r22_e7 = mode
        self.config.r22_e7 = mode
        self._build_e7()
        self.e7.to(dev)
        return {"r22_e7": mode, "e7_in": self.e7.in_features, "e7_params": self.e7.weight.numel(), "e7_zero_init": True}

    def item_parts(self, input_ids):
        """(B, n_items, d) sums of the f / text / image code embeddings of every history item, + mask, n."""
        emb = self.shared(input_ids)
        s, code, item, rec, n = self.struct.structure(input_ids)
        assert not bool((s > OTHER).any()), "E7 is defined for 1+2+2 SIDs"
        B, _, d = emb.shape
        nmax = max(int(n.max()), 1)
        idx = item.clamp(min=0)[..., None].expand(-1, -1, d)
        out = {}
        for k, slots in PARTS.items():
            m = code & torch.isin(s, torch.tensor(slots, device=s.device))
            out[k] = torch.zeros(B, nmax, d, device=emb.device, dtype=emb.dtype).scatter_add(1, idx, emb * m[..., None].to(emb.dtype))
        j = torch.arange(nmax, device=emb.device)[None]
        mask = (j < n[:, None]).long()
        return out, mask, n, j

    def struct_encode(self, input_ids, attention_mask):
        if not self.r22_e7:
            return super().struct_encode(input_ids, attention_mask)
        p, mask, n, j = self.item_parts(input_ids)
        x = p["f"] + p["t"] + p["i"]                                    # = A01's sum s_j
        x = x + self.e7(torch.cat([p["f"], p["t"], p["i"]], -1) if self.r22_e7 == "a" else p["f"])
        x = x + self.struct.rec((n[:, None] - 1 - j).clamp(0, MAX_ITEMS - 1)) * mask[..., None].to(x.dtype)
        out = self.encoder(inputs_embeds=x, attention_mask=mask, return_dict=True)
        return BaseModelOutput(last_hidden_state=out.last_hidden_state), mask


class E7GroupedItemMFDT5(E7Mixin, GroupedItemMFDT5):
    pass
