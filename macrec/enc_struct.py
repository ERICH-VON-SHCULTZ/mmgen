"""mmgen addition: item-level structure in the encoder of the chain decoder (docs/MFD_REPORT.md §16).

The control's encoder sees the history as a flat stream of SID code tokens (5 per item for 1+2+2) with T5's relative
positions only (exact within ~8 tokens). Two variants, everything else identical to the control (finetune_ext.PlainT5):
  E1  every history code token gets + slot embedding (which code: f, t1, t2, i1, i2; from the token's namespace
      <s0_/<t0_/<t1_/<v0_/<v1_) + recency embedding (items back from the end, 0..19). Both zero-initialised, so the
      model starts exactly at the control. Non-code tokens (</s>, pad) get nothing.
  E2  one encoder position per history item: the sum of its code embeddings (uses the SID's additive structure) +
      the recency embedding (zero-initialised); the encoder sequence is the number of items, so T5's relative
      positions count items. </s> is dropped. The decoder and its cross-attention mask follow the item sequence.
Items are found from the tokens themselves (an item starts at its <s0_..> token), so any order / padding side works.
ItemMFDT5 is E2 on the MFD decoder (§17 P1a; evaluation: mfd_eval.load_mfd_model).
The struct parameters live in their own submodule (`struct`); embeddings stay tied as in the control. The variant is
stored in config.struct_mode, so StructT5.from_pretrained rebuilds it (evaluation: eval_struct.py).
"""
import re

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput

from finetune_ext import PlainT5
from mfd_model import MFDT5

CODE_TOKEN = re.compile(r"^<([stv])(\d+)_(\d+)>$")
SLOTS = ("s0", "t0", "t1", "v0", "v1")
OTHER = len(SLOTS)
MAX_ITEMS = 20


class StructInput(nn.Module):
    def __init__(self, vocab, d_model):
        super().__init__()
        self.slot = nn.Embedding(len(SLOTS) + 1, d_model)
        self.rec = nn.Embedding(MAX_ITEMS, d_model)
        nn.init.zeros_(self.slot.weight)
        nn.init.zeros_(self.rec.weight)
        self.register_buffer("slot_of", torch.full((vocab,), OTHER, dtype=torch.long))

    def set_table(self, tokenizer):
        table = torch.full((len(tokenizer),), OTHER, dtype=torch.long)
        for tok, i in tokenizer.get_vocab().items():
            m = CODE_TOKEN.match(tok)
            if m and m.group(1) + m.group(2) in SLOTS:
                table[i] = SLOTS.index(m.group(1) + m.group(2))
        self.slot_of = table.to(self.slot_of.device)

    def structure(self, ids):
        s = self.slot_of[ids]
        code = s < OTHER
        start = s == 0
        item = torch.cumsum(start.long(), 1) - 1                     # item index of each token (0 = oldest)
        n = start.sum(1, keepdim=True)
        rec = (n - 1 - item).clamp(0, MAX_ITEMS - 1)
        return s, code & (item >= 0), item, rec, n.squeeze(1)


class StructMixin:
    def __init__(self, config):
        super().__init__(config)
        self.struct = StructInput(config.vocab_size, config.d_model)
        self.struct_mode = getattr(config, "struct_mode", None)

    def install_struct(self, tokenizer, mode):
        assert mode in ("E1", "E2")
        dev = next(self.parameters()).device
        self.struct = StructInput(len(tokenizer), self.config.d_model).to(dev)
        self.struct.set_table(tokenizer)
        self.struct_mode = mode
        self.config.struct_mode = mode
        return {"struct_mode": mode, "slot_tokens": {k: int((self.struct.slot_of == j).sum()) for j, k in enumerate(SLOTS)},
                "max_items": MAX_ITEMS, "zero_init": True}

    def struct_encode(self, input_ids, attention_mask):
        emb = self.shared(input_ids)
        s, code, item, rec, n = self.struct.structure(input_ids)
        if self.struct_mode == "E1":
            x = emb + (self.struct.slot(s) + self.struct.rec(rec)) * code[..., None].to(emb.dtype)
            mask = attention_mask
        else:                                                         # E2: one position per item
            B, _, d = emb.shape
            nmax = max(int(n.max()), 1)
            idx = item.clamp(min=0)[..., None].expand(-1, -1, d)
            x = torch.zeros(B, nmax, d, device=emb.device, dtype=emb.dtype).scatter_add(
                1, idx, emb * code[..., None].to(emb.dtype))
            j = torch.arange(nmax, device=emb.device)[None]
            mask = (j < n[:, None]).long()
            x = x + self.struct.rec((n[:, None] - 1 - j).clamp(0, MAX_ITEMS - 1)) * mask[..., None].to(emb.dtype)
        out = self.encoder(inputs_embeds=x, attention_mask=mask, return_dict=True)
        return BaseModelOutput(last_hidden_state=out.last_hidden_state), mask


class StructT5(StructMixin, PlainT5):
    def forward(self, input_ids=None, attention_mask=None, labels=None, decoder_input_ids=None,
                decoder_attention_mask=None, task_flag=None, encoder_outputs=None, **kwargs):
        if encoder_outputs is None and input_ids is not None and self.struct_mode is not None:
            encoder_outputs, attention_mask = self.struct_encode(input_ids, attention_mask)
            input_ids = None
        return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                               decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask,
                               encoder_outputs=encoder_outputs, **kwargs)

    def _prepare_encoder_decoder_kwargs_for_generation(self, inputs_tensor, model_kwargs, model_input_name, generation_config):
        if self.struct_mode is None:
            return super()._prepare_encoder_decoder_kwargs_for_generation(inputs_tensor, model_kwargs, model_input_name,
                                                                          generation_config)
        enc, mask = self.struct_encode(inputs_tensor, model_kwargs["attention_mask"])
        model_kwargs["encoder_outputs"] = enc
        model_kwargs["attention_mask"] = mask                        # E2: the item-level mask for cross-attention
        return model_kwargs


class ItemMFDT5(StructMixin, MFDT5):
    """MFD with one encoder position per history item (E2; §17 P1a): each item is the sum of the code embeddings present
    in its history view (5 in the full view, 3 in the text / image views of seqrec_mv) + recency. Decoder rows unchanged."""

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, labels=None,
                encoder_outputs=None, **kwargs):
        if encoder_outputs is None and input_ids is not None and self.struct_mode is not None:
            encoder_outputs, attention_mask = self.struct_encode(input_ids, attention_mask)
            input_ids = None
        return super().forward(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                               labels=labels, encoder_outputs=encoder_outputs, **kwargs)
