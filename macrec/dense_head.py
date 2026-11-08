"""mmgen addition (docs/MFD_REPORT.md §20 Task 2): a dense item-level score head on P1a (enc_struct.ItemMFDT5).

  h       = decoder hidden state at the start position (the state whose LM head predicts f), final layer (after T5's final
            layer norm), through a new Linear(d, d)
  e_item  = sum of the item's 5 code embeddings from the shared, tied input table (during training the codebook-initialised
            GeoEmbedding, after baking the plain table it becomes); no recency, no new item table
  s(u, i) = exp(log_scale) * <h, e_item>, log_scale learnable, init log(DENSE_SCALE0) so the initial logits are small
  L       = L_MFD + lambda * CE(softmax over every catalog item, target item), on every training example (all three views;
            the target is the same). TRAINING ONLY: in eval mode the loss is plain L_MFD, so validation loss and early
            stopping are exactly P1a's.
The target item is recovered from the MFD label codes (text row [f, -, t1, t2], image row [-, -, i1, i2]) through a sorted
key of the catalog's per-level local codes (asserted). config.dense_head / dense_lambda / dense_n_items let
mfd_eval.load_mfd_model rebuild the class.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers import T5ForConditionalGeneration

from enc_struct import ItemMFDT5

DENSE_SCALE0 = 0.05


class DenseItemMFDT5(ItemMFDT5):
    def __init__(self, config):
        super().__init__(config)
        d = config.d_model
        n = getattr(config, "dense_n_items", 1)
        self.dense_proj = nn.Linear(d, d)
        self.dense_log_scale = nn.Parameter(torch.tensor(math.log(DENSE_SCALE0)))
        self.dense_lambda = getattr(config, "dense_lambda", 0.0)
        self.register_buffer("dense_item_tok", torch.zeros(n, 5, dtype=torch.long))      # (N, 5) token ids, by item id
        self.register_buffer("dense_tok2code", torch.zeros(config.vocab_size, dtype=torch.long))
        self.register_buffer("dense_key_sorted", torch.zeros(n, dtype=torch.long))
        self.register_buffer("dense_key_item", torch.zeros(n, dtype=torch.long))

    def install_dense(self, item_tok, lam):
        """item_tok: (N, 5) token ids [f, t1, t2, i1, i2] indexed by item id (mfd_data.item_token_table)."""
        dev = next(self.parameters()).device
        N = item_tok.shape[0]
        V = self.config.vocab_size
        tok2code = torch.full((V,), -1, dtype=torch.long)
        for j in range(5):
            u = item_tok[:, j].unique()
            tok2code[u] = torch.arange(len(u))
        assert int(tok2code.max()) < 256
        key = self._key(tok2code[item_tok])
        ks, order = key.sort()
        assert bool((ks[1:] > ks[:-1]).all()), "SIDs must be unique"
        self.dense_item_tok = item_tok.to(dev)
        self.dense_tok2code = tok2code.to(dev)
        self.dense_key_sorted = ks.to(dev)
        self.dense_key_item = order.to(dev)
        self.dense_lambda = float(lam)
        self.config.dense_head = True
        self.config.dense_lambda = float(lam)
        self.config.dense_n_items = int(N)
        return {"dense_lambda": float(lam), "catalog_items": int(N), "scale0": DENSE_SCALE0, "head": "Linear(d, d) on decoder start state"}

    @staticmethod
    def _key(codes):
        k = torch.zeros(codes.shape[0], dtype=torch.long, device=codes.device)
        for j in range(5):
            k = k * 256 + codes[:, j]
        return k

    def items_from_labels(self, labels):
        """labels (B, 2, 4) MFD branch labels -> target item ids (B,)."""
        tok = torch.stack([labels[:, 0, 0], labels[:, 0, 2], labels[:, 0, 3], labels[:, 1, 2], labels[:, 1, 3]], 1)
        key = self._key(self.dense_tok2code[tok])
        pos = torch.searchsorted(self.dense_key_sorted, key).clamp(max=len(self.dense_key_sorted) - 1)
        assert bool((self.dense_key_sorted[pos] == key).all()), "label codes do not match a catalog item"
        return self.dense_key_item[pos]

    def item_vectors(self):
        return self.shared(self.dense_item_tok).sum(1)                                  # (N, d)

    def dense_logits(self, h_start):
        """h_start (B, d) final decoder state at the start position -> (B, N) dense logits over the catalog."""
        return self.dense_log_scale.exp() * self.dense_proj(h_start) @ self.item_vectors().t()

    @torch.no_grad()
    def start_state(self, enc_h, enc_mask):
        B = enc_h.shape[0]
        start = torch.full((B, 1), self.config.decoder_start_token_id, device=enc_h.device)
        o = T5ForConditionalGeneration.forward(self, encoder_outputs=(enc_h,), attention_mask=enc_mask, decoder_input_ids=start,
                                               use_cache=False, return_dict=True, output_hidden_states=True)
        return o.decoder_hidden_states[-1][:, 0]

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, labels=None,
                encoder_outputs=None, **kwargs):
        dense = labels is not None and self.training and self.dense_lambda > 0
        if not dense:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                                   labels=labels, encoder_outputs=encoder_outputs, **kwargs)
        if encoder_outputs is None:
            encoder_outputs, attention_mask = self.struct_encode(input_ids, attention_mask)
        B, nb, L = decoder_input_ids.shape
        h = encoder_outputs[0].repeat_interleave(nb, 0)
        m = attention_mask.repeat_interleave(nb, 0)
        out = T5ForConditionalGeneration.forward(self, encoder_outputs=(h,), attention_mask=m,
                                                 decoder_input_ids=decoder_input_ids.reshape(B * nb, L), use_cache=False,
                                                 return_dict=True, output_hidden_states=True)
        logits = out.logits.view(B, nb, L, -1)
        l_mfd = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1), ignore_index=-100)
        hs = out.decoder_hidden_states[-1].view(B, nb, L, -1)[:, 0, 0]                  # start position, text row
        l_dense = F.cross_entropy(self.dense_logits(hs).float(), self.items_from_labels(labels))
        self._last_losses = (l_mfd.detach(), l_dense.detach())
        return Seq2SeqLMOutput(loss=l_mfd + self.dense_lambda * l_dense, logits=logits)
