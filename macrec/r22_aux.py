"""mmgen addition (docs/MFD_REPORT.md §22.2, E2 step 2): A01 + two auxiliary heads at the decoder position that predicts f.

At the start position (decoder input = decoder_start_token, the state whose output is P(f)) the final decoder hidden state
h0 also predicts the target's first text code t1 and first image code i1 MARGINALLY (not conditioned on f):
    logits_t1(c) = < A_t h0 * d_model^-0.5 , e(c) >   over the t1-level code tokens <t0_*>  (same for i1: A_i, <v0_*>)
e(c) = the model's own (tied, codebook-initialised) code embedding, i.e. exactly the main head's output row for that code;
the d_model^-0.5 scale is T5's tied-head scale; A_t, A_i (d x d, no bias) start at the identity, so at init the aux
logits are the main head's logits at that position restricted to the level's codes.
Training (grouped batches, every row = every view): L = L_A01 + gamma * [CE(t1) + CE(i1)] (means over rows). Validation
loss / early stopping = the plain MFD loss (as A01). Inference (r22_e2_eval.py): score(item) = A01 score
+ w * [log P_aux(t1(item)) + log P_aux(i1(item))], candidates unchanged (top-K_f f codes by P(f)).
The heads and the code-token lists are saved with the model (config.r22_aux, r22_aux_nt / _ni); mfd_eval.load_mfd_model
rebuilds this class.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput

from r21_views import GroupedItemMFDT5


def level_tokens(tokenizer, prefix):
    """token ids of one level's code tokens, sorted by code number (<t0_0>, <t0_1>, ...)."""
    out = []
    for t, i in tokenizer.get_vocab().items():
        if t.startswith(f"<{prefix}_") and t.endswith(">"):
            out.append((int(t[len(prefix) + 2:-1]), i))
    return [i for _, i in sorted(out)]


class AuxGroupedItemMFDT5(GroupedItemMFDT5):
    def __init__(self, config):
        super().__init__(config)
        self.r22_gamma = 0.0
        if getattr(config, "r22_aux", False):
            self._build_aux(config.r22_aux_nt, config.r22_aux_ni)

    def _build_aux(self, nt, ni):
        d = self.config.d_model
        self.aux_t = nn.Linear(d, d, bias=False)
        self.aux_i = nn.Linear(d, d, bias=False)
        nn.init.eye_(self.aux_t.weight)
        nn.init.eye_(self.aux_i.weight)
        self.register_buffer("aux_t_tok", torch.zeros(nt, dtype=torch.long))
        self.register_buffer("aux_i_tok", torch.zeros(ni, dtype=torch.long))

    def install_aux(self, tokenizer, gamma):
        tt, ti = level_tokens(tokenizer, "t0"), level_tokens(tokenizer, "v0")
        dev = next(self.parameters()).device
        self.config.r22_aux, self.config.r22_aux_nt, self.config.r22_aux_ni = True, len(tt), len(ti)
        self._build_aux(len(tt), len(ti))
        self.to(dev)
        self.aux_t_tok.copy_(torch.tensor(tt, device=dev))
        self.aux_i_tok.copy_(torch.tensor(ti, device=dev))
        self._lookup()
        self.r22_gamma = gamma
        return {"r22_aux_gamma": gamma, "n_t1_codes": len(tt), "n_i1_codes": len(ti), "head_init": "identity, tied code embeddings"}

    def _lookup(self):
        """token id -> index within the t1 / i1 code list (-1 elsewhere)."""
        n = int(self.config.vocab_size)
        dev = self.aux_t_tok.device
        self._lk_t = torch.full((n,), -1, dtype=torch.long, device=dev)
        self._lk_i = torch.full((n,), -1, dtype=torch.long, device=dev)
        self._lk_t[self.aux_t_tok] = torch.arange(len(self.aux_t_tok), device=dev)
        self._lk_i[self.aux_i_tok] = torch.arange(len(self.aux_i_tok), device=dev)

    def start_state(self, enc_h, enc_mask):
        """final decoder hidden state at the start position (the state that predicts f), as dense_head.start_state."""
        B = enc_h.shape[0]
        start = torch.full((B, 1), self.config.decoder_start_token_id, device=enc_h.device)
        o = T5ForConditionalGeneration.forward(self, encoder_outputs=(enc_h,), attention_mask=enc_mask, decoder_input_ids=start,
                                               use_cache=False, return_dict=True, output_hidden_states=True)
        return o.decoder_hidden_states[-1][:, 0]

    def aux_logits(self, h0):
        hs = h0 * (self.config.d_model ** -0.5)
        emb = self.get_input_embeddings()
        et, ei = emb(self.aux_t_tok), emb(self.aux_i_tok)
        return self.aux_t(hs) @ et.t(), self.aux_i(hs) @ ei.t()

    @torch.no_grad()
    def aux_logprobs(self, enc_h, enc_mask):
        lt, li = self.aux_logits(self.start_state(enc_h, enc_mask))
        return lt.float().log_softmax(-1), li.float().log_softmax(-1)

    def aux_targets(self, labels):
        if not hasattr(self, "_lk_t") or self._lk_t.device != labels.device:
            self._lookup()
        t1, i1 = labels[:, 0, 2], labels[:, 1, 2]                    # text row [f, -, t1, ..], image row [-, -, i1, ..]
        return self._lk_t[t1], self._lk_i[i1]

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, labels=None,
                encoder_outputs=None, **kwargs):
        if not (self.training and self.r22_gamma > 0 and labels is not None and input_ids is not None):
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                                   labels=labels, encoder_outputs=encoder_outputs, **kwargs)
        enc, m = self.struct_encode(input_ids, attention_mask)
        out = super().forward(input_ids=None, attention_mask=m, decoder_input_ids=decoder_input_ids, labels=labels,
                              encoder_outputs=enc, **kwargs)
        lt, li = self.aux_logits(self.start_state(enc.last_hidden_state, m))
        yt, yi = self.aux_targets(labels)
        assert bool((yt >= 0).all()) and bool((yi >= 0).all()), "aux targets must be level codes"
        aux = F.cross_entropy(lt.float(), yt) + F.cross_entropy(li.float(), yi)
        g = self.r21_log
        g["aux22"] = g.get("aux22", 0.0) + aux.item()
        if g["steps"] % 200 == 0 and g["steps"]:
            print(f"[r22] mean aux CE (t1 + i1) over the last 200 micro-batches {g['aux22'] / 200:.4f}", flush=True)
            g["aux22"] = 0.0
        return Seq2SeqLMOutput(loss=out.loss + self.r22_gamma * aux, logits=out.logits)
