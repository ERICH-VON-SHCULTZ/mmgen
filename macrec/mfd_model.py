"""mmgen addition: MFD, a two-branch T5 decoder for MD-RQ-VAE SIDs [f; t1 t2; i1 i2] (1+2+2 shown; for 1+3+3
each branch has one more code: [start, f, <T>, t1, t2] -> [f, -, t1, t2, t3]).

Per example the encoder runs once; its output is reused by two decoder sequences:
    text branch   decoder input [start, f, <T>, t1]   targets [f,    -,  t1, t2]
    image branch  decoder input [start, f, <I>, i1]   targets [-,    -,  i1, i2]
(`start` = T5's decoder_start_token_id; "-" = -100, not trained.) The hidden state at <T> predicts t1,
at <I> predicts i1. The branches are separate rows of one batched decoder call with T5's standard
causal mask, so neither branch can attend to the other's codes. f is a target only in the text
branch (counted once); the image branch sees the same f as input. Loss = mean token cross-entropy over
the 5 targets per example (f, t1, t2, i1, i2). Same T5 (config/ckpt) as the chain decoder.
"""
import torch
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput

BRANCH_TOKENS = ("<T>", "<I>")


class MFDT5(T5ForConditionalGeneration):

    # Explicit argument names: HF Trainer keeps only batch keys that appear in forward()'s signature.
    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, labels=None,
                encoder_outputs=None, **kwargs):
        """decoder_input_ids, labels: (B, n_branch=2, L). Returns logits (B, 2, L, V). (L = SID branch depth + 2.)"""
        B, nb, L = decoder_input_ids.shape
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        h = encoder_outputs[0].repeat_interleave(nb, 0)                 # encoder run once, shared by both branches
        m = attention_mask.repeat_interleave(nb, 0)
        out = super().forward(encoder_outputs=(h,), attention_mask=m,
                              decoder_input_ids=decoder_input_ids.reshape(B * nb, L), use_cache=False, return_dict=True)
        logits = out.logits.view(B, nb, L, -1)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1), ignore_index=-100)
        return Seq2SeqLMOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def last_logprobs(self, enc_h, enc_mask, rows, dec_ids, cols=None, chunk=8192):
        """Full-vocabulary log-softmax at the LAST decoder position of each sequence.
        enc_h/enc_mask: encoder output of the user batch; rows[n]: user of sequence n; dec_ids (N, L).
        cols: optional list of column-index tensors; returns one (N, len(c)) tensor per entry."""
        outs = [[] for _ in (cols or [None])]
        for s in range(0, dec_ids.shape[0], chunk):
            r = rows[s:s + chunk]
            o = super().forward(encoder_outputs=(enc_h[r],), attention_mask=enc_mask[r],
                                decoder_input_ids=dec_ids[s:s + chunk], use_cache=False, return_dict=True)
            lp = o.logits[:, -1].float().log_softmax(-1)
            for j, c in enumerate(cols or [None]):
                outs[j].append(lp if c is None else lp[:, c])
        res = [torch.cat(x) for x in outs]
        return res if cols else res[0]
