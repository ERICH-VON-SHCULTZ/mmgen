"""mmgen addition (docs/MFD_REPORT.md §22.10, E10c): A01's teacher KL WITHOUT grouping, on P1a's exact data layout and optimisation path.
Data: P1a's MVMFDDataset unchanged (3 independent view-examples per pair per epoch, index % n = pair, index // n = view, shuffled
by the Trainer's sampler; 256 rows x 4 micro-batches, same clipping). E10cDataset only ATTACHES the text-view and image-view history
strings of the same pair to every full-view row (input_ids = [full, text, image]); E10cCollator builds P1a's batch with
MFDCollator unchanged (identical input_ids / labels / decoder inputs) and adds the extra inputs of the full rows:
  e10_rows (indices of the full rows in the micro-batch), e10_text_ids / e10_text_mask, e10_image_ids / e10_image_mask.
Loss (training only): P1a's loss (token-mean CE of the micro-batch, unchanged) + (1 / R) * sum over full rows of 3 beta * KL(pbar || p_full)
  p_v      = softmax over the f codes (<s0_*>) at the f position (decoder position 0, text branch: the logits P1a's f loss uses)
  p_text, p_image: two EXTRA forwards of the same model in train mode (dropout on) under torch.no_grad() on the text / image view
             of the same pair, decoder input = [start] only (position 0 of a causal decoder = position 0 of the full sequence)
  pbar     = (stopgrad(p_full) + p_text + p_image) / 3 (probability space, T = 1)
  R        = rows of the micro-batch; text / image rows get weight 0. Full rows are 1/3 of the rows in expectation, so the expected
             per-pair weight of the KL is beta, A01's per-pair weight (A01: mean over 3 views of the CE + beta * KL per pair).
The main forward runs FIRST, so its dropout masks and its loss are P1a's for the same RNG state; the extra forwards only
consume RNG afterwards. No new parameters: the saved (baked) model is a plain P1a-layout model.
"""
import math

import torch
from transformers.modeling_outputs import Seq2SeqLMOutput

from enc_struct import ItemMFDT5
from mfd_data import VIEWS, MFDCollator, MVMFDDataset
from r21_views import kl

EXTRA = ("e10_rows", "e10_text_ids", "e10_text_mask", "e10_image_ids", "e10_image_mask")


class E10cDataset(MVMFDDataset):
    """P1a's MVMFDDataset (same length, order and rows); a full-view row's input_ids is the list [full, text, image] of its pair's
    history strings (the HF Trainer drops dataset keys that are not forward() arguments, so the extra views ride in input_ids)."""

    def __getitem__(self, index):
        d = super().__getitem__(index)
        n = len(self.inter_data)
        if VIEWS[index // n] == "full":
            h = self.inter_data[index % n]["inters"]
            d["input_ids"] = [d["input_ids"]] + ["".join(self.vmap[v][x] for x in h) for v in ("text", "image")]
        return d


class E10cCollator(MFDCollator):
    """P1a's batch from MFDCollator on the main strings (identical tensors), + e10_* tensors of the full rows (training batches)."""

    def __call__(self, batch):
        train = any(isinstance(d["input_ids"], (list, tuple)) for d in batch)
        out = super().__call__([dict(input_ids=d["input_ids"][0] if isinstance(d["input_ids"], (list, tuple)) else d["input_ids"],
                                     labels=d["labels"]) for d in batch])
        rows = [j for j, d in enumerate(batch) if isinstance(d["input_ids"], (list, tuple))]
        if not train:
            return out                                                   # validation (MFDDataset) or a batch without full rows
        out["e10_rows"] = torch.tensor(rows, dtype=torch.long)
        for k, v in (("text", 1), ("image", 2)):
            t = self.tokenizer([batch[j]["input_ids"][v] for j in rows], return_tensors="pt", padding="longest",
                               max_length=self.tokenizer.model_max_length, truncation=True, return_attention_mask=True)
            out[f"e10_{k}_ids"], out[f"e10_{k}_mask"] = t["input_ids"], t["attention_mask"]
        return out


class E10cMixin:
    e10_beta = None

    def install_e10c(self, tokenizer, beta):
        assert beta >= 0
        fcols = [i for t, i in sorted(tokenizer.get_vocab().items(), key=lambda x: x[1]) if t.startswith("<s0_")]
        self.register_buffer("e10_fcols", torch.tensor(fcols, device=next(self.parameters()).device), persistent=False)
        self.e10_beta = float(beta)
        self.e10_log = {"micro_batches": 0, "full_rows": 0, "rows": 0, "kl_sum": 0.0, "term_sum": 0.0,
                        "win": {"n": 0, "kl": 0.0, "term": 0.0}}
        self.e10_last = None
        return {"e10c_beta": self.e10_beta, "n_f_codes": len(fcols), "kl_weight_full_rows": 3 * self.e10_beta}

    def e10_view_logprobs(self, ids, mask):
        """log P(f) over the f codes at position 0 for one view's history (decoder input [start] only)."""
        B = ids.shape[0]
        start = torch.full((B, 1, 1), self.config.decoder_start_token_id, device=ids.device, dtype=torch.long)
        lg = ItemMFDT5.forward(self, input_ids=ids, attention_mask=mask, decoder_input_ids=start).logits
        return lg[:, 0, 0, self.e10_fcols].float().log_softmax(-1)

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, labels=None, encoder_outputs=None,
                e10_rows=None, e10_text_ids=None, e10_text_mask=None, e10_image_ids=None, e10_image_mask=None, **kwargs):
        out = super().forward(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                              labels=labels, encoder_outputs=encoder_outputs, **kwargs)
        if not (self.training and self.e10_beta is not None and labels is not None and e10_rows is not None):
            return out
        R = out.logits.shape[0]
        term = out.loss.new_zeros(())
        if e10_rows.numel():                                           # (a micro-batch without full rows adds 0)
            lpf = out.logits[e10_rows, 0, 0][:, self.e10_fcols].float().log_softmax(-1)
            with torch.no_grad():
                lpt = self.e10_view_logprobs(e10_text_ids, e10_text_mask)
                lpi = self.e10_view_logprobs(e10_image_ids, e10_image_mask)
                lbar = torch.logsumexp(torch.stack([lpf.detach(), lpt, lpi]), 0) - math.log(3)
            klr = kl(lbar, lpf)
            term = 3 * self.e10_beta * klr.sum() / R
            self.e10_last = {"lpf": lpf, "lpt": lpt, "lpi": lpi, "kl_rows": klr.detach(), "term": term.detach(), "R": R}
            g = self.e10_log
            g["full_rows"] += e10_rows.numel(); g["kl_sum"] += float(klr.detach().sum()); g["term_sum"] += float(term.detach())
            g["win"]["n"] += 1; g["win"]["kl"] += float(klr.detach().mean()); g["win"]["term"] += float(term.detach())
        g = self.e10_log
        g["micro_batches"] += 1; g["rows"] += R
        if g["micro_batches"] % 200 == 0 and g["win"]["n"]:
            w = g["win"]
            print(f"[e10c] micro-batches {g['micro_batches']}: mean KL(pbar||p_full) {w['kl'] / w['n']:.4f}, "
                  f"mean added term {w['term'] / w['n']:.5f}, full-row share {g['full_rows'] / g['rows']:.3f}", flush=True)
            g["win"] = {"n": 0, "kl": 0.0, "term": 0.0}
        return Seq2SeqLMOutput(loss=out.loss + term, logits=out.logits)     # new object: the Trainer reads outputs["loss"]

    def e10c_stats(self):
        g = self.e10_log
        return {"beta": self.e10_beta, "micro_batches": g["micro_batches"], "full_row_share": g["full_rows"] / max(g["rows"], 1),
                "mean_kl_full_rows": g["kl_sum"] / max(g["full_rows"], 1), "mean_term": g["term_sum"] / max(g["micro_batches"], 1)}


class E10cItemMFDT5(E10cMixin, ItemMFDT5):
    pass
