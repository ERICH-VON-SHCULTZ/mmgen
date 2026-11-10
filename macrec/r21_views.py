"""mmgen addition (docs/MFD_REPORT.md §21): training-side view methods on top of P1a (enc_struct.ItemMFDT5 + codebook init).

G0  grouped views: one training example = one (history, target) pair carrying its three history views (full / text /
    image, mfd_data.VIEWS), encoded in the same step (GroupedMVDataset + GroupedCollator: rows 3k, 3k+1, 3k+2 of a batch
    are the full / text / image view of pair k, same target). Loss = MEAN over the 3 views of P1a's loss (the mean CE over
    the 5 targets f, t1, t2, i1, i2), averaged over pairs = P1a's token-mean CE on the same 3P sequences (P1a's loss scale:
    P1a clips gradients at max_grad_norm 1.0, so the loss scale is not free).
A   G0 (mean over views) + beta * KL( stopgrad(pbar) || p_full ),  pbar = (p_full + p_text + p_image) / 3 (probability space), T = 1.
B   G0 + beta * mean over the 3 view pairs (u, v) of [KL(p_u || p_v) + KL(p_v || p_u)] / 2, gradients to both sides.
C   P1a's layout (3 independent examples per pair per epoch, shuffled): every history item independently shows the full
    view (5 codes) with probability p_full, else the text or the image view (equal probability); redrawn every time
    the example is built. Model and loss = P1a (ItemRandViewDataset only).
p_v = the decoder's softmax over the shared codes (<s0_*> tokens) at the position that predicts f (text branch, position
0: the logits the f loss uses; f is counted once). The extra terms are added only in training (model.training); evaluation
loss, early stopping and the saved model are P1a's (validation = full view, MFDDataset). The terms are averaged over pairs.
"""
import torch
import torch.nn.functional as F
from transformers.modeling_outputs import Seq2SeqLMOutput

from enc_struct import ItemMFDT5
from mfd_data import VIEWS, MFDCollator, MFDDataset, view_maps

MODES = ("G0", "A", "B", "C")


class GroupedMVDataset(MFDDataset):
    """One example per (history, target) pair; input_ids = the three view strings (VIEWS order)."""
    task = "mfd_mv_grouped"

    def __init__(self, args, mode="train"):
        assert mode == "train"
        super().__init__(args, mode)
        self.task = "seqrec_mv_grouped"
        self.vmap = view_maps(self.indices)

    def __getitem__(self, index):
        d = self.inter_data[index]
        return dict(input_ids=["".join(self.vmap[v][h] for h in d["inters"]) for v in VIEWS], labels=self.sid2item[d["item"]])


class ItemRandViewDataset(MFDDataset):
    """3 examples per pair per epoch (index % n = pair, as MVMFDDataset); each history item draws its view independently."""
    task = "mfd_itemrand"

    def __init__(self, args, mode="train", p_full=1 / 3):
        assert mode == "train" and 0 < p_full < 1
        super().__init__(args, mode)
        self.task = "seqrec_itemrand"
        self.vmap = view_maps(self.indices)
        self.p_full = p_full

    def __len__(self):
        return len(VIEWS) * len(self.inter_data)

    def draw(self, k):
        u = torch.rand(k)                       # torch RNG: DataLoader workers are reseeded every epoch
        p2 = self.p_full + (1 - self.p_full) / 2
        return [VIEWS[0] if x < self.p_full else (VIEWS[1] if x < p2 else VIEWS[2]) for x in u.tolist()]

    def __getitem__(self, index):
        d = self.inter_data[index % len(self.inter_data)]
        views = self.draw(len(d["inters"]))
        return dict(input_ids="".join(self.vmap[v][h] for v, h in zip(views, d["inters"])), labels=self.sid2item[d["item"]])


class GroupedCollator(MFDCollator):
    """Grouped examples -> 3 consecutive rows per pair; plain (validation) examples unchanged."""

    def __call__(self, batch):
        if batch and isinstance(batch[0]["input_ids"], (list, tuple)):
            batch = [dict(input_ids=s, labels=d["labels"]) for d in batch for s in d["input_ids"]]
        return super().__call__(batch)


def f_logprobs_train(logits, fcols):
    """(R, 2, L, V) logits -> (R, nF) log-softmax over the shared codes at the f position (text branch, position 0)."""
    return logits[:, 0, 0, fcols].float().log_softmax(-1)


def kl(logp, logq):
    """KL(p || q) per row, from log-probabilities."""
    return (logp.exp() * (logp - logq)).sum(-1)


def aux_loss(lpf, mode):
    """lpf (P, 3, nF) view log-probs (VIEWS order). A: KL(stopgrad(pbar) || p_full); B: symmetric KL over view pairs."""
    if mode == "A":
        pbar = lpf.exp().mean(1).detach()
        lbar = pbar.clamp_min(1e-30).log()
        return kl(lbar, lpf[:, 0]).mean()
    if mode == "B":
        terms = [(kl(lpf[:, u], lpf[:, v]) + kl(lpf[:, v], lpf[:, u])) / 2 for u, v in ((0, 1), (0, 2), (1, 2))]
        return torch.stack(terms, 1).mean()
    raise ValueError(mode)


class GroupedItemMFDT5(ItemMFDT5):
    """P1a's model; in training with grouped batches (mode G0/A/B) the loss is the mean over views of P1a's loss + beta * aux."""
    r21_mode, r21_beta = None, 0.0

    def install_r21(self, tokenizer, mode, beta):
        assert mode in ("G0", "A", "B") and (beta > 0) == (mode != "G0")
        fcols = [i for t, i in sorted(tokenizer.get_vocab().items(), key=lambda x: x[1]) if t.startswith("<s0_")]
        self.register_buffer("r21_fcols", torch.tensor(fcols, device=next(self.parameters()).device), persistent=False)
        self.r21_mode, self.r21_beta = mode, beta
        self.r21_log = {"steps": 0, "ce": 0.0, "aux": 0.0}
        return {"r21_mode": mode, "r21_beta": beta, "n_f_codes": len(fcols)}

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, labels=None,
                encoder_outputs=None, **kwargs):
        if not (self.training and self.r21_mode and labels is not None):
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                                   labels=labels, encoder_outputs=encoder_outputs, **kwargs)
        out = super().forward(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                              labels=None, encoder_outputs=encoder_outputs, **kwargs)
        lg = out.logits                                                     # (3P, 2, L, V)
        R = lg.shape[0]
        assert R % 3 == 0, R
        assert torch.equal(labels[0::3], labels[1::3]) and torch.equal(labels[0::3], labels[2::3]), "rows 3k..3k+2 = one pair"
        ce = F.cross_entropy(lg.reshape(-1, lg.size(-1)).float(), labels.reshape(-1), ignore_index=-100)
        loss, aux = ce, torch.zeros((), device=lg.device)
        if self.r21_mode in ("A", "B"):
            aux = aux_loss(f_logprobs_train(lg, self.r21_fcols).view(R // 3, 3, -1), self.r21_mode)
            loss = ce + self.r21_beta * aux
        g = self.r21_log
        g["steps"] += 1; g["ce"] += ce.item(); g["aux"] += aux.item()
        if g["steps"] % 200 == 0:
            print(f"[r21] micro-batches {g['steps']}: mean CE (mean over views) {g['ce'] / 200:.4f}, mean aux {g['aux'] / 200:.4f}", flush=True)
            g["ce"] = g["aux"] = 0.0
        return Seq2SeqLMOutput(loss=loss, logits=lg)
