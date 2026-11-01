"""mmgen addition: data for MFD (mfd_model.py), SIDs [f; t_1..t_L; i_1..i_L] (L = 2 for 1+2+2, 3 for 1+3+3).

MFDDataset wraps MACRec's SeqRecDataset (task seqrec, same splits, same history of full SIDs as the
chain decoder) and returns the target as an item id; MFDCollator tokenizes the history exactly like
MACRec's Collator and builds the two decoder sequences. Catalog holds every item's codes and the
prefix tables used for exact scoring at inference.
SID layout is read from the index tokens: <s*> shared, <t*> text levels, <v*> image levels.

Multi-view history augmentation (MFD Part C, `--tasks seqrec_mv`): MVMFDDataset (and, for the chain
decoder, mv_aug_data.MVSeqRecDataset) holds every training example three times, with the whole history
shown as the full view [f t i], the text view [f t] or the image view [f i]. The target is always the
next item's full SID. Validation/test use the full view only.
C's task mix (`--tasks mt_*`, mfd_mt_data.py): load_mfd_datasets builds those datasets; labels may then be
(item id, "text" | "image"), and MFDCollator masks the loss of the other branch.
"""
import json
import os

import torch
from torch.utils.data import ConcatDataset

from data import SeqRecDataset
from mfd_model import BRANCH_TOKENS

VIEWS = ("full", "text", "image")


def sid_parts(tokens):
    """(f, [t_1..t_L], [i_1..i_L]) from one item's token list."""
    s = [x for x in tokens if x.startswith("<s")]
    t = [x for x in tokens if x.startswith("<t")]
    v = [x for x in tokens if x.startswith("<v")]
    assert len(s) == 1 and len(t) == len(v) >= 1 and len(tokens) == 1 + 2 * len(t), tokens
    return s[0], t, v


def view_maps(indices):
    """full SID string -> history string of each view."""
    out = {v: {} for v in VIEWS}
    for toks in indices.values():
        s, t, v = sid_parts(toks)
        full = "".join(toks)
        out["full"][full] = full
        out["text"][full] = s + "".join(t)
        out["image"][full] = s + "".join(v)
    return out


def load_index(args):
    return json.load(open(os.path.join(args.data_path, args.dataset, args.dataset + args.index_file)))


class MFDDataset(SeqRecDataset):
    task = "mfd"

    def __init__(self, args, mode="train"):
        super().__init__(args, task="seqrec", mode=mode)
        self.sid2item = {"".join(v): int(k) for k, v in self.indices.items()}
        assert len(self.sid2item) == len(self.indices), "SIDs must be unique"

    def get_new_tokens(self):
        return sorted(set(super().get_new_tokens()) | set(BRANCH_TOKENS))

    def __getitem__(self, index):
        d = self.inter_data[index]
        return dict(input_ids="".join(d["inters"]), labels=self.sid2item[d["item"]])


class MVMFDDataset(MFDDataset):
    """Training examples x 3 views (VIEWS): example index % n, view index // n."""
    task = "mfd_mv"

    def __init__(self, args, mode="train"):
        assert mode == "train"
        super().__init__(args, mode)
        self.task = "seqrec_mv"                                          # name in the training stats file
        self.vmap = view_maps(self.indices)

    def __len__(self):
        return len(VIEWS) * len(self.inter_data)

    def __getitem__(self, index):
        n = len(self.inter_data)
        d, view = self.inter_data[index % n], VIEWS[index // n]
        return dict(input_ids="".join(self.vmap[view][h] for h in d["inters"]), labels=self.sid2item[d["item"]])


def load_mfd_datasets(args):
    import mfd_mt_data                                                   # (imports this module)
    if mfd_mt_data.is_mt(args.tasks):                                    # C's task mix (mfd_mt_data.py)
        train = [mfd_mt_data.build(args, t) for t in args.tasks.split(",")]
        return ConcatDataset(train), ConcatDataset([mfd_mt_data.MTSeqDataset(args, mfd_mt_data.EVAL_TASK, "valid")])
    train = MVMFDDataset(args, "train") if "seqrec_mv" in args.tasks.split(",") else MFDDataset(args, "train")
    return ConcatDataset([train]), ConcatDataset([MFDDataset(args, "valid")])


def item_token_table(args, tokenizer):
    """(N, 1 + 2L) token ids [f, t_1..t_L, i_1..i_L] indexed by item id."""
    idx = load_index(args)
    width = len(next(iter(idx.values())))
    tab = torch.zeros(len(idx), width, dtype=torch.long)
    for k, v in idx.items():
        s, t, i = sid_parts(v)
        ids = tokenizer.convert_tokens_to_ids([s] + t + i)
        assert tokenizer.unk_token_id not in ids, v
        tab[int(k)] = torch.tensor(ids)
    return tab


def branch_tensors(tab, items, T, I, start=0):
    """decoder inputs / labels (B, 2, L + 2) for item ids `items` (tensor):
    text  [start, f, <T>, t_1..t_{L-1}] -> [f, -, t_1..t_L];  image [start, f, <I>, i_1..i_{L-1}] -> [-, -, i_1..i_L]."""
    c = tab[items]
    L = (c.shape[1] - 1) // 2
    f, t, i = c[:, 0], c[:, 1:1 + L], c[:, 1 + L:]
    z = torch.full_like(f, start)[:, None]
    ign = torch.full_like(f, -100)[:, None]
    dec = torch.stack([torch.cat([z, f[:, None], torch.full_like(z, T), t[:, :-1]], 1),
                       torch.cat([z, f[:, None], torch.full_like(z, I), i[:, :-1]], 1)], 1)
    lab = torch.stack([torch.cat([f[:, None], ign, t], 1),
                       torch.cat([ign, ign, i], 1)], 1)                   # f counted once (text branch)
    return dec, lab


class MFDCollator:
    def __init__(self, args, tokenizer):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        self.tab = item_token_table(args, tokenizer)
        self.T, self.I = tokenizer.convert_tokens_to_ids(list(BRANCH_TOKENS))

    def __call__(self, batch):
        inputs = self.tokenizer([d["input_ids"] for d in batch], return_tensors="pt", padding="longest",
                                max_length=self.tokenizer.model_max_length, truncation=True,
                                return_attention_mask=True)
        lb = [d["labels"] for d in batch]
        items = [x[0] if isinstance(x, (tuple, list)) else x for x in lb]
        dec, lab = branch_tensors(self.tab, torch.tensor(items), self.T, self.I)
        for n, x in enumerate(lb):           # single-branch task (mfd_mt_data.py): no loss on the other branch's codes
            if isinstance(x, (tuple, list)):
                if x[1] == "text":
                    lab[n, 1] = -100
                else:
                    assert x[1] == "image", x
                    lab[n, 0, 1:] = -100                                 # keep f (position 0) as the one f target
        inputs["decoder_input_ids"] = dec
        inputs["labels"] = lab
        return inputs


class Catalog:
    """Codes of every item as per-level local indices, plus CSR tables for exact scoring.
    Levels: 0 = f, 1..L = text, L+1..2L = image. For each shared code f: its items (sorted by item id) and,
    per branch and depth d = 1..L-1, its distinct catalog prefixes (f, c_1..c_d), sorted by code tuple.
    pref[(b, d)]: code (P, d) local codes, cnt/ptr per f, item_local = each item's prefix index within its f."""

    def __init__(self, args, tokenizer, device):
        tab = item_token_table(args, tokenizer)                          # (N, 1 + 2L) token ids
        self.N = tab.shape[0]
        self.L = (tab.shape[1] - 1) // 2
        self.levels = {"t": list(range(1, 1 + self.L)), "i": list(range(1 + self.L, 1 + 2 * self.L))}
        self.tok, loc = [], []
        for j in range(tab.shape[1]):
            u, inv = torch.unique(tab[:, j], return_inverse=True)       # codes used at this level
            self.tok.append(u.to(device)); loc.append(inv)
        self.code = torch.stack(loc, 1)                                  # (N, 1 + 2L) local code index
        f = self.code[:, 0]
        nF = len(self.tok[0])
        self.nF = nF
        order = torch.argsort(f * self.N + torch.arange(self.N))         # by f, then item id
        self.items_by_f = order.to(device)
        self.f_item_cnt = torch.bincount(f, minlength=nF).to(device)
        self.f_item_ptr = (torch.cumsum(self.f_item_cnt, 0) - self.f_item_cnt)
        self.pref = {}
        for b, lv in self.levels.items():
            for d in range(1, self.L):
                key = self.code[:, [0] + lv[:d]]                         # (N, 1 + d)
                uk, inv = torch.unique(key, dim=0, return_inverse=True)  # lexicographic: by f, then codes
                cnt = torch.bincount(uk[:, 0], minlength=nF)
                ptr = torch.cumsum(cnt, 0) - cnt
                self.pref[(b, d)] = dict(code=uk[:, 1:].to(device), cnt=cnt.to(device), ptr=ptr.to(device),
                                         item_local=(inv - ptr[f]).to(device))
        self.code = self.code.to(device)
        self.item_tok = tab.to(device)
