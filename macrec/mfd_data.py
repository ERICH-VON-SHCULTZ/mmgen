"""mmgen addition: data for MFD stage 1 (mfd_model.py).

MFDDataset wraps MACRec's SeqRecDataset (task seqrec, same splits, same history of full SIDs as the
chain decoder) and returns the target as an item id; MFDCollator tokenizes the history exactly like
MACRec's Collator and builds the two decoder sequences. Catalog holds every item's codes and the
prefix tables used for exact scoring at inference.
SID layout is read from the index tokens: <s*> shared, <t*> text levels, <v*> image levels.
"""
import json
import os

import torch
from torch.utils.data import ConcatDataset

from data import SeqRecDataset
from mfd_model import BRANCH_TOKENS


def sid_parts(tokens):
    s = [x for x in tokens if x.startswith("<s")]
    t = [x for x in tokens if x.startswith("<t")]
    v = [x for x in tokens if x.startswith("<v")]
    assert len(s) == 1 and len(t) == 2 and len(v) == 2 and len(tokens) == 5, tokens
    return s[0], t[0], t[1], v[0], v[1]


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


def load_mfd_datasets(args):
    return ConcatDataset([MFDDataset(args, "train")]), ConcatDataset([MFDDataset(args, "valid")])


def item_token_table(args, tokenizer):
    """(N, 5) token ids [f, t1, t2, i1, i2] indexed by item id."""
    idx = load_index(args)
    tab = torch.zeros(len(idx), 5, dtype=torch.long)
    for k, v in idx.items():
        ids = tokenizer.convert_tokens_to_ids(list(sid_parts(v)))
        assert tokenizer.unk_token_id not in ids, v
        tab[int(k)] = torch.tensor(ids)
    return tab


def branch_tensors(tab, items, T, I, start=0):
    """decoder inputs / labels (B, 2, 4) for item ids `items` (tensor)."""
    c = tab[items]
    f, t1, t2, i1, i2 = c.unbind(-1)
    z = torch.full_like(f, start)
    ign = torch.full_like(f, -100)
    dec = torch.stack([torch.stack([z, f, torch.full_like(f, T), t1], -1),
                       torch.stack([z, f, torch.full_like(f, I), i1], -1)], 1)
    lab = torch.stack([torch.stack([f, ign, t1, t2], -1),
                       torch.stack([ign, ign, i1, i2], -1)], 1)          # f counted once (text branch)
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
        dec, lab = branch_tensors(self.tab, torch.tensor([d["labels"] for d in batch]), self.T, self.I)
        inputs["decoder_input_ids"] = dec
        inputs["labels"] = lab
        return inputs


class Catalog:
    """Codes of every item as per-level local indices, plus CSR tables for exact scoring:
    for each shared code f, its items (sorted by item id) and its distinct (f, t1) and (f, i1) prefixes."""

    def __init__(self, args, tokenizer, device):
        tab = item_token_table(args, tokenizer)                          # (N, 5) token ids
        self.N = tab.shape[0]
        self.tok, loc = [], []
        for j in range(5):
            u, inv = torch.unique(tab[:, j], return_inverse=True)       # codes used at this level
            self.tok.append(u.to(device)); loc.append(inv)
        self.code = torch.stack(loc, 1)                                  # (N, 5) local code index
        f = self.code[:, 0]
        nF = len(self.tok[0])
        order = torch.argsort(f * self.N + torch.arange(self.N))         # by f, then item id
        self.items_by_f = order.to(device)
        self.f_item_cnt = torch.bincount(f, minlength=nF).to(device)
        self.f_item_ptr = (torch.cumsum(self.f_item_cnt, 0) - self.f_item_cnt)
        self.pref = {}
        for name, lvl in (("t", 1), ("i", 3)):
            key = f * len(self.tok[lvl]) + self.code[:, lvl]
            uk, inv = torch.unique(key, return_inverse=True)              # sorted by f, then code
            pf, pc = uk // len(self.tok[lvl]), uk % len(self.tok[lvl])
            cnt = torch.bincount(pf, minlength=nF)
            ptr = torch.cumsum(cnt, 0) - cnt
            self.pref[name] = dict(code=pc.to(device), cnt=cnt.to(device), ptr=ptr.to(device),
                                   item_local=(inv - ptr[f]).to(device))  # item's prefix index within its f
        self.code = self.code.to(device)
        self.item_tok = tab.to(device)
