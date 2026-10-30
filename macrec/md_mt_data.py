"""mmgen addition: C-style (MQL4GRec) multi-task training data for an MD-RQ-VAE SID, kept as ONE
SID sequence [s; t_1..t_L; v_1..v_L]. Mirrors MQL4GRec's task mix (seqrec + per-view prediction +
item-level translation) without changing the decoded object: valid/test stay `seqrec` (full SID).

Tasks (history is always the full MD SID of each past item, train split items[:-2] as in data.py):
  seqrec    history -> full SID                          (MACRec SeqRecDataset, unchanged)
  mdshared  soft prompt + history -> [s]                 (shared prediction)
  mdtext    soft prompt + history -> [s; t_1..t_L]       (text-view prediction)
  mdimage   soft prompt + history -> [s; v_1..v_L]       (image-view prediction)
  mdtrans   soft prompt + [s; t] -> [v]  and  soft prompt + [s; v] -> [t], one pair per item and
            direction (C's item2image + image2item as one task)
Token groups are read from the index file by prefix: <s*> shared, <t*> text, <v*> image.
"""
import json
import os

from data import BaseDataset, SeqRecDataset

MD_TASKS = ("mdshared", "mdtext", "mdimage", "mdtrans")


def split_sid(tokens):
    s = [x for x in tokens if x.startswith("<s")]
    t = [x for x in tokens if x.startswith("<t")]
    v = [x for x in tokens if x.startswith("<v")]
    assert len(s) + len(t) + len(v) == len(tokens), tokens
    return "".join(s), "".join(t), "".join(v)


class MDViewDataset(BaseDataset):
    """mdshared / mdtext / mdimage: same (history, next item) pairs as seqrec's train split."""

    def __init__(self, args, task):
        super().__init__(args)
        self.task = task
        self.soft_prompt = args.soft_prompts[task]
        self.inters = json.load(open(os.path.join(self.data_path, self.dataset + ".inter.json")))
        self.indices = json.load(open(os.path.join(self.data_path, self.dataset + self.index_file)))
        full = {k: "".join(v) for k, v in self.indices.items()}
        parts = {k: split_sid(v) for k, v in self.indices.items()}
        target = {"mdshared": lambda p: p[0], "mdtext": lambda p: p[0] + p[1],
                  "mdimage": lambda p: p[0] + p[2]}[task]
        self.data = []
        for items in self.inters.values():
            items = items[:-2]
            for i in range(1, len(items)):
                history = items[:i][-self.max_his_len:] if self.max_his_len > 0 else items[:i]
                self.data.append(("".join(full[str(h)] for h in history), target(parts[str(items[i])])))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        inp, out = self.data[index]
        return dict(input_ids=self.soft_prompt + inp, labels=out, task_flag=0)


class MDTransDataset(BaseDataset):
    """mdtrans: text view -> image codes and image view -> text codes, for every item."""

    def __init__(self, args, task="mdtrans"):
        super().__init__(args)
        self.task = task
        self.soft_prompt = args.soft_prompts[task]
        self.indices = json.load(open(os.path.join(self.data_path, self.dataset + self.index_file)))
        self.data = []
        for v in self.indices.values():
            s, t, i = split_sid(v)
            self.data.append((s + t, i))
            self.data.append((s + i, t))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        inp, out = self.data[index]
        return dict(input_ids=self.soft_prompt + inp, labels=out, task_flag=1)


def make_load_datasets(orig_load_datasets):
    """utils.load_datasets with the MD tasks added; every other task goes through MACRec's loader."""
    from torch.utils.data import ConcatDataset

    def load_datasets(args):
        tasks = args.tasks.split(",")
        base = [t for t in tasks if t.lower() not in MD_TASKS]
        saved = args.tasks
        args.tasks = ",".join(base)
        train_data, valid_data = orig_load_datasets(args)
        args.tasks = saved
        by_task = dict(zip(base, train_data.datasets))
        for t in tasks:
            if t.lower() in ("mdshared", "mdtext", "mdimage"):
                by_task[t] = MDViewDataset(args, t.lower())
            elif t.lower() == "mdtrans":
                by_task[t] = MDTransDataset(args)
        return ConcatDataset([by_task[t] for t in tasks]), valid_data

    return load_datasets
