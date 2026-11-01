"""mmgen addition: MQL4GRec's (baseline C's) task mix mapped onto MFD's two branches (docs/MFD_REPORT.md §13).

C's tasks (data.py, utils.load_datasets) and the MFD version used here. History views per item: full [f t1 t2 i1 i2],
text [f t1 t2], image [f i1 i2] (mfd_data.view_maps). "Branches" = which decoder branch carries loss; f is always a
target exactly once (text row, position 0: both rows have the same decoder input [start] there, so it is the same
prediction).

  C task        C definition (data.py)                                 MFD task         MFD version
  seqrec        bare text-SID history -> next text SID (no prompt)     mt_seqrec        text-view history -> text branch
  seqimage      bare image-SID history -> next image SID (no prompt)   mt_seqimage      image-view history -> image branch
  fusionseqrec  prompt + text history -> image SID, or (coin flip)     mt_fusionseqrec  prompt + FULL-view history -> both
                prompt + image history -> text SID                                      branches (deviation, see below)
  item2image    prompt + item text SID -> same item's image SID        mt_item2image    prompt + item text view -> image branch
  image2item    prompt + item image SID -> same item's text SID        mt_image2item    prompt + item image view -> text branch

Deviation (decided with the user, 2026-11-01): C's fusionseqrec is cross-modal and never shows both modalities in one
history; here it is full view -> both branches, the MFD stage-1 task, because validation/test use the full view with
the fusionseqrec prompt. Examples per epoch per task are C's: seqrec / seqimage / fusionseqrec = one per training
(history, next item) pair (SeqRecDataset, train_data_mode 0), item2image / image2item = one per catalog item.
Soft prompts: C's own (finetune_contrastive.train: prompt_num T5 vocabulary tokens at ids 100 (i + 1) ... for the
i-th entry of --tasks; c_soft_prompts below rebuilds them for evaluation). Like C, seqrec / seqimage get no prompt.
Labels: an int item id (both branches) or (item id, "text" | "image") for single-branch tasks; MFDCollator masks the
other branch.

Version C (prepared 2026-11-01) adds C's cross-modal sequential task as mt_crossseqrec, mirroring FusionSeqRecDataset:
one example per training (history, next item) pair, and at each __getitem__ a coin flip (random.random() > 0.5, as
in data.py) picks text-view history -> next item's image branch, or image-view history -> next item's text branch;
one soft prompt of its own (the fusionseqrec prompt stays with the full-view inference task).
"""
import random
import json
import os

from mfd_data import MFDDataset, view_maps
from mfd_model import BRANCH_TOKENS

SEQ = {"mt_seqrec": ("text", "text", False), "mt_seqimage": ("image", "image", False),
       "mt_fusionseqrec": ("full", "both", True)}                          # task: (history view, branches, prompt)
ITEM = {"mt_item2image": ("text", "image"), "mt_image2item": ("image", "text")}   # task: (input view, branch)
CROSS = {"mt_crossseqrec": (("text", "image"), ("image", "text"))}               # task: coin flip of (history view, branch)
MT_TASKS = tuple(SEQ) + tuple(ITEM) + tuple(CROSS)
EVAL_TASK = "mt_fusionseqrec"


def is_mt(tasks):
    t = tasks.split(",") if isinstance(tasks, str) else list(tasks)
    if any(x in MT_TASKS for x in t):
        assert all(x in MT_TASKS for x in t) and EVAL_TASK in t, f"MFD multi-task: tasks must be from {MT_TASKS} incl. {EVAL_TASK}"
        return True
    return False


def c_soft_prompts(tasks, tokenizer, prompt_num):
    """finetune_contrastive.train()'s soft prompts: task i -> tokens of ids 100 (i + 1) .. 100 (i + 1) + prompt_num - 1."""
    tasks = tasks.split(",") if isinstance(tasks, str) else list(tasks)
    return {t: "".join(tokenizer.convert_ids_to_tokens(list(range(100 * (i + 1), 100 * (i + 1) + prompt_num))))
            for i, t in enumerate(tasks)}


class MTSeqDataset(MFDDataset):
    """mt_seqrec / mt_seqimage / mt_fusionseqrec: the (history, next item) pairs of SeqRecDataset (any split)."""

    def __init__(self, args, task, mode="train"):
        super().__init__(args, mode)
        self.task = task
        self.view, self.branches, prompt = SEQ[task]
        self.prompt = args.soft_prompts[task] if prompt else ""
        self.vmap = view_maps(self.indices)

    def __getitem__(self, index):
        d = self.inter_data[index]
        item = self.sid2item[d["item"]]
        return dict(input_ids=self.prompt + "".join(self.vmap[self.view][h] for h in d["inters"]),
                    labels=item if self.branches == "both" else (item, self.branches))


class MTCrossDataset(MFDDataset):
    """mt_crossseqrec: C's fusionseqrec (cross-modal) on MFD's branches; the direction is drawn per __getitem__."""

    def __init__(self, args, task, mode="train"):
        super().__init__(args, mode)
        self.task = task
        self.prompt = args.soft_prompts[task]
        self.vmap = view_maps(self.indices)

    def __getitem__(self, index):
        d = self.inter_data[index]
        view, branch = CROSS[self.task][0] if random.random() > 0.5 else CROSS[self.task][1]
        return dict(input_ids=self.prompt + "".join(self.vmap[view][h] for h in d["inters"]),
                    labels=(self.sid2item[d["item"]], branch))


class MTItemDataset:
    """mt_item2image / mt_image2item: one example per catalog item (C's ItemImageDataset), item ids in index order."""

    def __init__(self, args, task):
        self.task = task
        self.view, self.branch = ITEM[task]
        self.prompt = args.soft_prompts[task]
        self.indices = json.load(open(os.path.join(args.data_path, args.dataset, args.dataset + args.index_file)))
        vmap = view_maps(self.indices)
        self.data = [(vmap[self.view]["".join(v)], int(k)) for k, v in self.indices.items()]

    def get_new_tokens(self):
        return sorted({t for v in self.indices.values() for t in v} | set(BRANCH_TOKENS))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, item = self.data[index]
        return dict(input_ids=self.prompt + x, labels=(item, self.branch))


def build(args, task, mode="train"):
    if task in CROSS:
        return MTCrossDataset(args, task, mode)
    return MTSeqDataset(args, task, mode) if task in SEQ else MTItemDataset(args, task)
