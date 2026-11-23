"""mmgen addition (docs/MFD_REPORT.md §24.2, Track B): item-level semantic examples for P1a.

Per catalog item and epoch, two extra training examples, shuffled in with P1a's sequence examples (MVMFDDataset):
  B-text   encoder input = ONE item position E(t1) + E(t2) + recency embedding 0  -- no E(f): f must be inferred
  B-image  encoder input = ONE item position E(i1) + E(i2) + recency embedding 0
  target   the item's full SID (MFD rows: f, text branch, image branch), the same loss as the sequence examples.
Loss weight w (B1: 1.0, B2: 0.3): the per-token CE of the item examples is multiplied by w and the batch loss is the weighted
mean sum_tokens(w * CE) / sum_tokens(w). With w = 1 for every example this is exactly P1a's loss (mean token CE).

Pieces (installed by finetune_r24.py --r24_item W; P1a's path is unchanged for every row that contains an f token):
- R24StructInput: P1a's item-level structure, plus one rule: in a row WITHOUT any f token (<s0_*>), the first code token
  starts the (single) item. Every P1a row (sequence examples, validation, test) contains f tokens, so its structure is
  identical (checked in r24_check.py).
- ItemSemDataset: the 2 x N item examples; input string = the item's text (or image) code tokens only; labels = (item, "item",
  modality, w).
- R24Collator: MFDCollator for both kinds; item rows get both branches as targets and an `example_weight` tensor (1.0 for the
  sequence rows) passed to forward (the HF Trainer keeps batch keys that are forward() arguments).
- R24ItemMFDT5: ItemMFDT5 + the weighted loss. After training finetune_geo's bake restores ItemMFDT5 (mro[2]); evaluation
  is P1a's unchanged.
"""
import json
import os

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import Seq2SeqLMOutput

import enc_struct
from enc_struct import OTHER, ItemMFDT5, StructInput
from mfd_data import MFDCollator, sid_parts


class R24StructInput(StructInput):
    def structure(self, ids):
        s = self.slot_of[ids]
        code = s != OTHER
        has_f = (s == 0).any(1, keepdim=True)
        first_code = code & (torch.cumsum(code.long(), 1) == 1)
        start = (s == 0) | (first_code & ~has_f)                       # f-less rows: first code token starts the item
        item = torch.cumsum(start.long(), 1) - 1
        n = start.sum(1, keepdim=True)
        rec = (n - 1 - item).clamp(0, enc_struct.MAX_ITEMS - 1)
        return s, code & (item >= 0), item, rec, n.squeeze(1)


class ItemSemDataset:
    """One B-text and one B-image example per catalog item (index order; the Trainer's sampler shuffles them in)."""
    task = "r24_item"                                                    # finetune_ext counts examples per dataset .task

    def __init__(self, args, weight):
        self.w = float(weight)
        self.indices = json.load(open(os.path.join(args.data_path, args.dataset, args.dataset + args.index_file)))
        self.data = []
        for k, toks in self.indices.items():
            f, t, v = sid_parts(toks)
            assert f and t and v and f not in "".join(t) + "".join(v)
            self.data.append(("".join(t), int(k), "text"))
            self.data.append(("".join(v), int(k), "image"))

    def get_new_tokens(self):
        return []                                                        # every code token is already added by P1a's dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x, item, mod = self.data[i]
        return dict(input_ids=x, labels=(item, "item", mod, self.w))


class R24Collator(MFDCollator):
    def __call__(self, batch):
        w = torch.ones(len(batch))
        seq = []
        for n, d in enumerate(batch):
            lb = d["labels"]
            if isinstance(lb, (tuple, list)) and len(lb) == 4 and lb[1] == "item":
                w[n] = lb[3]
                d = dict(d, labels=lb[0])                                 # both branches + f trained (full SID)
            seq.append(d)
        out = super().__call__(seq)
        out["example_weight"] = w
        return out


class R24ItemMFDT5(ItemMFDT5):
    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, labels=None,
                encoder_outputs=None, example_weight=None, **kwargs):
        if example_weight is None or labels is None:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                                   labels=labels, encoder_outputs=encoder_outputs, **kwargs)
        out = super().forward(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                              labels=None, encoder_outputs=encoder_outputs, **kwargs)
        lg = out.logits                                                   # (B, 2, L, V)
        ce = F.cross_entropy(lg.reshape(-1, lg.size(-1)).float(), labels.reshape(-1), ignore_index=-100, reduction="none")
        valid = (labels.reshape(-1) != -100).float()
        w = example_weight.to(ce.device)[:, None, None].expand_as(labels).reshape(-1).float() * valid
        # a new output object: ModelOutput only exposes keys set at construction (the Trainer reads outputs["loss"])
        return Seq2SeqLMOutput(loss=(w * ce).sum() / w.sum(), logits=lg)


def install():
    """Patch the module-level names that finetune_geo / finetune_mfd read at run time."""
    import mfd_data
    enc_struct.StructInput = R24StructInput                               # install_struct builds StructInput(...) by name
    enc_struct.ItemMFDT5 = R24ItemMFDT5                                   # finetune_geo.setup() reads it at call time
    mfd_data.MFDCollator = R24Collator                                    # finetune_mfd.py imports it at run time
