"""mmgen addition: train MFD stage 1 (mfd_model.MFDT5) with EXACTLY the chain-decoder control's path:
finetune_ext.main() -> MACRec finetune_contrastive.train() -> HF Trainer, same arguments
(T5 config/ckpt, AdamW, batch 256 x grad-accum 4, LR 5e-4, cosine, warmup 0.01, weight decay 0.01,
200 epochs, early stopping patience 10 on validation loss, best checkpoint by validation loss).
Swapped in: the model (finetune_ext.PlainT5 -> MFDT5), the datasets (seqrec examples, target as item id)
and the collator (two decoder sequences per example).
One evaluation-only setting: prediction_loss_only=True, so evaluation keeps the loss and does not
accumulate (B, 2, 4, V) logits; training and model selection are unaffected.
After training, the stats file also gets decoder_sequences_per_epoch (= 2 x examples).
--tasks seqrec_mv: multi-view history augmentation (mfd_data.MVMFDDataset, 3 views per example).
Works for any v1 MD SID width (1+2+2, 1+3+3): the decoder sequences have length L + 2.
Run with the same arguments as finetune_ext.py plus --model_class plain.
"""
import json
import os
import sys

import transformers

import finetune_contrastive as fc
import finetune_ext
from mfd_data import MFDCollator, load_mfd_datasets
from mfd_model import MFDT5

finetune_ext.PlainT5 = MFDT5            # finetune_ext's --model_class plain builds PlainT5(config)
fc.load_datasets = load_mfd_datasets    # finetune_ext wraps fc.load_datasets (examples per task)
finetune_ext.mv_aug_data.MV_TASKS = ()  # --tasks seqrec_mv is handled by load_mfd_datasets (MVMFDDataset), not the chain wrapper
fc.Collator = MFDCollator

_TA = transformers.TrainingArguments


def _training_args(*a, **k):
    k.setdefault("prediction_loss_only", True)
    return _TA(*a, **k)


transformers.TrainingArguments = _training_args

if __name__ == "__main__":
    assert "--model_class" in sys.argv and sys.argv[sys.argv.index("--model_class") + 1] == "plain"
    finetune_ext.main()
    stats = sys.argv[sys.argv.index("--stats_file") + 1] if "--stats_file" in sys.argv else None
    if stats and int(os.environ.get("LOCAL_RANK", 0)) == 0:
        s = json.load(open(stats))
        s["decoder_sequences_per_epoch"] = 2 * s["train_examples_per_epoch"]
        s["entry"] = "finetune_mfd.py"
        json.dump(s, open(stats, "w"), indent=1)
