"""mmgen addition: run MACRec's finetune_contrastive.train() unchanged, with two hooks.

  --model_class contrastive  MACRec's CrossModalContrastive (implicit alignment, weight 0.01, temp 0.1)
  --model_class plain        plain T5ForConditionalGeneration, as in MQL4GRec's finetune.py and TIGER.
                             (PlainT5 only drops the `task_flag` field that MACRec's collator adds.)
  --stats_file PATH          after training, write wall time, peak GPU memory, the selected
                             checkpoint, its epoch and validation loss.
Training data, loss, optimizer, schedule, early stopping and model selection are MACRec's own.
"""
import argparse
import json
import time

import torch
import transformers
from transformers import T5ForConditionalGeneration

import finetune_contrastive as fc
from utils import parse_global_args, parse_train_args, parse_dataset_args


class PlainT5(T5ForConditionalGeneration):
    def forward(self, *args, task_flag=None, **kwargs):
        return super().forward(*args, **kwargs)


_STATS = {}
_BaseTrainer = transformers.Trainer


class StatsTrainer(_BaseTrainer):
    def train(self, *args, **kwargs):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        out = super().train(*args, **kwargs)
        st = self.state
        evals = [h for h in st.log_history if "eval_loss" in h]
        best = min(evals, key=lambda h: h["eval_loss"]) if evals else {}
        _STATS.update({
            "train_wall_seconds": time.time() - t0,
            "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
            "gpu": torch.cuda.get_device_name(0),
            "epochs_run": st.epoch,
            "global_steps": st.global_step,
            "best_model_checkpoint": st.best_model_checkpoint,
            "best_eval_loss": best.get("eval_loss"),
            "best_epoch": best.get("epoch"),
            "eval_history": [(h["epoch"], h["eval_loss"]) for h in evals],
        })
        return out


def main():
    parser = argparse.ArgumentParser(description="LLMRec (mmgen ext)")
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)
    parser.add_argument("--model_class", choices=["contrastive", "plain"], default="contrastive")
    parser.add_argument("--stats_file", default=None)
    args = parser.parse_args()

    if args.model_class == "plain":
        fc.create_contrastive_model = lambda config: PlainT5(config)
    transformers.Trainer = StatsTrainer          # finetune_contrastive calls transformers.Trainer(...)
    fc.train(args)

    if args.stats_file and int(__import__("os").environ.get("LOCAL_RANK", 0)) == 0:
        _STATS["model_class"] = args.model_class
        with open(args.stats_file, "w") as f:
            json.dump(_STATS, f, indent=1)
        print("[finetune_ext]", json.dumps({k: v for k, v in _STATS.items() if k != "eval_history"}))


if __name__ == "__main__":
    main()
