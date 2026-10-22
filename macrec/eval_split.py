"""mmgen addition: run MACRec's test_ddp_save.test_ddp() unchanged on the validation OR test split,
and record inference cost.

  --split test   identical to test_ddp_save.py (target = last item).
  --split valid  same code path, but the dataset is SeqRecDataset(mode="valid") (target = 2nd-to-last
                 item, history = items[:-2]), i.e. the validation split used for model selection.
  --stats_file   writes beam-search cost: wall seconds inside generate() (CUDA-synchronised), number of
                 users, decoder steps per generate call (= generated length incl. EOS), beams, GPU.
                 HF beam search runs one batched decoder forward per step over all beams, so
                 decoder forward passes per user = steps; beam-hypothesis evaluations = steps x beams.
Launch with torchrun (test_ddp_save initialises torch.distributed).
"""
import argparse
import json
import os
import time

import torch
from transformers import T5ForConditionalGeneration

import test_ddp_save as T
from data import SeqRecDataset
from utils import parse_global_args, parse_dataset_args, parse_test_args

_COST = {"generate_seconds": 0.0, "calls": 0, "users": 0, "steps": [], "beams": None}
_orig_generate = T5ForConditionalGeneration.generate


def _timed_generate(self, *args, **kwargs):
    torch.cuda.synchronize()
    t0 = time.time()
    out = _orig_generate(self, *args, **kwargs)
    torch.cuda.synchronize()
    _COST["generate_seconds"] += time.time() - t0
    _COST["calls"] += 1
    _COST["users"] += int(kwargs["input_ids"].shape[0])
    _COST["steps"].append(int(out["sequences"].shape[1]) - 1)   # minus decoder_start token
    _COST["beams"] = int(kwargs.get("num_beams", 1))
    return out


def main():
    parser = argparse.ArgumentParser(description="LLMRec_test (mmgen ext)")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--stats_file", default=None)
    args = parser.parse_args()

    if args.split == "valid":
        T.load_test_dataset = lambda a: SeqRecDataset(a, task=a.test_task.lower(), mode="valid")
    T5ForConditionalGeneration.generate = _timed_generate
    t0 = time.time()
    T.test_ddp(args)
    total = time.time() - t0

    if args.stats_file and int(os.environ.get("LOCAL_RANK", 0)) == 0:
        steps = _COST.pop("steps")
        _COST.update({
            "split": args.split, "task": args.test_task, "total_script_seconds": total,
            "mean_decoder_steps_per_call": sum(steps) / max(len(steps), 1),
            "max_decoder_steps": max(steps) if steps else None,
            "decoder_forward_passes_per_user": sum(steps) / max(len(steps), 1),
            "beam_hypothesis_evals_per_user": (sum(steps) / max(len(steps), 1)) * (_COST["beams"] or 1),
            "ms_per_user": 1000 * _COST["generate_seconds"] / max(_COST["users"], 1),
            "gpu": torch.cuda.get_device_name(0),
        })
        with open(args.stats_file, "w") as f:
            json.dump(_COST, f, indent=1)
        print("[eval_split]", json.dumps(_COST))


if __name__ == "__main__":
    main()
