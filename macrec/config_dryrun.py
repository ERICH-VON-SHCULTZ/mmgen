"""mmgen addition: resolved-config dry run of finetune_ext.py (docs/PLUGIN_REPORT.md §2).

Runs finetune_ext.main() with the exact training command line, through finetune_contrastive.train() (tokenizer,
soft prompts, datasets, new tokens, model, TrainingArguments, Trainer), and stops at trainer.train(): instead of
training it writes --dump_file with
  resolved_args      vars(args) as finetune_contrastive.train() prints it at start-up (+ soft prompts),
  training_args      the HF TrainingArguments actually passed to the Trainer (to_dict()),
  model_config       the model config after resize_token_embeddings (vocab size),
  datasets           train examples per task, valid examples, number of added tokens, model parameters.
Usage: same arguments as finetune_ext.py plus --dump_file PATH (and a scratch --output_dir: train() saves the
tokenizer + T5 config there).
"""
import json
import sys

import finetune_ext


def install():
    """Take --dump_file out of sys.argv and hook finetune_ext / finetune_contrastive to dump and stop at train()."""
    i = sys.argv.index("--dump_file")
    dump = sys.argv[i + 1]
    del sys.argv[i:i + 2]

    def _dump_and_stop(self, *a, **k):
        out ={"training_args": self.args.to_dict(),
               "model_config": self.model.config.to_dict(),
               "model_class": type(self.model).__name__,
               "n_params": sum(p.numel() for p in self.model.parameters()),
               "train_examples_per_task": finetune_ext._STATS.get("train_examples_per_task"),
               "train_examples_per_epoch": finetune_ext._STATS.get("train_examples_per_epoch"),
               "valid_examples": len(self.eval_dataset) if self.eval_dataset is not None else None,
               "tokenizer_len": len(self.tokenizer) if getattr(self, "tokenizer", None) is not None else None}
        out["resolved_args"] = _ARGS
        json.dump(out, open(dump, "w"), indent=1, default=str)
        print(f"[config_dryrun] wrote {dump}; stopping before training")
        raise SystemExit(0)

    finetune_ext.StatsTrainer.train = _dump_and_stop
    import finetune_contrastive as fc
    _orig = fc.train

    def _capture(args):
        global _ARGS
        _ARGS = None
        _set = fc.load_datasets

        def _load(a):
            global _ARGS
            _ARGS = {k: v for k, v in vars(a).items()}             # soft_prompts are set by now
            return _set(a)
        fc.load_datasets = _load
        return _orig(args)
    fc.train = _capture


def main():
    install()
    finetune_ext.main()


if __name__ == "__main__":
    main()
