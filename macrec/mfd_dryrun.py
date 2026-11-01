"""mmgen addition: resolved-config dry run of finetune_mfd.py (docs/MFD_REPORT.md §13 sanity check 4).
Applies finetune_mfd's patches (MFDT5, MFD datasets / collator, prediction_loss_only) by importing it, then runs
config_dryrun (finetune_ext.main() up to trainer.train(), dump of resolved args, TrainingArguments, model config,
examples per task). Usage: the exact finetune_mfd.py command line plus --dump_file PATH and a scratch --output_dir.
"""
import config_dryrun
import finetune_mfd  # noqa: F401  (module-level patches)

if __name__ == "__main__":
    config_dryrun.main()
