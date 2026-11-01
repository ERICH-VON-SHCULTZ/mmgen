"""mmgen addition: resolved-config dry run of finetune_mfd.py (docs/MFD_REPORT.md §13 sanity check 4).
Installs config_dryrun's dump-and-stop hooks, then executes finetune_mfd.py itself as __main__, so its module-level
patches (MFDT5, MFD datasets / collator, prediction_loss_only) and finetune_ext.main() run in exactly the order of a
real training run (importing finetune_mfd from here instead changes that order and loses a TrainingArguments patch).
Usage: the exact finetune_mfd.py command line plus --dump_file PATH and a scratch --output_dir.
"""
import os
import runpy

import config_dryrun

if __name__ == "__main__":
    config_dryrun.install()
    runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetune_mfd.py"), run_name="__main__")
