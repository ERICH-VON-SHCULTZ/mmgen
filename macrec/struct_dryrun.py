"""mmgen addition: resolved-config dry run of finetune_struct.py (docs/MFD_REPORT.md §16 sanity check 1): config_dryrun's
dump-and-stop hooks (they stop at trainer.train(), after the structure is installed), then finetune_struct.py as __main__.
Usage: the exact finetune_struct.py command line plus --dump_file PATH and a scratch --output_dir.
"""
import os
import runpy

import config_dryrun

if __name__ == "__main__":
    config_dryrun.install()
    runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetune_struct.py"), run_name="__main__")
