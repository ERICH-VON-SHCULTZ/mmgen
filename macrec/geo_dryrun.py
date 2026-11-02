"""mmgen addition: resolved-config dry run of finetune_geo.py (docs/MFD_REPORT.md §14 sanity check 3).
Installs config_dryrun's dump-and-stop hooks (they stop at trainer.train(), i.e. after the geometry is installed),
then executes finetune_geo.py as __main__ exactly as a real run does.
Usage: the exact finetune_geo.py command line plus --dump_file PATH and a scratch --output_dir.
"""
import os
import runpy

import config_dryrun

if __name__ == "__main__":
    config_dryrun.install()
    runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetune_geo.py"), run_name="__main__")
