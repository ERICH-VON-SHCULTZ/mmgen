"""mmgen addition: run main_single.py (MQL4GRec single-modality RQ-VAE, unchanged) and also save the
final-epoch checkpoint as final_model.pth (MQL4GRec saves only best_collision_model.pth).
Used for the E-norm tokenizer under the final-epoch checkpoint rule (decision 2026-10-27), so
MD-RQ-VAE and E use the same rule. Training itself is untouched: the same seed (2023, hard-coded in
main_single.py), arguments and Trainer; only Trainer.fit is wrapped to save once more after it returns.
Usage: python main_single_final.py <main_single.py arguments>
"""
import os
import runpy
import sys

import trainer

_fit = trainer.Trainer.fit


def fit_and_save_final(self, data):
    out = _fit(self, data)
    self._save_checkpoint(self.epochs - 1, collision_rate=self.best_collision_rate, ckpt_file="final_model.pth")
    return out


trainer.Trainer.fit = fit_and_save_final
sys.argv[0] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main_single.py")
runpy.run_path(sys.argv[0], run_name="__main__")
