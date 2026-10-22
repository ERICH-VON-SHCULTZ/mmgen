"""mmgen addition: run cross_index/main.py unchanged, but actually apply the contrast weights.

main.py parses --text_contrast_weight / --image_contrast_weight / --recon_contrast_weight (l.55-57)
but never passes them to CrossRQVAE (l.107-124), so CrossRQVAE's defaults 1.0 / 1.0 / 0.001 are
always used. This wrapper injects the parsed values as the constructor defaults and then executes
main.py as __main__. Nothing else changes. Used only for the "paper-setting" MACRec variant.
"""
import argparse
import runpy
import sys

pre = argparse.ArgumentParser(add_help=False)
pre.add_argument("--text_contrast_weight", type=float, default=1.0)
pre.add_argument("--image_contrast_weight", type=float, default=1.0)
pre.add_argument("--recon_contrast_weight", type=float, default=0.001)
w, _ = pre.parse_known_args()

from models.rqvae import CrossRQVAE  # noqa: E402

_orig_init = CrossRQVAE.__init__


def _init(self, *args, **kwargs):
    kwargs.setdefault("text_contrast_weight", w.text_contrast_weight)
    kwargs.setdefault("image_contrast_weight", w.image_contrast_weight)
    kwargs.setdefault("recon_contrast_weight", w.recon_contrast_weight)
    _orig_init(self, *args, **kwargs)
    print(f"[main_cross_weights] contrast weights text={self.text_contrast_weight} "
          f"image={self.image_contrast_weight} recon={self.recon_contrast_weight}", flush=True)


CrossRQVAE.__init__ = _init
sys.argv[0] = "main.py"
runpy.run_path("main.py", run_name="__main__")
