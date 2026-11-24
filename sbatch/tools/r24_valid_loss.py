"""mmgen addition (docs/MFD_REPORT.md §24): is the B1 / B2 validation loss comparable with P1a's? CPU. The saved (best, baked)
checkpoint of P1a / B1 / B2 is evaluated on P1a's own validation set (MFDDataset "valid": sequence examples only) with P1a's own
collator (MFDCollator) and loss (plain mean token CE) and compared with the run's recorded best eval_loss.
Every validation example has 5 target tokens, so the example mean does not depend on the batch size."""
import json
import sys

import torch

sys.path.insert(0, "/scratch/yw8866/mmgen/macrec")
import r23_timing as RT  # noqa: E402
from mfd_data import MFDCollator, MFDDataset  # noqa: E402
from mfd_eval import load_mfd_model  # noqa: E402
from transformers import T5Tokenizer  # noqa: E402

out = {}
for D in ("Arts", "Instruments"):
    for V in ("P1a", "B1", "B2"):
        ck = f"{RT.R}/mfd/{D}/" + ("mfd122_mv_geoP1a" if V == "P1a" else f"r24_{V}") + "/seed42"
        tok = T5Tokenizer.from_pretrained(ck)
        args = RT.mk_args(D, RT.I122, RT.I122, ["--max_his_len", "20", "--tasks", "seqrec_mv", "--prompt_num", "4"])
        ds, coll, model = MFDDataset(args, "valid"), MFDCollator(args, tok), load_mfd_model(ck).eval()
        tot, n = 0.0, len(ds)
        with torch.no_grad():
            for s in range(0, n, 128):
                b = coll([ds[i] for i in range(s, min(n, s + 128))])
                k = b["input_ids"].shape[0]
                tot += float(model(**b).loss) * k
        rec = json.load(open(f"{ck}/train_stats.json"))["best_eval_loss"]
        out[f"{D} {V}"] = {"valid_examples": n, "recomputed": tot / n, "recorded_best_eval_loss": rec, "diff": tot / n - rec}
        print(D, V, json.dumps(out[f"{D} {V}"]), flush=True)
json.dump(out, open(f"{RT.R}/geo/r24_valid_loss.json", "w"), indent=1)
