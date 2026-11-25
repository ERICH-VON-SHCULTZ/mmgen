"""mmgen addition (§24.4): finetune_ext.py's exact baseline training command, stopped after --smoke_steps N real optimizer steps
(gate): <output_dir>/smoke.json with losses, learning rate, batch size; exit 0 only if every loss is finite."""
import json
import math
import os
import sys

i = sys.argv.index("--smoke_steps")
N = int(sys.argv[i + 1])
del sys.argv[i:i + 2]

import finetune_ext  # noqa: E402
from transformers import TrainerCallback  # noqa: E402

LOSSES = []
_Base = finetune_ext.StatsTrainer


class Stop(TrainerCallback):
    def on_step_end(self, args, state, control, **kw):
        if state.global_step < N:
            return
        res = {"steps": state.global_step, "losses": LOSSES, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
               "per_device_train_batch_size": args.per_device_train_batch_size, "max_grad_norm": args.max_grad_norm,
               "gradient_accumulation_steps": args.gradient_accumulation_steps, "max_steps": state.max_steps}
        res["ok"] = len(LOSSES) >= N and all(math.isfinite(x) for x in LOSSES)
        json.dump(res, open(os.path.join(args.output_dir, "smoke.json"), "w"), indent=1)
        print("[baseline_smoke]", json.dumps(res), flush=True)
        os._exit(0 if res["ok"] else 4)


class SmokeTrainer(_Base):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.add_callback(Stop())

    def training_step(self, *a, **k):
        loss = super().training_step(*a, **k)
        LOSSES.append(float(loss.detach().float()))
        return loss


finetune_ext.StatsTrainer = SmokeTrainer                                  # finetune_ext.main() reads the name at run time
if __name__ == "__main__":
    finetune_ext.main()
