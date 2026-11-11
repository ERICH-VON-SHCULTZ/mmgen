"""§21.4: which AC combos beat BOTH C033 and A01 on Arts VALIDATION HR@10 (plain inference, K_f = 50; strict >) ->
mmgen_runs/geo/r21_combo_choice.json and .txt (space-separated VARs to chain to Instruments, possibly empty)."""
import json
R = "/scratch/yw8866/mmgen_runs"
v = lambda run: json.load(open(f"{R}/mfd/Arts/{run}/seed42/r21_eval/eval.json"))["valid"]["plain"]["hr10"]
ref = {"r21_C033": v("r21_C033"), "r21_A01": v("r21_A01")}
res = {ac: {"valid_plain_hr10": v(ac), "beats_both": v(ac) > max(ref.values())} for ac in ("r21_AC1", "r21_AC0")}
win = [k[4:] for k, x in res.items() if x["beats_both"]]
json.dump({"rule": "Arts valid HR@10 (plain, K_f=50) > both C033 and A01", "refs": ref, "combos": res, "chain": win},
          open(f"{R}/geo/r21_combo_choice.json", "w"), indent=1)
open(f"{R}/geo/r21_combo_choice.txt", "w").write(" ".join(win))
print(json.dumps(res), "chain:", win)
