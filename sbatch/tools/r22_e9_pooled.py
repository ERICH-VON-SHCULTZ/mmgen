"""mmgen addition (docs/MFD_REPORT.md §22.9): E9k4 seed 42 POOLED over Arts + Instruments + Games (the round-22 selection rule,
§22.10.4) vs P1a, A01 and C, on validation and test: pooled HR@10 / NDCG@10 and stratified paired bootstrap CIs (resampled
within each dataset; r22_e10_report.strat_boot). Out: mmgen_runs/geo/r22_e9k4_pooled.json."""
import json

import numpy as np

import r22_e10_report as E

out = {}
for sp in ("valid", "test"):
    r = E.pooled_valid({D: "E9k4" for D in E.DS}, 42, sp)
    rk = {D: E.ev(D, "E9k4", 42)[1][sp] for D in E.DS}
    rc = {D: E.C_ranks(D, 42, sp) for D in E.DS}
    tot = sum(len(v) for v in rk.values())
    r["C"] = {"hr10": float(sum(E.hr(v).sum() for v in rc.values()) / tot), "ndcg10": float(sum(E.ndcg(v).sum() for v in rc.values()) / tot)}
    rng = np.random.default_rng(0)
    r["vs_C"] = {m: E.strat_boot({D: f(rk[D]) - f(rc[D]) for D in E.DS}, rng) for m, f in (("hr10", E.hr), ("ndcg10", E.ndcg))}
    out[sp] = r
    print(sp, json.dumps(r), flush=True)
json.dump(out, open(f"{E.R}/geo/r22_e9k4_pooled.json", "w"), indent=1)
