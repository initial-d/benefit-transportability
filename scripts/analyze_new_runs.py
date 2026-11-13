"""Pre-registered analysis of locked Environment F, plus E re-run drift arms.

    python analyze_new_runs.py <release_dir> <experiments_dir> <out_json>
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
src = (Path(__file__).resolve().parent / "revision_analyses.py").read_text().split("# ------------------------------------------------------------------------- main")[0]
exec(src)  # noqa: S102  (shared helpers)

rel, exp, outp = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
rng = np.random.default_rng(20261114)


def load_new(path, lower, upper, items_order=None):
    rows = {}
    for l in Path(path).read_text().splitlines():
        r = json.loads(l)
        if not r.get("error"):
            rows[(r["id"], r["config"])] = r
    ids = items_order or sorted({k[0] for k in rows})
    out = []
    for i in ids:
        if (i, lower) in rows and (i, upper) in rows:
            a, b = rows[(i, lower)], rows[(i, upper)]
            out.append((a.get("split"), a["family"], feat(a), int(a["correct"]), int(b["correct"]),
                        float(a["usage"]["total_tokens"]), float(b["usage"]["total_tokens"]), i))
    env, fam, X, yl, yu, kl, ku, ids = zip(*out)
    return Pair(f"{lower}->{upper}", env, fam, X, yl, yu, kl, ku, ids)


R = {}
# ---------------------------------------------------------------- Environment F
audit_items = json.loads((exp / "data/forge_f/f_audit_items.json").read_text())
test_items = json.loads((exp / "data/forge_f/f_test_items.json").read_text())
order_a = [x["id"] for x in audit_items]
order_t = [x["id"] for x in test_items]
Fa_all = load_new(exp / "results/forge_f/f_query_metrics.jsonl", "fast2", "slow", order_a)
Ft = load_new(exp / "results/forge_f/f_query_metrics.jsonl", "fast2", "slow", order_t)
R["F_counts"] = {"audit_pool": len(Fa_all.fam), "test": len(Ft.fam)}
# primary audit: first 20 per family in file order
first20 = np.zeros(len(Fa_all.fam), bool)
for f in FAM:
    first20[np.where(Fa_all.fam == f)[0][:20]] = True
Fa = Fa_all.sub(first20)

main_pair = load_main(rel)
hist = main_pair.envs("A", "B", "C", "D", "E_audit", "E_test")
dk_a = np.maximum(fam_dk(Fa), 0)
primary_fast = one_fast(fam_tau(Fa), dk_a)
res = family_policy_suite(hist, Fa, Ft, rng, hist_envs=["A", "B", "C", "D", "E_test"], with_items=True)
R["F_suite"] = res
R["F_primary_fast"] = [f for f, m in zip(FAM, primary_fast) if m]
R["F_audit_tau"] = dict(zip(FAM, map(float, fam_tau(Fa))))
R["F_audit_dk"] = dict(zip(FAM, map(float, dk_a)))
R["F_test_tau"] = dict(zip(FAM, map(float, fam_tau(Ft))))
R["F_hist_tau"] = dict(zip(FAM, map(float, fam_tau(hist))))

# secondary: two-stage adaptive audit (pre-registered)
def ci90(b):
    n = len(b)
    sd = max(b.std(ddof=1) if n > 1 else 1.0, 1 / math.sqrt(n))
    return b.mean() - 1.645 * sd / math.sqrt(n), b.mean() + 1.645 * sd / math.sqrt(n)

sel = first20.copy()
extended = []
for f in FAM:
    ii = np.where(Fa_all.fam == f)[0]
    lo, hi = ci90(Fa_all.b[ii[:20]])
    if lo <= GAMMA <= hi:
        sel[ii[20:40]] = True
        extended.append(f)
Fa2 = Fa_all.sub(sel)
ad_fast = one_fast(fam_tau(Fa2), np.maximum(fam_dk(Fa2), 0))
r2 = evaluate_family(Ft, ad_fast)
r2["audit_calls"] = int(sel.sum())
r2["extended_families"] = extended
r2["fast"] = [f for f, m in zip(FAM, ad_fast) if m]
R["F_adaptive"] = r2

# uncertainty for primary
fm = primary_fast
is_fast = np.array([dict(zip(FAM, fm))[f] for f in Ft.fam])
y = np.where(is_fast, Ft.yl, Ft.yu)
k = np.where(is_fast, Ft.kl, Ft.ku)
n = len(y)
bi = rng.integers(0, n, size=(10000, n))
rnd = 0.8 * Ft.yu[bi].mean(1) + 0.2 * Ft.yl[bi].mean(1)
R["F_primary_ci"] = {"acc": float(y.mean()), "acc_cp": clopper_pearson(int(y.sum()), n),
                     "gain_vs_random": float(y.mean() - (0.8 * Ft.yu.mean() + 0.2 * Ft.yl.mean())),
                     "gain_ci": [pct(y[bi].mean(1) - rnd, 2.5), pct(y[bi].mean(1) - rnd, 97.5)],
                     "save": float(1 - k.sum() / Ft.ku.sum()),
                     "save_ci": [pct(1 - k[bi].sum(1) / Ft.ku[bi].sum(1), 2.5), pct(1 - k[bi].sum(1) / Ft.ku[bi].sum(1), 97.5)],
                     "preregistered_test_pass": bool(y.mean() - (0.8 * Ft.yu.mean() + 0.2 * Ft.yl.mean()) > 0.015)}
# audit bootstrap for F (within-family resample of the 20-per-family audit)
picks = {}
accs = []
saves = []
a_by = {f: np.where(Fa.fam == f)[0] for f in FAM}
for _ in range(5000):
    idx = np.concatenate([a_by[f][rng.integers(0, len(a_by[f]), size=len(a_by[f]))] for f in FAM])
    a = Pair(Fa.name, Fa.env[idx], Fa.fam[idx], Fa.X[idx], Fa.yl[idx], Fa.yu[idx], Fa.kl[idx], Fa.ku[idx], Fa.ids[idx])
    m = one_fast(fam_tau(a), np.maximum(fam_dk(a), 0))
    key = "+".join(f for f, mm in zip(FAM, m) if mm) or "none"
    picks[key] = picks.get(key, 0) + 1
    e = evaluate_family(Ft, m)
    accs.append(e["acc"]); saves.append(e["save_vs_slow"])
R["F_audit_bootstrap"] = {"pick": {k2: v / 5000 for k2, v in sorted(picks.items(), key=lambda t: -t[1])},
                          "acc_mean": float(np.mean(accs)), "acc_p05": pct(accs, 5), "save_mean": float(np.mean(saves))}
# shift statistics F vs A-E
R["F_cells"] = [{"family": f, "n_test": int((Ft.fam == f).sum()), "tau_test": float(Ft.b[Ft.fam == f].mean()),
                 "tau_audit20": float(Fa.b[Fa.fam == f].mean()), "acc_fast": float(Ft.yl[Ft.fam == f].mean()),
                 "acc_slow": float(Ft.yu[Ft.fam == f].mean())} for f in FAM]
R["F_uniform"] = {"fast": float(Ft.yl.mean()), "slow": float(Ft.yu.mean()), "fast_tokens": float(Ft.kl.mean()), "slow_tokens": float(Ft.ku.mean())}

# ---------------------------------------------------------------- E drift arms
e_items = json.loads((rel / "data/forge/e_items.json").read_text())
split_of = {x["id"]: ("audit" if ".calibration." in x["id"] else "test") for x in e_items}
# reference route for the new pair: Algorithm 1 fitted on the t=0 re-run audit
_ref = load_new(exp / "results/drift_e/E_rerun_t0.jsonl", "fast2", "slow")
_msk = np.array([split_of[i] == "audit" for i in _ref.ids])
orig_audit = _ref.sub(_msk)
orig_route = one_fast(fam_tau(orig_audit), np.maximum(fam_dk(orig_audit), 0))
arms = {}
for name, path, lo, up in [("rerun_t0", "results/drift_e/E_rerun_t0.jsonl", "fast2", "slow"), ("temperature_0.7", "results/drift_e/E_rerun_t07.jsonl", "fast2_t07", "slow_t07"),
                           ("prompt_template", "results/drift_e/E_rerun_tpl2.jsonl", "fast2_tpl2", "slow_tpl2")]:
    p = exp / path
    if not p.exists():
        continue
    P = load_new(p, lo, up)
    msk_a = np.array([split_of[i] == "audit" for i in P.ids])
    A, T = P.sub(msk_a), P.sub(~msk_a)
    fr = evaluate_family(T, orig_route)
    re_route = one_fast(fam_tau(A), np.maximum(fam_dk(A), 0))
    rr = evaluate_family(T, re_route)
    arms[name] = {"n_test": len(T.fam), "tau_test": dict(zip(FAM, map(float, fam_tau(T)))),
                  "uniform_fast": float(T.yl.mean()), "uniform_slow": float(T.yu.mean()),
                  "frozen_route": "+".join(f for f, m in zip(FAM, orig_route) if m), "frozen": fr,
                  "reaudit_route": "+".join(f for f, m in zip(FAM, re_route) if m), "reaudit": rr,
                  "random80": random_matched(T, 0.8)}
R["E_drift_arms"] = arms
outp.write_text(json.dumps(R, indent=1, default=float))
print("wrote", outp)
