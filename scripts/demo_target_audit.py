"""Minimal demo: rebuild the Environment-E target-audited router from the released
paired audit and evaluate it once on the locked E test.

    python scripts/demo_target_audit.py            (run from the repository root)

Requires only the Python 3 standard library.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results" / "primary"
GAMMA, SLOW_FRACTION = 0.15, 0.8


def paired(path: Path) -> list[dict]:
    units: dict[str, dict] = defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if not row.get("error"):
            units[row["id"]][row["mode"]] = row
    out = []
    for unit in units.values():
        if {"fast", "slow"} <= unit.keys():
            f, s = unit["fast"], unit["slow"]
            out.append({"family": f["family"], "yf": int(f["correct"]), "ys": int(s["correct"]),
                        "kf": f["usage"]["total_tokens"], "ks": s["usage"]["total_tokens"]})
    return out


def fit(audit: list[dict]) -> set[str]:
    """Algorithm 1 of the revised manuscript (target-audited router)."""
    by = defaultdict(list)
    for r in audit:
        by[r["family"]].append(r)
    tau = {g: sum(r["ys"] - r["yf"] for r in rs) / len(rs) for g, rs in by.items()}
    dk = {g: sum(r["ks"] - r["kf"] for r in rs) / len(rs) for g, rs in by.items()}
    share = {g: len(rs) / len(audit) for g, rs in by.items()}
    fast, slow_share = set(), 1.0
    for g in sorted((g for g in tau if tau[g] <= GAMMA + 1e-12), key=lambda g: (-dk[g], g)):
        if slow_share - share[g] >= SLOW_FRACTION - 1e-12:
            fast.add(g)
            slow_share -= share[g]
    for g in sorted(tau):
        print(f"  {g:13s} audit benefit {100 * tau[g]:5.1f} pp   mean extra slow tokens {dk[g]:8.0f}")
    return fast


def evaluate(test: list[dict], fast: set[str]) -> dict:
    y = [r["yf"] if r["family"] in fast else r["ys"] for r in test]
    k = [r["kf"] if r["family"] in fast else r["ks"] for r in test]
    ff = sum(1 for r in test if r["family"] in fast and r["ys"] > r["yf"])
    return {"accuracy": sum(y) / len(y), "save_vs_slow": 1 - sum(k) / sum(r["ks"] for r in test), "false_fast": ff}


def main() -> None:
    audit = paired(ROOT / "target_audit_e_calibration_metrics.jsonl")
    test = paired(ROOT / "target_audit_e_test_metrics.jsonl")
    print(f"E audit: {len(audit)} paired items; E locked test: {len(test)} paired items")
    fast = fit(audit)
    print("fast families:", sorted(fast) or "none")
    res = evaluate(test, fast)
    print(f"locked test: accuracy {100 * res['accuracy']:.2f}%  saving vs uniform slow "
          f"{100 * res['save_vs_slow']:.2f}%  false-fast {res['false_fast']}")
    rng = random.Random(0)
    slow_n = round(SLOW_FRACTION * len(test))
    accs = []
    for _ in range(2000):
        idx = set(rng.sample(range(len(test)), slow_n))
        accs.append(sum(r["ys"] if i in idx else r["yf"] for i, r in enumerate(test)) / len(test))
    print(f"random routing at {int(100 * SLOW_FRACTION)}% slow: mean accuracy {100 * sum(accs) / len(accs):.2f}%")


if __name__ == "__main__":
    main()
