from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: str):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def main() -> None:
    primary_d = load_json("results/primary/locked_d_confirmation.json")
    primary_e = load_json("results/primary/target_audit_e_confirmation.json")
    qwen = load_json("results/qwen_control/same_series_qwen_summary.json")
    livebench = load_json("results/livebench/livebench_qwen35b_397b_analysis.json")

    print("Primary locked D:")
    print("  historical probe accuracy:", pct(primary_d["results"]["htr_cdr"]["accuracy"]))
    print("  matched random mean:", pct(primary_d["results"]["random_slow_rate_matched"]["accuracy_mean_mc"]))

    print("Target-audited E:")
    print("  audited policy accuracy:", pct(primary_e["results"]["a_cdr"]["accuracy"]))
    print("  token saving vs uniform slow:", pct(primary_e["relative_to_uniform_slow"]["token_savings"]))

    print("Same-series Qwen control:")
    print("  complete Forge items:", qwen["complete_items"])
    print("  tiers:", ", ".join(qwen["tiers"]))

    print("LiveBench external validation:")
    print("  paired items:", livebench["overall"]["paired_n"])
    print("  mean benefit:", pct(livebench["overall"]["mean_benefit_upper_minus_lower"]))


if __name__ == "__main__":
    main()
