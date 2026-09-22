# Benefit Transportability Experiments

This repository contains public research data and offline analysis support for a study of benefit transportability in adaptive LLM routing.

The repository is deliberately limited to reproducibility material. It does not include draft source files, review materials, local paths, credentials, provider account information, raw service payloads, or model response text.

## Contents

- `data/forge/`: generated Forge benchmark items and manifests.
- `data/scale500/`: Scale-500 items used for the IID sanity check.
- `results/primary/`: processed primary Forge results for calibration, locked shift confirmation, target-audit confirmation, and audit-size sensitivity.
- `results/qwen_control/`: processed same-series Qwen control results.
- `results/livebench/`: processed LiveBench external-validation summaries and sanitized scored rows.
- `scripts/reproduce_summary.py`: offline checker that reloads the released results and prints the headline numbers.

## What is intentionally excluded

The public artifact excludes:

- API keys, tokens, endpoint URLs, account identifiers, and local secret files.
- Raw model response text and hidden reasoning traces.
- Draft PDFs, LaTeX files, forms, cover letters, and review materials.
- Temporary third-party checkouts and local build outputs.

## Reproduce the released summaries

The included script requires only Python 3.10+ and the files in this repository:

```bash
python scripts/reproduce_summary.py
```

This checks the presence of the released datasets and prints the main numerical summaries reported in the study.

## Data notes

Forge and Scale-500 are original generated benchmarks. LiveBench is an independently maintained benchmark; this repository includes only the processed scores and task level summaries used for the external validation, not the upstream benchmark codebase.

## License

Code in this repository is released under the MIT License. Original generated benchmark data and processed result tables are released under CC BY 4.0. LiveBench-derived summaries are included for research reproducibility and should be used together with the original LiveBench license and citation requirements.
