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

Equivalent Make target:

```bash
make summary
```

To verify the file sizes and SHA-256 digests recorded in
`RELEASE_MANIFEST.json`, run:

```bash
make check-manifest
```

The manifest check validates the released files listed in the manifest. It does
not require network access and does not inspect excluded private materials.


## Revision R1 additions

- `scripts/demo_target_audit.py` - standard-library demo: rebuilds the Environment-E target-audited router from the 100 audit items and evaluates it once on the locked test (99.40% accuracy, 27.02% saving, 3 false-fast).
- `scripts/revision_analyses.py` - all re-analyses of released outcomes (shift tests, intervals, baselines, replications). `python scripts/revision_analyses.py . out/`
- `preregistration/F_preregistration.json` - plan for Environment F, committed before any F model call (amendment 1 records the fast-configuration choice made on E-audit data only).
- `data/forge_f/` - Environment F items, audit pool, and locked test; `scripts/forge_generate.py` generates them (every item passes two distinct exact solvers).
- `results/forge_f/`, `results/drift_e/` - sanitized paired outcomes for Environment F and the temperature / prompt-template drift runs.
- `scripts/analyze_new_runs.py`, `scripts/extra_new_runs.py` - pre-registered F analysis and drift analysis. `python scripts/analyze_new_runs.py . . out/new_runs.json`
- `scripts/run_paired.py` - provider-neutral, resumable paired-call adapter for OpenAI-compatible endpoints; the API key is read from `--key-file` or `$BT_API_KEY` and is never written to disk. `scripts/configs.json` lists the request settings.

Analyses need Python 3 with NumPy (`pip install numpy matplotlib`); the demo needs only the standard library.

## Data notes

Forge and Scale-500 are original generated benchmarks. LiveBench is an independently maintained benchmark; this repository includes only the processed scores and task level summaries used for the external validation, not the upstream benchmark codebase.

## License

Code in this repository is released under the MIT License. Original generated benchmark data and processed result tables are released under CC BY 4.0. LiveBench-derived summaries are included for research reproducibility and should be used together with the original LiveBench license and citation requirements.
