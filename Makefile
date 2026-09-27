.PHONY: summary check-manifest

PYTHON ?= python3

summary:
	$(PYTHON) scripts/reproduce_summary.py

check-manifest:
	$(PYTHON) scripts/check_release_manifest.py
