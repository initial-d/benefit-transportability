from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "RELEASE_MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures = []

    for entry in manifest.get("files", []):
        relative_path = entry["path"]
        path = ROOT / relative_path
        if not path.is_file():
            failures.append(f"missing: {relative_path}")
            continue
        actual_size = path.stat().st_size
        if actual_size != entry["bytes"]:
            failures.append(f"size mismatch: {relative_path} expected {entry['bytes']} got {actual_size}")
        actual_sha = sha256(path)
        if actual_sha != entry["sha256"]:
            failures.append(f"sha256 mismatch: {relative_path}")

    if failures:
        for failure in failures:
            print(failure)
        raise SystemExit(1)

    print(f"release manifest check passed ({len(manifest.get('files', []))} files)")


if __name__ == "__main__":
    main()
