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
    seen_paths = set()

    for entry in manifest.get("files", []):
        missing_keys = [key for key in ("path", "bytes", "sha256") if key not in entry]
        if missing_keys:
            failures.append(f"manifest entry missing {', '.join(missing_keys)}: {entry!r}")
            continue
        relative_path = entry["path"]
        if not isinstance(relative_path, str) or not relative_path:
            failures.append(f"invalid manifest path: {relative_path!r}")
            continue
        if not isinstance(entry["bytes"], int) or entry["bytes"] < 0:
            failures.append(f"invalid byte count: {relative_path}")
        if not isinstance(entry["sha256"], str) or len(entry["sha256"]) != 64:
            failures.append(f"invalid sha256: {relative_path}")
        if relative_path in seen_paths:
            failures.append(f"duplicate manifest path: {relative_path}")
            continue
        seen_paths.add(relative_path)
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
