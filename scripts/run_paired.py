"""Provider-neutral paired evaluation adapter (OpenAI-compatible chat endpoint).

Reads the API key from the file given by --key-file or from $BT_API_KEY; the key
is never written to disk.  Calls are resumable: rows already present in the
output JSONL (successful, keyed by item id + config) are skipped.

    python scripts/run_paired.py --items items.json --configs configs.json \
        --config-names fast slow --out raw.jsonl --workers 24
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import threading
import time
from pathlib import Path

import requests

FAST_INSTRUCTION = "Answer immediately. Return exactly one option letter and nothing else."
SLOW_INSTRUCTION = "Construct and verify the complete solution. End with FINAL: <letter>."
VERIFY_INSTRUCTION = SLOW_INSTRUCTION


def build_prompt(item: dict, instruction: str) -> str:
    return item["question"] + "\n\n" + "\n".join(item["options"]) + "\n\n" + instruction


_FINAL = re.compile(r"FINAL\s*[:：]\s*\**\s*\(?([ABCD])\b", re.I)
_LETTER = re.compile(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])")


def extract_answer(text: str) -> str | None:
    if not text:
        return None
    m = _FINAL.findall(text)
    if m:
        return m[-1].upper()
    m = _LETTER.findall(text)
    return m[-1].upper() if m else None


class Client:
    def __init__(self, base_url: str, key: str, timeout: float = 900.0):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.key = key
        self.timeout = timeout

    def chat(self, model: str, prompt: str, params: dict) -> dict:
        body = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False}
        body.update(params)
        t0 = time.time()
        r = requests.post(self.url, headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
                          data=json.dumps(body), timeout=self.timeout)
        lat = time.time() - t0
        j = r.json()
        if "choices" not in j:
            raise RuntimeError(json.dumps(j)[:300])
        msg = j["choices"][0]["message"]
        usage = j.get("usage") or {}
        det = usage.get("completion_tokens_details") or {}
        return {"content": msg.get("content") or "", "finish_reason": j["choices"][0].get("finish_reason"),
                "served_model": j.get("model"), "latency_s": lat,
                "usage": {"prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0),
                          "reasoning_tokens": det.get("reasoning_tokens", 0) or 0, "total_tokens": usage.get("total_tokens", 0)}}


def load_done(out: Path) -> set:
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if not r.get("error"):
                    done.add(r["key"])
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--config-names", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--base-url", default=os.environ.get("BT_API_BASE", "https://api.360.cn/v1"))
    ap.add_argument("--key-file", default=None)
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    key = Path(a.key_file).read_text().strip() if a.key_file else os.environ["BT_API_KEY"]
    client = Client(a.base_url, key)
    items = json.loads(Path(a.items).read_text(encoding="utf-8"))
    if a.limit:
        items = items[: a.limit]
    configs = json.loads(Path(a.configs).read_text(encoding="utf-8"))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out)
    lock = threading.Lock()
    jobs = [(it, name) for name in a.config_names for it in items if f"{it['id']}|{name}" not in done]
    print(f"{len(jobs)} calls to make ({len(done)} already done)", flush=True)

    def work(job):
        it, name = job
        cfg = configs[name]
        prompt = build_prompt(it, cfg["instruction"])
        row = {"key": f"{it['id']}|{name}", "id": it["id"], "split": it.get("split"), "family": it["family"],
               "features": it.get("features"), "config": name, "model": cfg["model"], "gold": it["gold"], "tag": a.tag}
        last = None
        for attempt in range(a.retries):
            try:
                res = client.chat(cfg["model"], prompt, cfg.get("params", {}))
                ans = extract_answer(res["content"])
                row.update({"answer": ans, "correct": ans == it["gold"], "usage": res["usage"], "latency_s": res["latency_s"],
                            "finish_reason": res["finish_reason"], "served_model": res["served_model"], "error": False,
                            "content_tail": res["content"][-200:]})
                break
            except Exception as e:  # noqa: BLE001
                last = str(e)[:300]
                time.sleep(min(60, 3 * (attempt + 1)))
        else:
            row.update({"error": True, "error_message": last})
        with lock:
            with out.open("a", encoding="utf-8") as h:
                h.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    n_ok = n_err = 0
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(work, jobs)):
            n_ok += not r["error"]
            n_err += r["error"]
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(jobs)} ok={n_ok} err={n_err}", flush=True)
    print(f"finished ok={n_ok} err={n_err}", flush=True)


if __name__ == "__main__":
    main()
