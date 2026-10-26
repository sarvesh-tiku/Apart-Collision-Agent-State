"""Minimal OpenRouter chat client with a disk cache and a hard spend cap."""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

CACHE = Path(__file__).resolve().parent.parent / "results" / "llm_cache.jsonl"
PRICES = {  # USD per token (prompt, completion), from the OpenRouter catalog 2026-10-26
    "openai/gpt-4o-mini": (0.15e-6, 0.6e-6),
    "openai/gpt-4.1-mini": (0.4e-6, 1.6e-6),
    "meta-llama/llama-3.3-70b-instruct": (0.1e-6, 0.32e-6),
    "google/gemini-2.5-flash-lite": (0.1e-6, 0.4e-6),
    "anthropic/claude-haiku-4.5": (1e-6, 5e-6),
    "google/gemini-2.5-flash": (0.3e-6, 2.5e-6),
}
SPEND_CAP_USD = float(os.environ.get("SPEND_CAP_USD", "15"))

_lock = threading.Lock()
_cache: dict[str, dict] = {}
_spent = 0.0


def _load():
    if CACHE.exists():
        for line in CACHE.open():
            try:  # several processes append concurrently; tolerate a torn line
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            _cache[rec["key"]] = rec


_load()


def spent() -> float:
    return _spent


def _key(model, messages, temperature, sample):
    blob = json.dumps([model, messages, temperature, sample], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def parse_json(text: str):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def chat(model: str, messages: list[dict], temperature: float = 0.7, sample: int = 0) -> dict:
    """Return {'text', 'json', 'usage'}; cached by (model, messages, temperature, sample)."""
    global _spent
    k = _key(model, messages, temperature, sample)
    with _lock:
        if k in _cache:
            return _cache[k]
        if _spent > SPEND_CAP_USD:
            raise RuntimeError(f"spend cap reached: ${_spent:.3f}")
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": 700}
    if model.startswith("openai/"):
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                 "Content-Type": "application/json"},
    )
    last = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                resp = json.load(r)
            text = resp["choices"][0]["message"]["content"] or ""
            usage = resp.get("usage", {})
            js = parse_json(text)
            if js is None and attempt < 2:
                last = "unparseable"
                continue
            break
        except Exception as e:  # network drops, 5xx, malformed bodies
            last = e
            time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"LLM call failed: {last}")
    p, c = PRICES.get(model, (1e-6, 4e-6))
    cost = usage.get("prompt_tokens", 0) * p + usage.get("completion_tokens", 0) * c
    rec = {"key": k, "model": model, "text": text, "json": js, "usage": usage, "cost": cost,
           # provenance (recorded from v2 onward; v1 cache rows lack these fields)
           "resp_model": resp.get("model"), "provider": resp.get("provider"), "resp_id": resp.get("id"),
           "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    with _lock:
        _spent += cost
        _cache[k] = rec
        with CACHE.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    return rec
