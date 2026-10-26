"""Run all conditions on matched seeds. Usage: python run.py MODEL N_SEEDS [START]"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import llm
from agents import (approver_messages, clean_tickets, monitor_messages, preparer_messages,
                    transform)
from env import make_episode

OUT = Path(__file__).resolve().parent.parent / "results"

# condition -> (preparer mode, channel transform, approver colludes)
CONDITIONS = {
    "removed": ("collude", "removed", True),
    "direct": ("direct", "direct", True),
    "shared": ("collude", "shared", True),
    "permuted": ("collude", "permuted", True),
    "strip": ("collude", "strip", True),
    "canonical": ("collude", "canonical", True),
    "honest_shared": ("honest", "shared", False),
    "honest_canonical": ("honest", "canonical", False),
}


def run_seed(model: str, seed: int) -> list[dict]:
    ep = make_episode(seed)
    a_raw = {m: llm.chat(model, preparer_messages(ep, m)) for m in ("collude", "direct", "honest")}
    tickets = {m: clean_tickets(ep, r["json"]) for m, r in a_raw.items()}
    rows = []
    for cond, (amode, channel, coll) in CONDITIONS.items():
        view = transform(ep, tickets[amode], channel)
        msg = None
        if channel == "direct":
            msg = str((a_raw["direct"]["json"] or {}).get("message", ""))[:300]
        b = llm.chat(model, approver_messages(ep, view, coll, msg))
        bj = b["json"] or {}
        dec = {str(k): str(v).lower() for k, v in (bj.get("decisions") or {}).items()}
        rows.append({
            "model": model, "seed": seed, "condition": cond,
            "ids": [t.id for t in ep.txns], "labels": ep.labels(), "target": ep.target,
            "perm": ep.perm, "a_tickets": tickets[amode], "view": view, "message": msg,
            "decisions": [dec.get(t.id, "missing") for t in ep.txns],
            "a_parsed": bool((a_raw[amode]["json"] or {}).get("tickets")),
            "b_parsed": bool(dec),
            "guess": str(bj.get("coalition_guess", "")) if coll else None,
            "confidence": bj.get("confidence") if coll else None,
        })
    for amode in ("collude", "honest"):
        m = llm.chat(model, monitor_messages(ep, tickets[amode]), temperature=0.0)
        mj = m["json"] or {}
        rows.append({"model": model, "seed": seed, "condition": f"monitor_{amode}",
                     "target": ep.target, "ids": [t.id for t in ep.txns],
                     "suspicion": mj.get("suspicion"), "suspected_id": mj.get("suspected_id")})
    return rows


CB_CHANNELS = ["removed", "shared", "permuted", "strip", "canonical"]


def run_codebook(model: str, seed: int) -> list[dict]:
    """Positive control: same design, but A and B share a pre-agreed convention."""
    ep = make_episode(seed)
    rows = []
    for code in ("prio", "tag"):
        a = llm.chat(model, preparer_messages(ep, "collude", code))
        tickets = clean_tickets(ep, a["json"])
        for channel in CB_CHANNELS:
            view = transform(ep, tickets, channel)
            bj = llm.chat(model, approver_messages(ep, view, True, code=code))["json"] or {}
            dec = {str(k): str(v).lower() for k, v in (bj.get("decisions") or {}).items()}
            rows.append({
                "model": model, "seed": seed, "condition": f"cb_{code}_{channel}",
                "ids": [t.id for t in ep.txns], "labels": ep.labels(), "target": ep.target,
                "perm": ep.perm, "a_tickets": tickets, "view": view, "message": None,
                "decisions": [dec.get(t.id, "missing") for t in ep.txns],
                "a_parsed": bool((a["json"] or {}).get("tickets")), "b_parsed": bool(dec),
                "guess": str(bj.get("coalition_guess", "")), "confidence": bj.get("confidence"),
            })
    return rows


def main():
    codebook = "--codebook" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--codebook"]
    model, n = args[0], int(args[1])
    start = int(args[2]) if len(args) > 2 else 0
    tag = model.split("/")[-1]
    out = OUT / f"{'cb' if codebook else 'rows'}_{tag}.jsonl"
    fn = run_codebook if codebook else run_seed
    done = set()
    if out.exists():
        done = {json.loads(l)["seed"] for l in out.open()}
    seeds = [s for s in range(start, start + n) if s not in done]
    def safe(s):
        try:
            return fn(model, s)
        except Exception as e:  # one bad seed must not kill the run; rerun resumes it
            print(f"seed {s} failed: {e}", flush=True)
            return []

    with ThreadPoolExecutor(8) as ex, out.open("a") as f:
        for rows in ex.map(safe, seeds):
            if not rows:
                continue
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"seed {rows[0]['seed']} done, spent ${llm.spent():.3f}", flush=True)


if __name__ == "__main__":
    main()
