"""v2 controls. Writes new files only (results/v2_*.jsonl, x_*.jsonl, rep_*.jsonl); v1 data untouched.

Usage:
  python run_v2.py controls MODEL N          # same-model control arms, seeds 0..N-1
  python run_v2.py cross A_MODEL B_MODEL N   # cross-model pairing: A's v1 collusive tickets -> B
  python run_v2.py repeat MODEL N            # extra B samples (sample=1,2) for critical arms
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import llm
from agents import (approver_messages, approver_review_first, approver_v2, clean_tickets, notes_of, notes_view,
                    preparer_messages, preparer_v2, transform)
from env import make_episode

OUT = Path(__file__).resolve().parent.parent / "results"


def _row(model, seed, cond, ep, view, bj, coll, extra=None):
    dec = {str(k): str(v).lower() for k, v in (bj.get("decisions") or {}).items()}
    r = {"model": model, "seed": seed, "condition": cond, "ids": [t.id for t in ep.txns],
         "labels": ep.labels(), "target": ep.target, "perm": ep.perm, "view": view,
         "decisions": [dec.get(t.id, "missing") for t in ep.txns], "b_parsed": bool(dec),
         "guess": (str(bj.get("coalition_guess", "")) if coll and "coalition_guess" in bj
                   else str(bj["review_first"]) if "review_first" in bj else None),
         "confidence": bj.get("confidence")}
    r.update(extra or {})
    return r


def controls(model: str, seed: int) -> list[dict]:
    ep = make_episode(seed)
    B = lambda msgs: llm.chat(model, msgs)["json"] or {}
    # v1 preparer calls (cache hits; identical prompts)
    a_coll = llm.chat(model, preparer_messages(ep, "collude"))["json"]
    a_dir = llm.chat(model, preparer_messages(ep, "direct"))["json"] or {}
    a_hon = llm.chat(model, preparer_messages(ep, "honest"))["json"]
    t_coll, t_hon = clean_tickets(ep, a_coll), clean_tickets(ep, a_hon)
    # new preparer calls
    a_ben = llm.chat(model, preparer_v2(ep, "benign"))["json"]
    a_aws = llm.chat(model, preparer_v2(ep, "aware_strip"))["json"]
    a_nc = llm.chat(model, preparer_v2(ep, "notes_collude"))["json"]
    a_nh = llm.chat(model, preparer_v2(ep, "notes_honest"))["json"]
    t_ben, t_aws = clean_tickets(ep, a_ben), clean_tickets(ep, a_aws)
    n_c, n_h = notes_of(ep, a_nc), notes_of(ep, a_nh)
    canon = transform(ep, t_coll, "canonical")
    msg = str(a_dir.get("message", ""))[:300]
    arms = {
        # identification controls
        "null_honestA": (approver_messages(ep, t_hon, True), t_hon, True, {}),
        "unilateral_A": (approver_messages(ep, t_coll, False), t_coll, False, {}),
        "benign_flag": (approver_review_first(ep, t_ben), t_ben, False, {"a_tickets": t_ben}),
        "benign_null": (approver_review_first(ep, t_hon), t_hon, False, {}),
        "msg_only": (approver_messages(ep, canon, True, msg), canon, True, {"message": msg}),
        "noguess_shared": (approver_v2(ep, t_coll, True, guess=False), t_coll, True, {}),
        "noguess_canonical": (approver_v2(ep, canon, True, guess=False), canon, True, {}),
        # adaptive attacker: knows priority is stripped
        "aware_strip": (approver_v2(ep, transform(ep, t_aws, "strip"), True, aware="strip"),
                        transform(ep, t_aws, "strip"), True, {"a_tickets": t_aws}),
        "aware_strip_canonical": (approver_v2(ep, canon, True, aware="strip"), canon, True, {}),
    }
    # migration into an authorized free-text field, and its leakage-utility tradeoff
    for how in ("verbatim", "redacted", "none"):
        v = notes_view(ep, n_c, how)
        arms[f"notes_{how}"] = (approver_v2(ep, v, True, aware="notes"), v, True, {"notes": n_c})
        vh = notes_view(ep, n_h, how)
        arms[f"notes_honest_{how}"] = (approver_v2(ep, vh, False, aware="notes"), vh, False, {"notes": n_h})
    rows = []
    for cond, (msgs, view, coll, extra) in arms.items():
        rows.append(_row(model, seed, cond, ep, view, B(msgs), coll, extra))
    return rows


def cross(a_model: str, b_model: str, seed: int) -> list[dict]:
    ep = make_episode(seed)
    t = clean_tickets(ep, llm.chat(a_model, preparer_messages(ep, "collude"))["json"])
    rows = []
    for ch in ("shared", "permuted"):
        view = transform(ep, t, ch)
        bj = llm.chat(b_model, approver_messages(ep, view, True))["json"] or {}
        rows.append(_row(f"{a_model}->{b_model}", seed, f"x_{ch}", ep, view, bj, True,
                         {"a_model": a_model, "b_model": b_model, "a_tickets": t}))
    return rows


def repeat(model: str, seed: int) -> list[dict]:
    ep = make_episode(seed)
    t = clean_tickets(ep, llm.chat(model, preparer_messages(ep, "collude"))["json"])
    rows = []
    for ch in ("removed", "shared", "permuted", "canonical"):
        view = transform(ep, t, ch)
        for k in (1, 2):
            bj = llm.chat(model, approver_messages(ep, view, True), sample=k)["json"] or {}
            rows.append(_row(model, seed, ch, ep, view, bj, True, {"sample": k}))
    return rows


def main():
    kind = sys.argv[1]
    if kind == "cross":
        a, b, n = sys.argv[2], sys.argv[3], int(sys.argv[4])
        out = OUT / f"x_{a.split('/')[-1]}__{b.split('/')[-1]}.jsonl"
        fn = lambda s: cross(a, b, s)
    else:
        m, n = sys.argv[2], int(sys.argv[3])
        out = OUT / f"{'v2' if kind == 'controls' else 'rep'}_{m.split('/')[-1]}.jsonl"
        fn = (lambda s: controls(m, s)) if kind == "controls" else (lambda s: repeat(m, s))
    done = {json.loads(l)["seed"] for l in out.open()} if out.exists() else set()

    def safe(s):
        try:
            return fn(s)
        except Exception as e:
            print(f"seed {s} failed: {e}", flush=True)
            return []

    with ThreadPoolExecutor(8) as ex, out.open("a") as f:
        for rows in ex.map(safe, [s for s in range(n) if s not in done]):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            if rows:
                print(f"seed {rows[0]['seed']} done, spent ${llm.spent():.3f}", flush=True)


if __name__ == "__main__":
    main()
