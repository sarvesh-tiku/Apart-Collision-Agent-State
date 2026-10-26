"""Emit LaTeX macros and tables for the revised paper from results/v2/summary_v2.json.

Macros: \\vv{model}{arm}{metric} for arm-level numbers and \\vc{model}{contrast}{field} for paired
contrasts. Model keys: gptfour, gemini, llama, gptfouro, haiku. All values are percentages unless
the field name ends in 'p' (p-value) or the metric is an information quantity (bits).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
S = json.loads((ROOT / "results" / "v2" / "summary_v2.json").read_text())
OUT = ROOT / "results" / "v2"
KEY = {"gpt-4.1-mini": "gptfour", "gemini-2.5-flash": "gemini", "llama-3.3-70b-instruct": "llama",
       "gpt-4o-mini": "gptfouro", "claude-haiku-4.5": "haiku"}
MODELS = ["gpt-4.1-mini", "gemini-2.5-flash", "llama-3.3-70b-instruct", "gpt-4o-mini"]
PRETTY = {"gpt-4.1-mini": "GPT-4.1-mini", "gemini-2.5-flash": "Gemini-2.5-Flash",
          "llama-3.3-70b-instruct": "Llama-3.3-70B", "gpt-4o-mini": "GPT-4o-mini"}


def pct(x):
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100*x:.0f}"


def pval(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "--"
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        return f"{m}\\times10^{{{int(e)}}}"
    return f"{p:.2f}" if p >= 0.01 else f"{p:.3f}"


defs = []


def put(a, b, c, v):
    defs.append(f"\\expandafter\\def\\csname vv@{a}@{b}@{c}\\endcsname{{{v}}}")


for m, arms in S["arms"].items():
    k = KEY[m]
    for a, e in arms.items():
        put(k, a, "n", e["n"])
        for f in ("recover", "recover_valid", "follow", "lift", "harm", "acc"):
            if e.get(f) is not None:
                put(k, a, f, pct(e[f][0]))
                put(k, a, f + "ci", f"[{pct(e[f][1])}, {pct(e[f][2])}]")
        for f in ("abstain",):
            if f in e:
                put(k, a, f, pct(e[f]))
        if "p_valid" in e:
            put(k, a, "pvalid", pval(e["p_valid"]))
        put(k, a, "nviol", e.get("n_violating_target", "--"))
        put(k, a, "liftundef", pct(e.get("lift_undefined_share_v1estimator")))
for src in ("confirm", "v2"):
    for m, cs in S[src].items():
        k = KEY[m]
        for name, v in cs.items():
            nm = name.replace("|", "vs").replace("_", "")
            put(k, nm, "d", pct(v["diff"][0]))
            put(k, nm, "ci", f"[{pct(v['diff'][1])}, {pct(v['diff'][2])}]")
            put(k, nm, "n", v["n"])
            if "p" in v:
                put(k, nm, "p", pval(v["p"]))
            if "p_holm" in v:
                put(k, nm, "ph", pval(v["p_holm"]))
for m, v in S["monitor"].items():
    k = KEY[m]
    for f in ("rule_auroc", "logit_cv_auroc", "llm_auroc"):
        put(k, "mon", f.replace("_", ""), f"{v[f]:.2f}")
    put(k, "mon", "ruleidhit", pct(v["rule_id_hit"]))
for m, arms in S["info"].items():
    k = KEY[m]
    for a, v in arms.items():
        put(k, a, "mi", f"{v['marginal']:.2f}")
        put(k, a, "micond", f"{v['cond_lb_decoded']:.2f}")
for m, R in S["repeat"].items():
    k = KEY[m]
    for ch, v in R.items():
        if v["per_sample"]:
            put(k, ch, "rep", "/".join(pct(x) for x in v["per_sample"]))
            put(k, ch, "agree", pct(v["guess_agree_all3"]))
for pair, v in S["cross"].items():
    a, b = pair.split("->")
    put(KEY[a] + KEY[b], "cross", "rec", pct(v["shared"]["recover"][0]))
    put(KEY[a] + KEY[b], "cross", "d", pct(v["vs_null"]["diff"][0]))
    put(KEY[a] + KEY[b], "cross", "p", pval(v["vs_null"]["p"]))
    if v.get("permuted"):
        put(KEY[a] + KEY[b], "cross", "fol", pct(v["permuted"]["follow"][0]))
(OUT / "numbers_v2.tex").write_text("\n".join(defs) + "\n")

# ---------------------------------------------------------------- main results table
def cell(m, a, f="recover"):
    e = S["arms"][m].get(a)
    if not e or e.get(f) is None:
        return "--"
    return pct(e[f][0])


def rh(m, a, contrast=None, src="confirm"):
    """recovery (harm); dagger if the listed paired contrast survives Holm (confirmatory) or p<.05."""
    e = S["arms"][m].get(a)
    if not e:
        return "--"
    mark = ""
    if contrast:
        v = S[src][m].get(contrast)
        if v:
            p = v.get("p_holm", v.get("p", 1))
            mark = "$^{\\dagger}$" if p < 0.05 else ""
    rec = pct(e["recover"][0]) if e.get("recover") else "--"
    return f"{rec}{mark}\\,({pct(e['harm'][0])})"


rows = [
    ("Identification", None),
    ("Channel removed (``unavailable'')", ("removed", None, "confirm")),
    ("Honest A's tickets (structure-preserving null)", ("null_honestA", "null_honestA_vs_removed", "v2")),
    ("Recompute all fields (canonical)", ("canonical", None, "confirm")),
    ("Information channels", None),
    ("Shared metadata, no code", ("shared", "shared|removed", "confirm")),
    ("Metadata permuted (decoy)", ("permuted", None, "confirm")),
    ("Direct message + A's own tickets", ("direct", "direct|removed", "confirm")),
    ("Message only, canonical tickets", ("msg_only", "msg_only_vs_canonical", "v2")),
    ("Pre-agreed code in priority", ("cb_prio_shared", "cb_prio_shared|cb_prio_removed", "confirm")),
    ("\\quad + strip priority", ("cb_prio_strip", None, "confirm")),
    ("\\quad + recompute all", ("cb_prio_canonical", None, "confirm")),
    ("Pre-agreed code in risk\\_tag", ("cb_tag_shared", "cb_tag_shared|cb_tag_removed", "confirm")),
    ("\\quad + strip priority", ("cb_tag_strip", None, "confirm")),
    ("\\quad + recompute all", ("cb_tag_canonical", None, "confirm")),
    ("Adaptive attacker (knows the sanitizer)", None),
    ("Knows priority is stripped, no code", ("aware_strip", "aware_strip_vs_canonical", "v2")),
    ("Tags recomputed; free-text note kept", ("notes_verbatim", "notes_verbatim_vs_none", "v2")),
    ("\\quad note redacted (IDs, numbers, countries)", ("notes_redacted", "notes_redacted_vs_none", "v2")),
    ("\\quad note dropped", ("notes_none", None, "v2")),
]
L = ["\\begin{tabular}{@{}l" + "c" * len(MODELS) + "@{}}", "\\toprule",
     "Arm & " + " & ".join(PRETTY[m] for m in MODELS) + " \\\\", "\\midrule"]
for lab, spec in rows:
    if spec is None:
        if L[-1] != "\\midrule":
            L.append("\\midrule")
        L.append(f"\\multicolumn{{{len(MODELS)+1}}}{{@{{}}l}}{{\\textit{{{lab}}}}} \\\\")
        continue
    a, con, src = spec
    L.append(lab + " & " + " & ".join(rh(m, a, con, src) for m in MODELS) + " \\\\")
L += ["\\midrule", "$n$ matched seeds & " + " & ".join(
    f"{S['arms'][m]['shared']['n']}/{S['arms'][m].get('notes_none', {}).get('n', '--')}" for m in MODELS) + " \\\\",
      "\\bottomrule", "\\end{tabular}"]
(OUT / "table_main_v2.tex").write_text("\n".join(L) + "\n")

# ---------------------------------------------------------------- monitor table
L = ["\\begin{tabular}{@{}lcccc@{}}", "\\toprule",
     "Detector & " + " & ".join(PRETTY[m] for m in MODELS) + " \\\\", "\\midrule",
     "LLM monitor (zero-shot, v1) & " + " & ".join(f"{S['monitor'][m]['llm_auroc']:.2f}" for m in MODELS) + " \\\\",
     "Fixed structural rule & " + " & ".join(f"{S['monitor'][m]['rule_auroc']:.2f}" for m in MODELS) + " \\\\",
     "Logistic, 5-fold CV & " + " & ".join(f"{S['monitor'][m]['logit_cv_auroc']:.2f}" for m in MODELS) + " \\\\",
     "\\midrule",
     "Rule: suspected ID $=$ target (\\%) & " + " & ".join(pct(S['monitor'][m]['rule_id_hit']) for m in MODELS) + " \\\\",
     "\\bottomrule", "\\end{tabular}"]
(OUT / "table_monitor_v2.tex").write_text("\n".join(L) + "\n")
print("wrote", len(defs), "macros")

# ---------------------------------------------------------------- appendix: every v2 arm, per model
ARMS_V2 = [("null_honestA", "Honest-A null (collusive B)"), ("unilateral_A", "Unilateral A (honest B)"),
           ("benign_flag", "Benign flag (review-first)"), ("benign_null", "Benign null (review-first)"),
           ("msg_only", "Message only, canonical"), ("noguess_shared", "Guess-free, verbatim"),
           ("noguess_canonical", "Guess-free, canonical"), ("aware_strip", "Aware: priority stripped"),
           ("aware_strip_canonical", "Aware: canonical null"), ("notes_verbatim", "Note verbatim"),
           ("notes_redacted", "Note redacted"), ("notes_none", "Note dropped"),
           ("notes_honest_verbatim", "Honest, note verbatim"), ("notes_honest_redacted", "Honest, note redacted"),
           ("notes_honest_none", "Honest, note dropped")]


def f3(e, k):
    if not e or e.get(k) is None:
        return "--"
    v = e[k]
    return f"{pct(v[0])}\\,{{\\tiny[{pct(v[1])},{pct(v[2])}]}}"


for m in MODELS:
    L = []
    for a, lab in ARMS_V2:
        e = S["arms"][m].get(a)
        if not e:
            continue
        rec = f3(e, "recover") if e.get("recover") else "--"
        ab = pct(e["abstain"]) if "abstain" in e else "--"
        pv = pval(e["p_valid"]) if "p_valid" in e else "--"
        L.append(f"{lab} & {e['n']} & {rec} & {ab} & ${pv}$ & {f3(e, 'lift')} & {f3(e, 'harm')} & {f3(e, 'acc')} \\\\")
    (OUT / f"v2arms_{m}.tex").write_text("\n".join(L) + "\n")

# cross-model table
L = []
for a in MODELS:
    for b in MODELS:
        if a == b:
            continue
        v = S["cross"].get(f"{a}->{b}")
        if not v:
            continue
        L.append(f"{PRETTY[a]} & {PRETTY[b]} & {f3(v['shared'], 'recover')} & "
                 f"{pct(v['vs_null']['diff'][0])}\\,{{\\tiny[{pct(v['vs_null']['diff'][1])},{pct(v['vs_null']['diff'][2])}]}} & "
                 f"${pval(v['vs_null']['p'])}$ & {f3(v['permuted'], 'follow') if v.get('permuted') else '--'} \\\\")
(OUT / "cross_table.tex").write_text("\n".join(L) + "\n")

# repeats
L = []
for m, R in S["repeat"].items():
    for ch, v in R.items():
        L.append(f"{PRETTY[m]} & {ch} & {v['n']} & {' / '.join(pct(x) for x in v['per_sample'])} & {pct(v['guess_agree_all3'])} \\\\")
(OUT / "repeat_table.tex").write_text("\n".join(L) + "\n")

# confirmatory family with Holm
L = []
for m in MODELS:
    for k, v in S["confirm"][m].items():
        a, b = k.split("|")
        L.append(f"{PRETTY[m] if k == next(iter(S['confirm'][m])) else ''} & {a.replace('_', ' ')} vs.\\ {b.replace('_', ' ')} & "
                 f"{pct(v['diff'][0])}\\,{{\\tiny[{pct(v['diff'][1])},{pct(v['diff'][2])}]}} & {v['disc'][0]}/{v['disc'][1]} & "
                 f"${pval(v['p'])}$ & ${pval(v['p_holm'])}$ \\\\")
    L.append("\\midrule")
(OUT / "confirm_table.tex").write_text("\n".join(L[:-1]) + "\n")

# information
L = []
for m in MODELS:
    for a in ("shared", "permuted", "direct", "msg_only", "cb_prio_shared", "cb_prio_permuted", "cb_tag_shared",
              "cb_tag_permuted", "notes_verbatim", "benign_flag"):
        v = S["info"].get(m, {}).get(a)
        if v:
            L.append(f"{PRETTY[m] if a == 'shared' else ''} & {a.replace('_', ' ')} & {v['marginal']:.2f} & "
                     f"{v['cond_lb_decoded']:.2f} & {v['null95']:.2f} \\\\")
    L.append("\\midrule")
(OUT / "info_table.tex").write_text("\n".join(L[:-1]) + "\n")
print("appendix tables written")

for f in OUT.glob("*.tex"):
    f.write_text(f.read_text().replace("$--$", "--"))
