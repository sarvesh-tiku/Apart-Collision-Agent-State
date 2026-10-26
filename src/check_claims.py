"""Check every qualitative claim in paper/body.tex against results/v2/summary_v2.json. Prints PASS/FAIL."""
import json
from pathlib import Path

S = json.loads((Path(__file__).resolve().parent.parent / "results/v2/summary_v2.json").read_text())
M = ["gpt-4.1-mini", "gemini-2.5-flash", "llama-3.3-70b-instruct", "gpt-4o-mini"]
A, C, V, X, MON = S["arms"], S["confirm"], S["v2"], S["cross"], S["monitor"]
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))


for m in M:
    for a in ("removed", "canonical", "null_honestA"):
        p2 = A[m][a]["p_valid_two"]
        check(f"F1 null holds (two-sided p>.05/12) {m} {a}", p2 > 0.05 / 12, f"rec={A[m][a]['recover'][0]:.2f} p={p2:.3g}")
    v = V[m]["null_honestA_vs_removed"]
    check(f"F1 honest-A null ~ removed after x4 correction {m}", v["p"] * 4 > 0.05, f"d={v['diff'][0]:.2f} p={v['p']:.3g}")
for m in M:
    p = C[m]["shared|removed"]["p_holm"]
    check(f"F2 shared vs removed {'sig' if m == 'gemini-2.5-flash' else 'n.s.'} {m}",
          (p < 0.05) == (m == "gemini-2.5-flash"), f"holm={p:.3g}")
for k, v in X.items():
    a = k.split("->")[0]
    sig = v["vs_null"]["p"] < 0.05 / 16
    check(f"F2 cross {k} {'sig' if a == 'gemini-2.5-flash' else 'n.s.'} at .05/16", sig == (a == "gemini-2.5-flash"),
          f"rec={v['shared']['recover'][0]:.2f} p={v['vs_null']['p']:.3g}")
check("F3 gemini shared vs permuted Holm sig", C["gemini-2.5-flash"]["shared|permuted"]["p_holm"] < 0.05)
for m in M:
    check(f"F3 prio-code permuted follow == 100% {m}", A[m]["cb_prio_permuted"]["follow"][0] == 1.0,
          f"{A[m]['cb_prio_permuted']['follow'][0]:.2f}")
mo = {m: A[m]["msg_only"]["recover"][0] for m in M}
check("F4 msg_only min is Gemini, max is GPT-4.1-mini", min(mo, key=mo.get) == "gemini-2.5-flash" and max(mo, key=mo.get) == "gpt-4.1-mini", str({k: round(v, 2) for k, v in mo.items()}))
dr = {m: A[m]["direct"]["recover"][0] for m in M}
check("F4 direct range gpt4o-mini..gemini", min(dr, key=dr.get) in ("gpt-4o-mini", "llama-3.3-70b-instruct") and max(dr, key=dr.get) == "gemini-2.5-flash", str({k: round(v, 2) for k, v in dr.items()}))
bf = {m: A[m]["benign_flag"]["recover"][0] for m in M}
print("     benign review-first:", {k: round(v, 2) for k, v in bf.items()})
for m in M:
    v = V[m]["benign_harm_vs_null"]
    check(f"F5 benign harm not detectably higher {m}", v["p"] > 0.05 or v["diff"][0] <= 0, f"d={v['diff'][0]:.2f} p={v['p']:.3g}")
    v = V[m]["unilateral_harm_vs_honest"]
    check(f"F5 unilateral harm n.s. {m}", v["p"] > 0.05, f"d={v['diff'][0]:.2f} p={v['p']:.3g}")
v = V["gemini-2.5-flash"]["harm_shared_vs_removed"]
check("F5 gemini shared harm n.s. vs removed", v["p"] > 0.05, f"d={v['diff']} p={v['p']:.3g}")
v = V["gemini-2.5-flash"]["noguess_lift_shared_vs_canonical"]
check("F6 gemini guess-free lift sig", v["p"] < 0.05, f"d={v['diff'][0]:.2f} p={v['p']:.3g}")
for m in M:
    check(f"F6 rule monitor AUROC > LLM AUROC {m}", MON[m]["rule_auroc"] > MON[m]["llm_auroc"],
          f"rule={MON[m]['rule_auroc']:.2f} llm={MON[m]['llm_auroc']:.2f} logit={MON[m]['logit_cv_auroc']:.2f}")
for m in M:
    v = V[m]["aware_strip_vs_canonical"]
    check(f"D aware-strip converges only for Llama {m}", (v["p"] < 0.05) == (m == "llama-3.3-70b-instruct"), f"d={v['diff'][0]:.2f} p={v['p']:.3g}")
mig = [m for m in M if V[m]["notes_verbatim_vs_none"]["p"] < 0.05 and V[m]["notes_verbatim_vs_none"]["diff"][0] > 0]
check("D notes migration exactly Gemini + GPT-4o-mini + Llama", set(mig) == {"gemini-2.5-flash", "gpt-4o-mini", "llama-3.3-70b-instruct"}, str(mig))
for m in mig:
    v = V[m]["notes_redacted_vs_none"]
    check(f"D redaction still leaks {m}", v["p"] < 0.05, f"d={v['diff'][0]:.2f}")
for m in M:
    v = V[m]["notes_honest_acc_verbatim_vs_none"]["diff"]
    print(f"     honest note utility {m}: {100*v[0]:.1f} [{100*v[1]:.1f},{100*v[2]:.1f}]")
for m in M:
    for k in ("cb_prio_shared|cb_prio_removed", "cb_prio_shared|cb_prio_strip", "cb_prio_shared|cb_prio_canonical",
              "cb_tag_shared|cb_tag_removed", "cb_tag_shared|cb_tag_canonical"):
        check(f"S code contrast Holm sig {m} {k}", C[m][k]["p_holm"] < 0.05, f"holm={C[m][k]['p_holm']:.3g}")
    check(f"S tag strip n.s. {m}", C[m]["cb_tag_shared|cb_tag_strip"]["p_holm"] > 0.05)
    v = V[m]["harm_cbprio_shared_vs_canonical"]
    check(f"D prio-code harm falls with recompute {m}", v["p"] < 0.05, f"d={v['diff'][0]:.2f}")
print("\nALL PASS" if ok else "\nSOME CLAIMS FAIL — revise text")
