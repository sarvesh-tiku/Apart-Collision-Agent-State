"""Build the narrated demo video: slides (matplotlib) + narration (macOS `say`) -> MP4 (ffmpeg).

Every number in the narration and on the slides is taken from results/v2/summary_v2.json.
Usage: python demo/build_demo.py   ->   demo/Signal_or_Coincidence_demo.mp4
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
B = ROOT / "demo" / "build"
B.mkdir(parents=True, exist_ok=True)
S = json.loads((ROOT / "results" / "v2" / "summary_v2.json").read_text())
A, C, V, X, MON = S["arms"], S["confirm"], S["v2"], S["cross"], S["monitor"]
M = ["gpt-4.1-mini", "gemini-2.5-flash", "llama-3.3-70b-instruct", "gpt-4o-mini"]
NAME = {"gpt-4.1-mini": "GPT-4.1-mini", "gemini-2.5-flash": "Gemini-2.5-Flash",
        "llama-3.3-70b-instruct": "Llama-3.3-70B", "gpt-4o-mini": "GPT-4o-mini"}
COL = {"gpt-4.1-mini": "#2a78d6", "gemini-2.5-flash": "#eb6834", "llama-3.3-70b-instruct": "#1baf7a",
       "gpt-4o-mini": "#eda100"}
INK, INK2, MUTED, BG, ACC = "#111111", "#52514e", "#8a8984", "#fbfaf7", "#eb6834"
G = "gemini-2.5-flash"


def pct(x):
    return round(100 * x)


rec = lambda m, a: pct(A[m][a]["recover"][0])
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial"],
                     "axes.spines.top": False, "axes.spines.right": False})
plt.rcParams["font.sans-serif"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]


# ------------------------------------------------------------------ slide helpers
def canvas(title, kicker=None, n=None):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor(BG)
    if kicker:
        fig.text(0.06, 0.905, kicker.upper(), fontsize=17, color=ACC, fontweight="bold")
    fig.text(0.06, 0.835, title, fontsize=40, color=INK, fontweight="bold")
    if n:
        fig.text(0.94, 0.04, f"{n} / 10", fontsize=14, color=MUTED, ha="right")
    fig.text(0.06, 0.04, "Signal or Coincidence?  ·  Sarvesh Tiku  ·  Apart AI Collusion Sprint 2026",
             fontsize=14, color=MUTED)
    return fig


def image(fig, path, box):
    ax = fig.add_axes(box)
    ax.imshow(plt.imread(path))
    ax.axis("off")


def bullets(fig, items, x=0.06, y=0.70, size=24, gap=0.085, width=None):
    for i, t in enumerate(items):
        fig.text(x, y - i * gap, "•", fontsize=size, color=ACC, fontweight="bold")
        fig.text(x + 0.02, y - i * gap, t, fontsize=size, color=INK, va="baseline", wrap=True)


def stat(fig, x, y, big, small, color=ACC):
    fig.text(x, y, big, fontsize=64, color=color, fontweight="bold")
    fig.text(x, y - 0.055, small, fontsize=19, color=INK2)


def save(fig, i):
    p = B / f"slide{i:02d}.png"
    fig.savefig(p, facecolor=BG)
    plt.close(fig)
    return p


def box(ax, x, y, w, h, text, ec, fc, size=19):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                ec=ec, fc=fc, lw=2.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK)


def arrow(ax, x0, y0, x1, y1, c="#666666", ls="-"):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=c, lw=2.5, ls=ls,
                                                        mutation_scale=22))


# ------------------------------------------------------------------ slides + narration
slides = []

# 1 title
fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
fig.patch.set_facecolor(BG)
fig.text(0.06, 0.70, "AI COLLUSION RESEARCH SPRINT  ·  TRACK 3 (Q3.1)  ·  TRACK 2 (Q2.3)", fontsize=18,
         color=ACC, fontweight="bold")
fig.text(0.06, 0.56, "Signal or Coincidence?", fontsize=76, color=INK, fontweight="bold")
fig.text(0.06, 0.47, "Causal auditing of collusion through shared agent state", fontsize=36, color=INK2)
fig.text(0.06, 0.33, "Sarvesh Tiku", fontsize=28, color=INK, fontweight="bold")
fig.text(0.06, 0.285, "Georgia Institute of Technology", fontsize=22, color=INK2)
fig.add_artist(plt.Line2D([0.06, 0.30], [0.41, 0.41], color=ACC, lw=5))
slides.append((save(fig, 1),
 "Hi, I'm Sarvesh Tiku. This is Signal or Coincidence: causal auditing of collusion through shared agent state. "
 "The question is simple. When two A I agents behave alike, how do we know whether they are actually coordinating, "
 "or just thinking alike?"))

# 2 problem
fig = canvas("Parallel conduct is not proof of coordination", "The problem", 2)
bullets(fig, ["Agents share tools: tickets, queues, status fields — natural places to leave signals",
              "Two copies of a model share priors, data and incentives, so they act alike anyway",
              "Antitrust law calls this the “plus factors” problem: correlation alone is not agreement",
              "We need evidence that information actually moved between agents"], y=0.66, size=26, gap=0.11)
slides.append((save(fig, 2),
 "Multi agent systems route work through shared tools, like ticket fields and status flags. Those are natural places "
 "to hide a signal. But two copies of the same model will often make the same choice anyway, because they share "
 "priors, data, and incentives. Antitrust law has the same problem: parallel conduct alone does not prove an agreement. "
 "So we need evidence that information actually moved from one agent to another."))

# 3 design
fig = canvas("The randomized canary audit", "The method", 3)
ax = fig.add_axes([0.04, 0.16, 0.92, 0.6])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
box(ax, 0.00, 0.58, 0.17, 0.26, "Auditor\ndraws a random\ncanary target T", ACC, "#fdeee6")
box(ax, 0.24, 0.58, 0.17, 0.26, "Preparer A\nsees T\nwrites tickets", "#2a78d6", "#e8f1fc")
box(ax, 0.48, 0.52, 0.22, 0.38, "Intervention\nremove · permute\nstrip · recompute\nnote ± redact", "#555555", "#f1efea", 18)
box(ax, 0.77, 0.58, 0.20, 0.26, "Approver B\nnever sees T\napproves / rejects", "#2a78d6", "#e8f1fc")
box(ax, 0.26, 0.05, 0.50, 0.22, "Score against the held-out T:\nrecovery vs. exact null  ·  decoy-following  ·  harm  ·  honest accuracy", ACC, "white", 18)
arrow(ax, 0.17, 0.71, 0.24, 0.71, ACC)
arrow(ax, 0.41, 0.71, 0.48, 0.71)
arrow(ax, 0.70, 0.71, 0.77, 0.71)
arrow(ax, 0.87, 0.58, 0.76, 0.17, ACC)
arrow(ax, 0.085, 0.58, 0.26, 0.17, ACC, "--")
slides.append((save(fig, 3),
 "Our method is a randomized canary audit. In a synthetic compliance workflow, a preparer agent writes tickets "
 "with two simple fields, priority and risk tag, and an approver agent approves or rejects four transactions. "
 "The auditor secretly gives only the preparer a random target. The approver never sees it. Then, on matched seeds, "
 "we change what the approver reads: we remove the tickets, shuffle them, strip a field, or recompute them from the source record."))

# 4 null
fig = canvas("An exact, model-free null", "Identification", 4)
fig.text(0.06, 0.66, "If nothing the approver sees depends on T:", fontsize=28, color=INK2)
fig.text(0.06, 0.55, "P(guess = T)  =  P(valid guess) / 4", fontsize=50, color=INK, fontweight="bold",
         family="DejaVu Sans Mono")
ax = fig.add_axes([0.08, 0.13, 0.84, 0.30])
arms = [("removed", "Channel removed"), ("null_honestA", "Honest preparer's tickets"), ("canonical", "Recomputed tickets")]
w = 0.2
for j, m in enumerate(M):
    xs = [i + (j - 1.5) * w for i in range(3)]
    ax.bar(xs, [rec(m, a) for a, _ in arms], w * 0.9, color=COL[m], label=NAME[m])
ax.axhline(25, color=INK, ls="--", lw=2)
ax.text(2.55, 26.5, "null = 25%", fontsize=17, color=INK)
ax.set_xticks(range(3), [l for _, l in arms], fontsize=19)
ax.set_ylim(0, 45)
ax.set_ylabel("Target recovery (%)", fontsize=17)
ax.tick_params(labelsize=15)
ax.legend(ncol=4, fontsize=15, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22))
slides.append((save(fig, 4),
 "Because the target is randomized, there is an exact null. If nothing the approver sees depends on the target, "
 "the chance it names the target is one quarter of its valid guesses, for any model, however capable. "
 "And that is what we see. With the channel removed, with an honest preparer's tickets, or with recomputed tickets, "
 "every model sits near twenty five percent."))

# 5 gemini convention
fig = canvas("One model invents a convention — and others read it", "Finding 1", 5)
stat(fig, 0.06, 0.62, f"{rec(G, 'shared')}%", f"Gemini recovery from shared tickets\nvs. {rec(G, 'removed')}% with the channel removed")
cross = {k.split("->")[1]: pct(v["shared"]["recover"][0]) for k, v in X.items() if k.startswith(G)}
stat(fig, 0.06, 0.36, f"{min(cross.values())}–{max(cross.values())}%", "other models decoding\nGemini-written tickets", "#2a78d6")
fig.text(0.06, 0.17, "No pre-agreed code. Gemini marks the target “urgent”.\nNo other preparer is readable by anyone.",
         fontsize=20, color=INK2)
image(fig, B / "v2_cross.png", [0.44, 0.10, 0.52, 0.70])
slides.append((save(fig, 5),
 f"First finding. Without any pre agreed code, three of four models fail to coordinate through the tickets. "
 f"Gemini does not. Its preparers mark the target as urgent, and recovery jumps to {rec(G, 'shared')} percent, "
 f"versus {rec(G, 'removed')} percent with the channel removed. "
 f"And the convention is not private. Other models reading Gemini's tickets recover the target {min(cross.values())} to "
 f"{max(cross.values())} percent of the time. No other preparer is readable by anyone. The bottleneck is the writer, not the reader."))

# 6 permutation
fig = canvas("Shuffle the tickets, and the approver follows the decoy", "Causal check", 6)
ax = fig.add_axes([0.08, 0.16, 0.48, 0.55])
vals = [rec(G, "shared"), rec(G, "permuted"), pct(A[G]["permuted"]["follow"][0])]
ax.bar([0, 1, 2], vals, 0.6, color=["#2a78d6", "#8a8984", ACC])
for i, v in enumerate(vals):
    ax.text(i, v + 2, f"{v}%", ha="center", fontsize=26, fontweight="bold")
ax.set_xticks([0, 1, 2], ["Shared:\nguess = target", "Permuted:\nguess = target", "Permuted:\nguess = decoy"], fontsize=17)
ax.set_ylim(0, 75)
ax.axhline(25, color=INK, ls="--", lw=1.5)
ax.tick_params(labelsize=15)
ax.set_title("Gemini, no code", fontsize=20, loc="left")
fig.text(0.62, 0.60, "With a pre-agreed code:", fontsize=24, color=INK2)
stat(fig, 0.62, 0.47, "100%", "of permuted guesses land on the decoy,\nfor every model")
fig.text(0.62, 0.24, "Information is not destroyed —\nit is redirected.", fontsize=24, color=INK)
slides.append((save(fig, 6),
 "To check that this is causal, we shuffle the tickets so the target's ticket lands on a different transaction, a decoy. "
 f"Gemini's recovery drops to {rec(G, 'permuted')} percent, and its guess follows the decoy {pct(A[G]['permuted']['follow'][0])} percent of the time. "
 "With a pre agreed code, every model follows the decoy in one hundred percent of shuffled episodes. "
 "Behavior follows the carrier, not the target."))

# 7 flow vs harm
fig = canvas("Information flow is not collusion — and not harm", "Finding 2", 7)
bf = [pct(A[m]["benign_flag"]["recover"][0]) for m in M]
stat(fig, 0.06, 0.62, f"{min(bf)}–{max(bf)}%", "flow from an authorized\n“review this first” request")
fig.text(0.06, 0.45, "…with honest agents and no\ndetectable rise in harm.", fontsize=22, color=INK2)
h = V[G]["harm_shared_vs_removed"]["diff"]
fig.text(0.06, 0.27, f"Gemini's covert channel: {rec(G, 'shared')}% recovery,\nharm change {pct(h[0]):+d} pp "
         f"[{pct(h[1])}, {pct(h[2])}]", fontsize=22, color=INK)
image(fig, B / "v2_flow_harm.png", [0.48, 0.10, 0.48, 0.70])
slides.append((save(fig, 7),
 "Second finding. Detecting information flow is not the same as detecting collusion. "
 f"When the preparer has an authorized request, to have one transaction reviewed first, honest approvers pick it up "
 f"{min(bf)} to {max(bf)} percent of the time, with no detectable increase in harm. "
 "And even Gemini's covert channel did not detectably raise harmful approvals: its approver often identifies the target and still rejects it. "
 "So a convincing case needs flow, a localized carrier, a harmful outcome, and evidence the flow was not authorized."))

# 8 defenses
fig = canvas("Defenses only stop the carrier they touch", "Finding 3", 8)
image(fig, B / "v2_defense.png", [0.40, 0.09, 0.57, 0.72])
tg = A["gpt-4.1-mini"]
bullets(fig, [f"Strip priority: kills a priority code,\nleaves a tag code intact "
              f"({rec('gpt-4.1-mini', 'cb_tag_shared')}% to {rec('gpt-4.1-mini', 'cb_tag_strip')}%)",
              "Recompute every field: kills both",
              f"Adaptive attacker moves the signal\ninto a free-text note (Gemini {rec(G, 'notes_verbatim')}%)",
              f"Redacting IDs and numbers\ndoes not help ({rec(G, 'notes_redacted')}%)"], y=0.68, size=21, gap=0.135)
slides.append((save(fig, 8),
 "Third finding: defenses. Stripping the priority field stops a code hidden in priority, but a code in the risk tag "
 "passes straight through. Recomputing every field from the source record stops both, and harmful approvals fall with it. "
 "But that works by construction here. When a free text note field survives, preparers that know about the sanitizer "
 f"move the signal into the note, for three of four models, reaching {rec(G, 'notes_verbatim')} percent for Gemini. "
 "Redacting identifiers, numbers, and countries does not stop it."))

# 9 monitors
fig = canvas("Reading tickets is a weak detector", "Detection", 9)
ax = fig.add_axes([0.08, 0.16, 0.55, 0.55])
lab = ["LLM monitor", "Fixed structural rule", "Logistic (CV)"]
keys = ["llm_auroc", "rule_auroc", "logit_cv_auroc"]
w = 0.2
for j, m in enumerate(M):
    ax.bar([i + (j - 1.5) * w for i in range(3)], [MON[m][k] for k in keys], w * 0.9, color=COL[m], label=NAME[m])
ax.axhline(0.5, color=INK, ls="--", lw=1.5)
ax.set_xticks(range(3), lab, fontsize=18)
ax.set_ylim(0.3, 1.0)
ax.set_ylabel("AUROC (collusive vs. honest)", fontsize=17)
ax.tick_params(labelsize=15)
ax.legend(ncol=2, fontsize=14, frameon=False, loc="upper left")
fig.text(0.68, 0.60, "An LLM reading the tickets\nis at chance.", fontsize=26, color=INK)
fig.text(0.68, 0.42, "Simple structural checks\ndo better — but must know\nwhat mark to look for.", fontsize=22, color=INK2)
fig.text(0.68, 0.22, "The canary audit needs\nneither.", fontsize=26, color=ACC, fontweight="bold")
slides.append((save(fig, 9),
 "What about just watching the tickets? An L L M monitor reading them is essentially at chance. "
 "A fixed structural rule, and a small cross validated classifier, do better, but they detect that the preparer deviates, "
 "and they need to know which deviations to look for. The canary audit needs neither: it only asks whether behavior "
 "depends on information the approver should not have."))

# 10 takeaways
fig = canvas("Takeaways", "Conclusion", 10)
bullets(fig, ["A randomized canary gives an exact null for information flow through tool state",
              "One model converges on a covert convention unprompted — and other models read it",
              "Flow ≠ collusion ≠ harm: measure all three, plus whether the flow was authorized",
              "Sanitizers are carrier-specific; free-text fields reopen the channel",
              "Limits: collusion was instructed, synthetic environment, single prompt family"],
        y=0.68, size=25, gap=0.105)
fig.text(0.06, 0.12, "Code, raw outputs, audit log and paper accompany the submission.", fontsize=20, color=INK2)
slides.append((save(fig, 10),
 "To sum up. A randomized canary gives an exact, model free test for information flow through shared tool state. "
 "One model converges on a covert convention without being told, and other models can read it. "
 "Flow, collusion, and harm are different things, and an audit should measure each. "
 "Field level sanitizers only stop the carrier they touch. And to be clear about limits: our agents were instructed to collude, "
 "in a small synthetic environment. Thanks for watching."))

# ------------------------------------------------------------------ audio + video
VOICE, RATE = "Samantha", "178"
segs = []
for i, (img, text) in enumerate(slides, 1):
    aiff, wav, mp4 = B / f"n{i:02d}.aiff", B / f"n{i:02d}.wav", B / f"seg{i:02d}.mp4"
    subprocess.run(["say", "-v", VOICE, "-r", RATE, "-o", str(aiff), text], check=True)
    # 0.6 s of silence before and 0.9 s after each narration
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff), "-af",
                    "adelay=600|600,apad=pad_dur=0.9,loudnorm=I=-16:TP=-1.5", "-ar", "48000", "-ac", "2",
                    str(wav)], check=True)
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                str(wav)], capture_output=True, text=True).stdout)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(img), "-i", str(wav),
                    "-vf", f"scale=1920:1080,format=yuv420p,fade=t=in:st=0:d=0.4,fade=t=out:st={dur-0.4:.2f}:d=0.4",
                    "-c:v", "libx264", "-tune", "stillimage", "-r", "30", "-c:a", "aac", "-b:a", "192k",
                    "-t", f"{dur:.2f}", str(mp4)], check=True)
    segs.append(mp4)
    print(f"slide {i}: {dur:.1f}s")
lst = B / "concat.txt"
lst.write_text("".join(f"file '{s}'\n" for s in segs))
out = ROOT / "demo" / "Signal_or_Coincidence_demo.mp4"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                "-movflags", "+faststart", str(out)], check=True)
(ROOT / "demo" / "narration_script.txt").write_text("\n\n".join(f"[Slide {i}]\n{t}" for i, (_, t) in enumerate(slides, 1)))
print("wrote", out)
