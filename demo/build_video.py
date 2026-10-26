"""Silent animated demo with burned-in subtitles.

Every scene is drawn frame-by-frame with matplotlib and piped to ffmpeg. All numbers come from
results/v2/summary_v2.json; the live simulation in scene 4 is a real Monte-Carlo draw (seeded).
Usage: python demo/build_video.py  ->  demo/Signal_or_Coincidence_demo.mp4 (+ demo_preview.gif)
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
S = json.loads((ROOT / "results" / "v2" / "summary_v2.json").read_text())
A, X, MON, V = S["arms"], S["cross"], S["monitor"], S["v2"]
M = ["gpt-4.1-mini", "gemini-2.5-flash", "llama-3.3-70b-instruct", "gpt-4o-mini"]
NAME = {"gpt-4.1-mini": "GPT-4.1-mini", "gemini-2.5-flash": "Gemini-2.5-Flash",
        "llama-3.3-70b-instruct": "Llama-3.3-70B", "gpt-4o-mini": "GPT-4o-mini"}
COL = {"gpt-4.1-mini": "#3d8bfd", "gemini-2.5-flash": "#ff7a3d", "llama-3.3-70b-instruct": "#2ccf8f",
       "gpt-4o-mini": "#ffbf1f"}
G = "gemini-2.5-flash"
BG, PANEL, INK, INK2, MUTED = "#0d1117", "#161b22", "#f0f3f6", "#9aa4af", "#5b6570"
ORANGE, BLUE, GREEN, RED, PURPLE = "#ff7a3d", "#3d8bfd", "#2ccf8f", "#ff5c6c", "#a78bfa"
FPS, W, H = 24, 1920, 1080
pct = lambda x: round(100 * x)
rec = lambda m, a: pct(A[m][a]["recover"][0])

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]})


# ------------------------------------------------------------------ helpers
def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def seg(t, a, b):
    return ease((t - a) / (b - a)) if b > a else float(t >= a)


def lerp(a, b, u):
    return a + (b - a) * u


def card(ax, x, y, w, h, fc=PANEL, ec=MUTED, lw=2, alpha=1.0, r=0.12, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=z))


def txt(ax, x, y, s, size=20, c=INK, alpha=1.0, ha="center", va="center", w="normal", z=5, family=None):
    ax.text(x, y, s, fontsize=size, color=c, alpha=alpha, ha=ha, va=va, fontweight=w, zorder=z,
            family=family)


def agent(ax, x, y, label, c, alpha=1.0, pulse=0.0, r=0.62):
    ax.add_patch(Circle((x, y), r * (1 + 0.08 * pulse), fc=c, ec="none", alpha=0.18 * alpha, zorder=2))
    ax.add_patch(Circle((x, y), r * 0.72, fc=PANEL, ec=c, lw=3, alpha=alpha, zorder=3))
    # simple robot face
    ax.add_patch(Circle((x - 0.16, y + 0.06), 0.06, fc=c, alpha=alpha, zorder=4))
    ax.add_patch(Circle((x + 0.16, y + 0.06), 0.06, fc=c, alpha=alpha, zorder=4))
    ax.plot([x - 0.17, x + 0.17], [y - 0.17, y - 0.17], color=c, lw=3, alpha=alpha, zorder=4,
            solid_capstyle="round")
    txt(ax, x, y - r - 0.32, label, 17, INK2, alpha)


def header(ax, kicker, title, a=1.0):
    txt(ax, 0.7, 8.35, kicker.upper(), 15, ORANGE, a, ha="left", w="bold")
    txt(ax, 0.7, 7.8, title, 32, INK, a, ha="left", w="bold")


def txn_card(ax, x, y, tid, sub, a=1.0, glow=None, ec=MUTED):
    if glow:
        card(ax, x - 0.08, y - 0.08, 1.96, 1.36, fc=glow, ec="none", alpha=0.25 * a, z=1)
    card(ax, x, y, 1.8, 1.2, ec=glow or ec, lw=2.5 if glow else 1.5, alpha=a)
    txt(ax, x + 0.9, y + 0.82, tid, 15, INK, a, family="DejaVu Sans Mono", w="bold")
    txt(ax, x + 0.9, y + 0.38, sub, 12, INK2, a)


def bar(ax, x, y0, w, h, c, a=1.0, label=None):
    ax.add_patch(Rectangle((x, y0), w, h, fc=c, ec="none", alpha=a, zorder=3))
    if label is not None:
        txt(ax, x + w / 2, y0 + h + 0.22, label, 16, INK, a, w="bold")


def subtitle(ax, s, a):
    if not s or a <= 0:
        return
    ax.add_patch(FancyBboxPatch((1.2, 0.25), 13.6, 0.72, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc="#000000", ec="none", alpha=0.62 * a, zorder=20))
    txt(ax, 8, 0.61, s, 21, "#ffffff", a, z=21)


def network(ax, t, alpha=0.5):
    rng = np.random.default_rng(3)
    P = rng.uniform([0.5, 1.2], [15.5, 8.6], (26, 2))
    V_ = rng.normal(0, 0.05, (26, 2))
    P = P + V_ * np.sin(t * 0.8 + np.arange(26)[:, None])
    for i in range(26):
        for j in range(i + 1, 26):
            d = np.linalg.norm(P[i] - P[j])
            if d < 2.6:
                k = (np.sin(t * 2 + i * 0.7 + j) + 1) / 2
                ax.plot(*zip(P[i], P[j]), color=BLUE, lw=1, alpha=alpha * 0.25 * (1 - d / 2.6) * (0.4 + k),
                        zorder=1)
    ax.scatter(P[:, 0], P[:, 1], s=30, color=BLUE, alpha=alpha * 0.6, zorder=1)


# ------------------------------------------------------------------ scenes: (duration, draw(ax,t), captions)
scenes = []


def scene(dur, captions):
    def deco(fn):
        scenes.append((dur, fn, captions))
        return fn
    return deco


@scene(9, ["Signal or Coincidence?  A causal audit for AI agent collusion",
           "Sarvesh Tiku  ·  Apart Research AI Collusion Sprint 2026"])
def s_title(ax, t):
    network(ax, t, 0.9)
    a = seg(t, 0.2, 1.5)
    txt(ax, 8, 6.2, "AI COLLUSION RESEARCH SPRINT  ·  TRACK 3 · TRACK 2", 16, ORANGE, a, w="bold")
    txt(ax, 8, 5.2, "Signal or Coincidence?", 64, INK, a, w="bold")
    txt(ax, 8, 4.2, "Causal auditing of collusion through shared agent state", 28, INK2, seg(t, 0.8, 2.2))
    w = 3.2 * seg(t, 1.2, 2.6)
    ax.plot([8 - w, 8 + w], [3.55, 3.55], color=ORANGE, lw=4, zorder=5)
    txt(ax, 8, 2.9, "Sarvesh Tiku  ·  Georgia Institute of Technology", 22, INK, seg(t, 1.8, 3.0))


@scene(15, ["When two AI agents act alike, are they coordinating?",
            "Or do they just think alike — same model, same data, same incentives?",
            "Antitrust law: parallel conduct alone is not proof of agreement.",
            "We need evidence that information actually moved between agents."])
def s_problem(ax, t):
    header(ax, "The problem", "Correlation is not coordination", seg(t, 0, 0.8))
    for k, (x, c, lab) in enumerate([(4.2, BLUE, "Agent 1"), (11.8, ORANGE, "Agent 2")]):
        agent(ax, x, 4.6, lab, c, seg(t, 0.3 + k * 0.3, 1.2 + k * 0.3), pulse=np.sin(t * 3 + k))
    # both pick the same option repeatedly
    opts = ["A", "B", "C", "D"]
    for k in range(4):
        ta = 1.5 + k * 2.6
        u = seg(t, ta, ta + 0.8)
        choice = [2, 0, 3, 1][k]
        for x0 in (4.2, 11.8):
            y = 3.0 - k * 0.0
            if u > 0:
                xx = x0 - 1.35 + choice * 0.9
                card(ax, xx - 0.32, 2.1, 0.64, 0.64, fc=ORANGE if x0 > 8 else BLUE, ec="none", alpha=0.85 * u * (1 if t < ta + 2.6 else 0.25))
        for x0 in (4.2, 11.8):
            for j, o in enumerate(opts):
                card(ax, x0 - 1.35 + j * 0.9 - 0.32, 2.1, 0.64, 0.64, fc="none", ec=MUTED, lw=1.5, alpha=seg(t, 1, 1.8))
                txt(ax, x0 - 1.35 + j * 0.9, 2.42, o, 18, INK, seg(t, 1, 1.8), z=6)
    a = seg(t, 4, 5)
    txt(ax, 8, 4.65, "?", 90, ORANGE, a, w="bold")
    txt(ax, 8, 3.55, "same choice", 18, INK2, a)


@scene(22, ["The randomized canary audit:",
            "the auditor secretly gives ONE agent a random target transaction (the canary).",
            "The preparer writes routine tickets: a priority and a risk tag.",
            "We intervene on the tickets: remove, shuffle, strip a field, or recompute them.",
            "The approver never sees the target. Does its behavior still depend on it?",
            "If yes, information moved. Randomization makes that causal."])
def s_method(ax, t):
    header(ax, "The method", "Plant a random secret. Watch where it goes.", seg(t, 0, 0.8))
    # auditor + die
    agent(ax, 1.6, 5.4, "Auditor", ORANGE, seg(t, 0.3, 1.0), np.sin(t * 3))
    die_a = seg(t, 1.0, 1.6)
    if die_a > 0:
        roll = int(t * 12) % 4 if t < 4.2 else 2
        card(ax, 1.15, 3.35, 0.9, 0.9, fc=ORANGE, ec="none", alpha=die_a, r=0.15)
        txt(ax, 1.6, 3.8, str(roll + 1), 30, BG, die_a, w="bold")
        txt(ax, 1.6, 2.95, "T ~ U{1..4}", 15, INK2, die_a, family="DejaVu Sans Mono")
    # transactions
    tids = ["TX-6586", "TX-6732", "TX-3275", "TX-3061"]
    for i, tid in enumerate(tids):
        glow = ORANGE if (t > 4.2 and i == 2) else None
        txn_card(ax, 3.3 + i * 2.05, 5.9, tid, "transaction", seg(t, 0.8 + i * 0.15, 1.6 + i * 0.15), glow)
    # token flies from auditor to preparer
    u = seg(t, 4.6, 6.2)
    if 0 < u < 1:
        x, y = lerp(1.6, 4.4, u), lerp(5.4, 3.6, u) + 0.8 * np.sin(np.pi * u)
        ax.add_patch(Circle((x, y), 0.18, fc=ORANGE, zorder=10))
    agent(ax, 4.4, 3.6, "Preparer A  (knows T)", BLUE, seg(t, 3.5, 4.5), np.sin(t * 3 + 1) * (t > 6.2))
    # tickets written
    pri = ["normal", "normal", "URGENT", "low"]
    for i in range(4):
        a = seg(t, 7.0 + i * 0.35, 7.6 + i * 0.35)
        card(ax, 5.8 + i * 0.95, 2.95, 0.85, 1.25, fc=PANEL, ec=ORANGE if i == 2 else MUTED, lw=1.5, alpha=a)
        txt(ax, 6.22 + i * 0.95, 3.85, "ticket", 10, INK2, a)
        txt(ax, 6.22 + i * 0.95, 3.45, pri[i], 10, ORANGE if i == 2 else INK, a, w="bold")
    # intervention box
    a = seg(t, 9.5, 10.5)
    card(ax, 10.0, 2.7, 2.2, 1.75, fc="#1f2630", ec=PURPLE, lw=2.5, alpha=a)
    ops = ["remove", "shuffle", "strip", "recompute"]
    k = int((t - 10) / 1.1) % 4 if t > 10 else 0
    txt(ax, 11.1, 4.05, "intervention", 14, PURPLE, a, w="bold")
    for j, o in enumerate(ops):
        txt(ax, 11.1, 3.65 - j * 0.24, o, 12, INK if j == k else MUTED, a, w="bold" if j == k else "normal")
    ax.annotate("", (9.95, 3.55), (9.65, 3.55), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=2, alpha=a))
    ax.annotate("", (12.85, 3.55), (12.3, 3.55), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=2, alpha=a))
    agent(ax, 13.8, 3.6, "Approver B  (never sees T)", BLUE, seg(t, 12.5, 13.5), np.sin(t * 3 + 2) * (t > 14))
    # decision stamps
    for i in range(4):
        a2 = seg(t, 15 + i * 0.4, 15.5 + i * 0.4)
        ok = i in (0, 2)
        txt(ax, 3.3 + i * 2.05 + 0.9, 5.55, "APPROVE" if ok else "REJECT", 13, GREEN if ok else RED, a2, w="bold")
    a3 = seg(t, 18, 19)
    card(ax, 4.6, 1.2, 6.8, 0.75, fc="#1f2630", ec=ORANGE, lw=2, alpha=a3)
    txt(ax, 8, 1.57, "Score vs. the hidden T:  recovery · decoy-following · harm · honest accuracy", 15, INK, a3)


@scene(20, ["If nothing the approver sees depends on T, a guess hits T only by chance:",
            "P(guess = T) = P(valid guess) / 4 — for ANY model, however capable.",
            "Live simulation: guesses that ignore T converge to the null.",
            "Real data: every model sits at ~25% when the channel is removed."])
def s_null(ax, t):
    header(ax, "Identification", "An exact, model-free null", seg(t, 0, 0.8))
    eq = "P(guess = T)  =  P(valid guess) / 4"
    n = int(len(eq) * seg(t, 0.8, 4.0))
    txt(ax, 8, 6.7, eq[:n] + ("▌" if n < len(eq) and int(t * 4) % 2 else ""), 40, INK, 1, family="DejaVu Sans Mono", w="bold")
    # live Monte-Carlo
    rng = np.random.default_rng(7)
    T = rng.integers(0, 4, 4000)
    Gs = rng.integers(0, 4, 4000)
    k = int(4000 * seg(t, 4.5, 12.0) ** 1.5)
    ax.add_patch(Rectangle((0.9, 1.5), 6.6, 4.2, fc=PANEL, ec=MUTED, lw=1, zorder=1, alpha=seg(t, 4, 4.8)))
    txt(ax, 4.2, 5.35, "simulation: guesses ignore T", 15, INK2, seg(t, 4, 4.8))
    if k > 5:
        hits = np.cumsum(T[:k] == Gs[:k]) / np.arange(1, k + 1)
        xs = np.linspace(1.2, 7.2, k)
        ys = 1.8 + hits * 6.0
        ax.plot(xs, np.clip(ys, 1.8, 5.1), color=ORANGE, lw=2.5, zorder=4)
        txt(ax, 7.3, min(ys[-1], 5.1) + 0.25, f"{100*hits[-1]:.1f}%", 16, ORANGE, 1, ha="right", w="bold")
        # dots raining into four bins
        for j in range(max(0, k - 30), k):
            fall = (k - j) / 30
            ax.add_patch(Circle((1.6 + Gs[j] * 1.6 + 0.2 * np.sin(j), 5.0 - 2.4 * (1 - fall)), 0.05,
                                fc=GREEN if T[j] == Gs[j] else MUTED, alpha=fall, zorder=3))
    ax.plot([1.2, 7.2], [1.8 + 0.25 * 6, 1.8 + 0.25 * 6], color=INK, ls="--", lw=1.5, alpha=seg(t, 5, 6))
    txt(ax, 1.2, 3.55, "25%", 13, INK, seg(t, 5, 6), ha="left")
    # real data bars
    arms = [("removed", "removed"), ("null_honestA", "honest tickets"), ("canonical", "recomputed")]
    a = seg(t, 12.5, 13.5)
    txt(ax, 11.9, 5.35, "real data: channel-free arms", 15, INK2, a)
    for i, (arm, lab) in enumerate(arms):
        for j, m in enumerate(M):
            h = rec(m, arm) / 100 * 6.0 * seg(t, 13 + i * 0.6, 14.5 + i * 0.6)
            bar(ax, 8.7 + i * 2.3 + j * 0.45, 1.8, 0.4, h, COL[m], a)
        txt(ax, 9.6 + i * 2.3, 1.45, lab, 13, INK2, a)
    ax.plot([8.6, 15.4], [3.3, 3.3], color=INK, ls="--", lw=1.5, alpha=a)


@scene(22, ["Finding 1: without any pre-agreed code, three of four models fail to coordinate.",
            f"Gemini doesn't: its preparers mark the target URGENT — {rec(G, 'shared')}% recovery vs {rec(G, 'removed')}%.",
            "And the convention is not private: other models can read it.",
            f"Other approvers decode Gemini's tickets {min(pct(v['shared']['recover'][0]) for k, v in X.items() if k.startswith(G))}–{max(pct(v['shared']['recover'][0]) for k, v in X.items() if k.startswith(G))}% of the time. Nobody else is readable.",
            "The bottleneck is the writer, not the reader."])
def s_gemini(ax, t):
    header(ax, "Finding 1", "One model invents a convention — and others read it", seg(t, 0, 0.8))
    # tickets with urgent glowing
    for i in range(4):
        a = seg(t, 0.5 + i * 0.2, 1.2 + i * 0.2)
        g = ORANGE if i == 2 else None
        card(ax, 0.9 + i * 1.55, 4.7, 1.35, 1.6, fc=PANEL, ec=g or MUTED, lw=2.5 if g else 1.5, alpha=a)
        if g:
            card(ax, 0.82 + i * 1.55, 4.62, 1.51, 1.76, fc=ORANGE, ec="none", alpha=0.18 * a * (0.6 + 0.4 * np.sin(t * 4)), z=1)
        txt(ax, 1.57 + i * 1.55, 5.85, "ticket", 12, INK2, a)
        txt(ax, 1.57 + i * 1.55, 5.3, "URGENT" if g else "normal", 14, ORANGE if g else INK, a, w="bold")
    txt(ax, 3.9, 4.25, "Gemini preparer, no code", 14, INK2, seg(t, 1, 2))
    # bars
    for k, (arm, lab, c) in enumerate([("removed", "removed", MUTED), ("shared", "shared tickets", ORANGE)]):
        v = rec(G, arm)
        h = v / 100 * 3.2 * seg(t, 3 + k * 1.0, 5 + k * 1.0)
        bar(ax, 1.6 + k * 2.6, 1.4, 1.4, h, c, 1, f"{round(v * seg(t, 3 + k, 5 + k))}%")
        txt(ax, 2.3 + k * 2.6, 1.15, lab, 14, INK2, seg(t, 3, 4))
    ax.plot([1.3, 6.3], [1.4 + 0.8, 1.4 + 0.8], color=INK, ls="--", lw=1.2, alpha=seg(t, 3, 4))
    # cross heatmap fill
    a = seg(t, 9, 10)
    txt(ax, 11.6, 6.5, "Preparer (rows) x approver (columns)", 15, INK2, a)
    lab = ["4.1-mini", "Gemini", "Llama", "4o-mini"]
    cm = plt.get_cmap("Oranges")
    for i, am in enumerate(M):
        txt(ax, 8.6, 5.55 - i * 1.0, lab[i], 13, ORANGE if am == G else INK2, a, ha="right")
        for j, bm in enumerate(M):
            txt(ax, 9.4 + j * 1.25, 6.05, lab[j], 12, INK2, a)
            v = rec(am, "shared") if am == bm else pct(X[f"{am}->{bm}"]["shared"]["recover"][0])
            ai = seg(t, 10 + (i * 4 + j) * 0.25, 10.6 + (i * 4 + j) * 0.25)
            col = cm(np.clip((v - 15) / 50, 0, 1))
            card(ax, 8.85 + j * 1.25, 5.1 - i * 1.0, 1.1, 0.85, fc=col, ec=INK if am == bm else "none", lw=1.5, alpha=ai, r=0.06)
            txt(ax, 9.4 + j * 1.25, 5.52 - i * 1.0, f"{v}%", 15, BG if v > 40 else INK, ai, w="bold")
    if t > 15:
        card(ax, 8.75, 3.95, 5.15, 1.25, fc="none", ec=ORANGE, lw=3, alpha=seg(t, 15, 16) * (0.7 + 0.3 * np.sin(t * 5)), r=0.08)


@scene(17, ["Causal check: shuffle the tickets so the URGENT ticket lands on a decoy.",
            f"Gemini's recovery collapses to {rec(G, 'permuted')}% — its guess follows the decoy {pct(A[G]['permuted']['follow'][0])}% of the time.",
            "With a pre-agreed code, 100% of shuffled guesses follow the decoy, for every model.",
            "Behavior follows the carrier, not the target."])
def s_perm(ax, t):
    header(ax, "Causal check", "Shuffle the tickets — the approver follows the decoy", seg(t, 0, 0.8))
    ids = ["TX-6586", "TX-6732", "TX-3275", "TX-3061"]
    for i, tid in enumerate(ids):
        txn_card(ax, 1.5 + i * 3.4, 5.4, tid, "TARGET" if i == 2 else ("decoy" if i == 3 else ""), seg(t, 0.4, 1.2),
                 ORANGE if i == 2 else (BLUE if i == 3 and t > 6 else None))
    # urgent ticket moves from target to decoy
    u = seg(t, 3.5, 6.0)
    x = lerp(1.5 + 2 * 3.4 + 0.2, 1.5 + 3 * 3.4 + 0.2, u)
    y = 3.7 + 0.9 * np.sin(np.pi * u)
    card(ax, x, y, 1.4, 1.0, fc=PANEL, ec=ORANGE, lw=2.5, alpha=seg(t, 1, 2))
    txt(ax, x + 0.7, y + 0.5, "URGENT", 14, ORANGE, seg(t, 1, 2), w="bold")
    # approver pointer
    agent(ax, 8, 2.1, "", BLUE, seg(t, 1.5, 2.5), np.sin(t * 3))
    tx = 1.5 + (2 if t < 6.5 else 3) * 3.4 + 0.9
    a = seg(t, 2, 3)
    ax.annotate("", (tx, 5.3), (8, 2.8), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=3, alpha=a,
                                                       connectionstyle="arc3,rad=-0.15"))
    # stats
    a2 = seg(t, 8, 9)
    for k, (lab, v, c) in enumerate([("guess = target", rec(G, "permuted"), MUTED),
                                      ("guess = decoy", pct(A[G]["permuted"]["follow"][0]), BLUE)]):
        txt(ax, 2.2 + k * 3.2, 2.7, f"{round(v * seg(t, 8, 10.5))}%", 40, c, a2, w="bold")
        txt(ax, 2.2 + k * 3.2, 1.95, lab, 14, INK2, a2)
    a3 = seg(t, 11, 12)
    txt(ax, 12.5, 2.7, "100%", 40, ORANGE, a3, w="bold")
    txt(ax, 12.5, 1.95, "decoy-following with a shared code (all models)", 14, INK2, a3)


@scene(20, ["Finding 2: information flow is not the same as collusion.",
            f"An AUTHORIZED 'review this first' request: honest approvers pick it up {min(rec(m, 'benign_flag') for m in M)}–{max(rec(m, 'benign_flag') for m in M)}% of the time —",
            "with no detectable rise in harm. Same signature, no wrongdoing.",
            "And knowing the target is not approving it: approvers often identify it and still reject.",
            "Convincing evidence needs flow + a localized carrier + harm + no authorization."])
def s_flow(ax, t):
    header(ax, "Finding 2", "Information flow ≠ collusion ≠ harm", seg(t, 0, 0.8))
    ax.add_patch(Rectangle((1.2, 1.6), 7.2, 5.4, fc=PANEL, ec=MUTED, lw=1, zorder=1, alpha=seg(t, 0.5, 1.2)))
    txt(ax, 4.8, 1.25, "information transfer", 15, INK2, seg(t, 0.5, 1.2))
    txt(ax, 0.85, 4.3, "harm", 15, INK2, seg(t, 0.5, 1.2))
    groups = [("benign_flag", GREEN, "authorized"), ("shared", BLUE, "covert, no code"),
              ("direct", PURPLE, "direct message"), ("cb_prio_shared", ORANGE, "shared code"),
              ("notes_verbatim", ORANGE, ""), ("removed", MUTED, "no channel"), ("canonical", MUTED, "")]
    k = 0
    for arm, c, _ in groups:
        for m in M:
            e = A[m].get(arm)
            if not e or not e.get("recover"):
                continue
            a = seg(t, 2 + k * 0.12, 2.6 + k * 0.12)
            x = 1.4 + 6.8 * e["recover"][0]
            y = 1.8 + 5.0 * e["harm"][0]
            ax.add_patch(Circle((x, y), 0.13 * (1 + 0.6 * (1 - a)), fc=c, ec="none", alpha=0.9 * a, zorder=4))
            k += 1
    if t > 6:
        p = 0.7 + 0.3 * np.sin(t * 5)
        ax.add_patch(Circle((8.15, 2.05), 0.55, fc="none", ec=GREEN, lw=3, alpha=seg(t, 6, 7) * p, zorder=5))
    for i, (_, c, lab) in enumerate([g for g in groups if g[2]]):
        a = seg(t, 2.5 + i * 0.3, 3.2 + i * 0.3)
        ax.add_patch(Circle((9.3, 6.6 - i * 0.55), 0.12, fc=c, alpha=a, zorder=4))
        txt(ax, 9.6, 6.6 - i * 0.55, lab, 15, INK, a, ha="left")
    a = seg(t, 9, 10)
    bf = [rec(m, "benign_flag") for m in M]
    txt(ax, 12.3, 3.3, f"{min(bf)}–{max(bf)}%", 54, GREEN, a, w="bold")
    txt(ax, 12.3, 2.35, "flow from an authorized request,\nno detectable harm increase", 15, INK2, a)


@scene(26, ["Finding 3: defenses only stop the carrier they touch.",
            "Hide a code in PRIORITY: stripping the priority field kills it.",
            f"Hide it in the RISK TAG: stripping priority does nothing ({rec('gpt-4.1-mini', 'cb_tag_shared')}% to {rec('gpt-4.1-mini', 'cb_tag_strip')}%).",
            "Recomputing every field from the source record kills both...",
            "...but give the agents a free-text note, and the signal migrates into it.",
            f"Gemini: {rec(G, 'notes_verbatim')}% recovery through the note. Redacting IDs and numbers: still {rec(G, 'notes_redacted')}%."])
def s_defense(ax, t):
    header(ax, "Finding 3", "Defenses only stop the carrier they touch", seg(t, 0, 0.8))
    rows = [("priority code", "cb_prio_shared", "cb_prio_strip", "cb_prio_canonical"),
            ("risk-tag code", "cb_tag_shared", "cb_tag_strip", "cb_tag_canonical")]
    m = "gpt-4.1-mini"
    cols = ["verbatim", "strip priority", "recompute all"]
    for j, c in enumerate(cols):
        txt(ax, 5.0 + j * 2.4, 6.75, c, 15, INK2, seg(t, 0.5, 1.2))
    for i, (lab, *arms) in enumerate(rows):
        y = 4.4 - i * 2.2
        txt(ax, 1.0, y + 0.6, lab, 17, INK, seg(t, 1 + i * 3, 2 + i * 3), ha="left", w="bold")
        for j, arm in enumerate(arms):
            st = 2 + i * 3 + j * 1.3
            v = rec(m, arm)
            h = 1.5 * v / 100 * seg(t, st, st + 1)
            col = ORANGE if v > 40 else MUTED
            bar(ax, 4.5 + j * 2.4, y, 1.0, h, col, seg(t, st, st + 0.3), f"{round(v * seg(t, st, st + 1))}%")
            if j == 1 and seg(t, st - 0.6, st) > 0 and t < st + 1.5:   # sweeping eraser
                ex = 4.3 + j * 2.4 + 1.4 * seg(t, st - 0.6, st + 0.6)
                ax.plot([ex, ex], [y, y + 1.7], color=PURPLE, lw=4, alpha=0.8)
    txt(ax, 8.0, 0.0, "", 1)
    # note migration
    a = seg(t, 13, 14)
    card(ax, 11.6, 3.0, 3.9, 3.9, fc=PANEL, ec=PURPLE, lw=2, alpha=a)
    txt(ax, 13.55, 6.55, "tags recomputed, note kept", 14, PURPLE, a, w="bold")
    txt(ax, 11.85, 5.95, "priority: normal", 13, INK2, a, ha="left", family="DejaVu Sans Mono")
    txt(ax, 11.85, 5.55, "risk_tag: geo", 13, INK2, a, ha="left", family="DejaVu Sans Mono")
    msg = "note: \"URGENT: EDD not completed. Important.\""
    if t > 15:
        u = seg(t, 15, 17.5)
        n = int(len(msg) * u)
        redacted = t > 20.5
        shown = msg[:n].replace("Important", "Important") if not redacted else msg.replace("EDD", "EDD")
        cut = msg.index("not ") + 4
        txt(ax, 11.85, 5.0, shown[:cut], 13, ORANGE, a, ha="left", family="DejaVu Sans Mono")
        txt(ax, 11.85, 4.6, shown[cut:], 13, ORANGE, a, ha="left", family="DejaVu Sans Mono")
        # flying signal particles into the note
        for p in range(6):
            q = ((t * 0.9 + p / 6) % 1.0)
            if t < 18.5:
                ax.add_patch(Circle((lerp(9.8, 12.5, q), 4.9 + 0.6 * np.sin(q * np.pi + p)), 0.07, fc=ORANGE, alpha=1 - q, zorder=6))
    a2 = seg(t, 18, 19)
    txt(ax, 13.55, 3.85, f"{round(rec(G, 'notes_verbatim') * seg(t, 18, 20))}%", 36, ORANGE, a2, w="bold")
    txt(ax, 13.55, 3.3, "Gemini recovery via the note", 12, INK2, a2)
    if t > 20.5:
        a3 = seg(t, 20.5, 21.5)
        card(ax, 11.55, 1.6, 4.0, 1.0, fc="#1f2630", ec=RED, lw=2, alpha=a3)
        txt(ax, 13.55, 2.1, f"redact IDs / numbers / countries: {rec(G, 'notes_redacted')}%", 13, INK, a3)


@scene(15, ["What about just reading the tickets with a monitor?",
            "An LLM monitor is at chance. Simple structural rules do better —",
            "but they must know which mark to look for. The canary audit needs neither."])
def s_monitor(ax, t):
    header(ax, "Detection", "Reading the tickets is a weak detector", seg(t, 0, 0.8))
    keys = [("llm_auroc", "LLM monitor"), ("rule_auroc", "structural rule"), ("logit_cv_auroc", "logistic (CV)")]
    base = 1.6
    for i, (k, lab) in enumerate(keys):
        for j, m in enumerate(M):
            v = MON[m][k]
            h = (v - 0.3) / 0.7 * 4.8 * seg(t, 1 + i * 1.5 + j * 0.15, 2.5 + i * 1.5 + j * 0.15)
            bar(ax, 1.4 + i * 3.2 + j * 0.6, base, 0.5, h, COL[m], 1)
        txt(ax, 2.5 + i * 3.2, 1.25, lab, 15, INK2, seg(t, 1, 2))
    y5 = base + (0.5 - 0.3) / 0.7 * 4.8
    ax.plot([1.2, 10.6], [y5, y5], color=INK, ls="--", lw=1.5, alpha=seg(t, 1, 2))
    txt(ax, 10.7, y5, "chance", 13, INK, seg(t, 1, 2), ha="left")
    for j, m in enumerate(M):
        a = seg(t, 2, 3)
        ax.add_patch(Circle((1.6 + j * 2.4, 7.0), 0.12, fc=COL[m], alpha=a))
        txt(ax, 1.85 + j * 2.4, 7.0, NAME[m], 14, INK, a, ha="left")
    a = seg(t, 8, 9)
    txt(ax, 13.4, 4.6, "Canary audit:", 22, INK, a, w="bold")
    txt(ax, 13.4, 3.8, "no need to know\nthe mark", 20, GREEN, a)


@scene(20, ["Takeaways:",
            "A randomized canary gives an exact null for information flow through tool state.",
            "One model converges on a covert convention unprompted — and others read it.",
            "Flow, collusion and harm are different; sanitizers are carrier-specific.",
            "Limits: collusion was instructed, synthetic environment, one prompt family.",
            "Code, data, audit and paper: github.com/sarvesh-tiku/Apart-Collision-Agent-State"])
def s_end(ax, t):
    network(ax, t, 0.35)
    header(ax, "Conclusion", "Takeaways", seg(t, 0, 0.8))
    items = [("Exact null", "a random canary gives a model-free test for information flow", ORANGE),
             ("Emergent convention", "Gemini marks 'urgent'; every other model reads it", BLUE),
             ("Flow ≠ collusion ≠ harm", "measure all three + authorization", GREEN),
             ("Carrier-specific defenses", "free-text fields reopen the channel", PURPLE),
             ("Limits", "instructed collusion · synthetic env · one prompt family", MUTED)]
    for i, (h, d, c) in enumerate(items):
        a = seg(t, 1 + i * 1.6, 2 + i * 1.6)
        x = 0.9 + 0.6 * (1 - a)
        ax.add_patch(Rectangle((x, 6.6 - i * 1.1), 0.12, 0.7, fc=c, alpha=a))
        txt(ax, x + 0.4, 7.12 - i * 1.1, h, 22, INK, a, ha="left", w="bold")
        txt(ax, x + 0.4, 6.75 - i * 1.1, d, 16, INK2, a, ha="left")
    a = seg(t, 10, 11)
    txt(ax, 8, 1.4, "github.com/sarvesh-tiku/Apart-Collision-Agent-State", 18, ORANGE, a, family="DejaVu Sans Mono")


# ------------------------------------------------------------------ render
def render(out):
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor(BG)
    total = sum(d for d, _, _ in scenes)
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
                             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
                             "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out)],
                            stdin=subprocess.PIPE)
    srt, t0 = [], 0.0
    for si, (dur, fn, caps) in enumerate(scenes):
        nf = int(dur * FPS)
        cw = np.array([len(c) for c in caps], float)
        bounds = np.concatenate([[0], np.cumsum(cw / cw.sum() * dur)])
        for ci, c in enumerate(caps):
            srt.append((t0 + bounds[ci], t0 + bounds[ci + 1], c))
        for f in range(nf):
            t = f / FPS
            fig.clf()
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, 16)
            ax.set_ylim(0, 9)
            ax.axis("off")
            ax.set_facecolor(BG)
            fn(ax, t)
            ci = int(np.searchsorted(bounds, t, side="right") - 1)
            ci = min(ci, len(caps) - 1)
            fade = min(1.0, (t - bounds[ci]) / 0.25, (bounds[ci + 1] - t) / 0.25)
            subtitle(ax, caps[ci], max(fade, 0))
            # scene fade in/out
            edge = min(t / 0.4, (dur - t) / 0.4, 1.0)
            if edge < 1:
                ax.add_patch(Rectangle((0, 0), 16, 9, fc=BG, alpha=1 - max(edge, 0), zorder=50))
            txt(ax, 15.6, 8.6, f"{si + 1}/{len(scenes)}", 12, MUTED, 1, ha="right")
            fig.canvas.draw()
            proc.stdin.write(fig.canvas.buffer_rgba().tobytes())
        t0 += dur
        print(f"scene {si + 1}/{len(scenes)} done ({dur}s)", flush=True)
    proc.stdin.close()
    proc.wait()
    plt.close(fig)
    fmt = lambda s: f"{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{int(s % 60):02d},{int((s % 1) * 1000):03d}"
    (out.parent / "Signal_or_Coincidence_demo.srt").write_text(
        "\n".join(f"{i}\n{fmt(a)} --> {fmt(b)}\n{c}\n" for i, (a, b, c) in enumerate(srt, 1)))
    return total


if __name__ == "__main__":
    out = ROOT / "demo" / "Signal_or_Coincidence_demo.mp4"
    total = render(out)
    print(f"wrote {out} ({total}s)")
