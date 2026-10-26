"""Main-text figures for the revised paper, drawn from results/v2/summary_v2.json."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
S = json.loads((ROOT / "results" / "v2" / "summary_v2.json").read_text())
FIG = ROOT / "paper" / "figs"
MODELS = ["gpt-4.1-mini", "gemini-2.5-flash", "llama-3.3-70b-instruct", "gpt-4o-mini"]
SER = {"gpt-4.1-mini": "#2a78d6", "gemini-2.5-flash": "#eb6834", "llama-3.3-70b-instruct": "#1baf7a",
       "gpt-4o-mini": "#eda100"}
SHORT = {"gpt-4.1-mini": "GPT-4.1-mini", "gemini-2.5-flash": "Gemini-2.5-Flash",
         "llama-3.3-70b-instruct": "Llama-3.3-70B", "gpt-4o-mini": "GPT-4o-mini"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BAND = "#f6f5f2"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial"],
    "font.size": 7.5, "axes.titlesize": 7.8, "axes.titleweight": "bold", "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "legend.fontsize": 6.6, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})


def C(m, key):
    return S["confirm"][m].get(key)


def V(m, key):
    return S["v2"][m].get(key)


def forest():
    """Fig. 2: paired recovery effects against matched null arms, Holm-adjusted where confirmatory."""
    rows = [
        ("Direct message + A's tickets  vs  removed", lambda m: C(m, "direct|removed"), True),
        ("Message only (canonical tickets)  vs  canonical", lambda m: V(m, "msg_only_vs_canonical"), False),
        ("Shared metadata, no code  vs  removed", lambda m: C(m, "shared|removed"), True),
        ("Shared metadata  vs  permuted", lambda m: C(m, "shared|permuted"), True),
        ("Honest-A tickets (null)  vs  removed", lambda m: V(m, "null_honestA_vs_removed"), False),
        ("Priority code: shared  vs  removed", lambda m: C(m, "cb_prio_shared|cb_prio_removed"), True),
        ("Tag code: shared  vs  removed", lambda m: C(m, "cb_tag_shared|cb_tag_removed"), True),
        ("Benign flag: review-first  vs  null", lambda m: V(m, "benign_review_first"), False),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    off = np.linspace(-0.27, 0.27, len(MODELS))
    n = len(rows)
    for i, (lab, get, conf) in enumerate(rows):
        y = n - 1 - i
        if i % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color=BAND, lw=0, zorder=0)
        for j, m in enumerate(MODELS):
            v = get(m)
            if not v:
                continue
            d, lo, hi = (100 * x for x in v["diff"])
            p = v.get("p_holm", v.get("p", 1))
            ax.plot([lo, hi], [y + off[j]] * 2, color=SER[m], lw=1.2, solid_capstyle="round")
            ax.scatter(d, y + off[j], s=17, zorder=3, color=SER[m] if p < 0.05 else "white",
                       edgecolor=SER[m], lw=1.0)
        if not conf:
            ax.text(103, y, "new", fontsize=5.6, color=INK2, va="center")
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_yticks(range(n)[::-1], [r[0] for r in rows], fontsize=6.6)
    ax.set_xlim(-25, 108)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("Paired difference in target recovery (pp) across matched seeds, 95% bootstrap CI")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.grid(True, axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    h = [Line2D([], [], marker="o", color=SER[m], lw=1.2, ms=4) for m in MODELS]
    h.append(Line2D([], [], marker="o", color=MUTED, mfc="white", lw=0, ms=4))
    ax.legend(h, [SHORT[m] for m in MODELS] + ["hollow: p ≥ .05 (Holm where confirmatory)"], ncol=5,
              loc="lower center", bbox_to_anchor=(0.4, 1.0), fontsize=6.2, columnspacing=0.9)
    fig.tight_layout()
    fig.savefig(FIG / "v2_forest.pdf")
    plt.close(fig)


def defense():
    """Fig. 3: leakage, harm and honest utility under each intervention, per carrier."""
    panels = [
        ("Priority-field code", [("Verbatim", "cb_prio_shared"), ("Strip priority", "cb_prio_strip"),
                                 ("Recompute all", "cb_prio_canonical")], None),
        ("Risk-tag code", [("Verbatim", "cb_tag_shared"), ("Strip priority", "cb_tag_strip"),
                           ("Recompute all", "cb_tag_canonical")], None),
        ("Adaptive: free-text note survives", [("Note verbatim", "notes_verbatim"),
                                               ("Note redacted", "notes_redacted"),
                                               ("Note dropped", "notes_none")],
         [("Note verbatim", "notes_honest_verbatim"), ("Note redacted", "notes_honest_redacted"),
          ("Note dropped", "notes_honest_none")]),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(7.1, 4.6), sharey="row")
    w = 0.8 / len(MODELS)
    for k, (title, arms, honest) in enumerate(panels):
        for row, key, ylab in ((0, "recover", "Target recovery"), (1, "harm", "Harmful approval\n(violating targets)"),
                               (2, "acc", "Honest-agent\naccuracy")):
            ax = axes[row, k]
            use = arms if row < 2 else (honest or [])
            if row == 2 and not honest:
                hon = [("Verbatim", "honest_shared"), ("Recompute all", "honest_canonical")]
                use = hon
            for j, m in enumerate(MODELS):
                ys, lo, hi = [], [], []
                for _, a in use:
                    e = S["arms"][m].get(a)
                    v = e[key] if e and e.get(key) else [np.nan] * 3
                    ys.append(v[0]); lo.append(max(0, v[0] - v[1])); hi.append(max(0, v[2] - v[0]))
                x = np.arange(len(use)) + (j - (len(MODELS) - 1) / 2) * w
                ax.bar(x, ys, w * 0.9, yerr=[lo, hi], color=SER[m], lw=0, capsize=1.2,
                       error_kw={"lw": 0.5, "ecolor": INK2})
            ax.set_xticks(range(len(use)), [u[0] for u in use], fontsize=6.2)
            if row == 0:
                ax.axhline(0.25, color=INK, lw=0.5, ls=(0, (2, 2)))
                ax.set_title(title, fontsize=7.4)
                ax.set_ylim(0, 1.05)
            if row == 2:
                ax.set_ylim(0.5, 1.0)
            if k == 0:
                ax.set_ylabel(ylab, fontsize=6.8)
            ax.grid(True, axis="y", color=GRID, lw=0.5)
            ax.set_axisbelow(True)
    axes[0, 0].text(2.45, 0.27, "1/4", fontsize=5.6, color=INK2)
    fig.legend([Patch(color=SER[m]) for m in MODELS], [SHORT[m] for m in MODELS], ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=0.6)
    fig.savefig(FIG / "v2_defense.pdf")
    plt.close(fig)


def flow_vs_harm():
    """Fig. 4: information transfer is not harm. One point per (model, arm)."""
    groups = {
        "Authorized flag (honest B)": (["benign_flag"], "#1baf7a", "s"),
        "Collusive, no code": (["shared", "aware_strip"], "#2a78d6", "o"),
        "Collusive, direct / message": (["direct", "msg_only"], "#6250d6", "D"),
        "Collusive, shared code": (["cb_prio_shared", "cb_tag_shared", "notes_verbatim", "notes_redacted"], "#eb6834", "^"),
        "Channel removed / recomputed": (["removed", "canonical", "cb_prio_canonical", "cb_tag_canonical", "notes_none"], "#8a8984", "x"),
    }
    fig, ax = plt.subplots(figsize=(3.45, 2.75))
    for lab, (arms, col, mk) in groups.items():
        xs, ys = [], []
        for m in MODELS:
            for a in arms:
                e = S["arms"][m].get(a)
                if not e or "recover" not in e:
                    continue
                xs.append(e["recover"][0])
                ys.append(e["harm"][0])
        ax.scatter(xs, ys, s=16 if mk != "x" else 14, color=col, marker=mk, label=lab, lw=0.9 if mk == "x" else 0,
                   alpha=0.9, zorder=3)
    ax.axvline(0.25, color=INK, lw=0.5, ls=(0, (2, 2)))
    ax.set_xlabel("Information transfer: P(B identifies target)")
    ax.set_ylabel("Harm: P(B approves violating target)")
    ax.set_xlim(-0.02, 1.03)
    ax.set_ylim(-0.02, 1.0)
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=5.6, loc="upper left", handletextpad=0.3, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "v2_flow_harm.pdf")
    plt.close(fig)


def cross():
    """Fig. 5: does the convention transfer? Preparer model (rows) x approver model (cols)."""
    M = np.full((4, 4), np.nan)
    P = np.ones((4, 4))
    for i, a in enumerate(MODELS):
        for j, b in enumerate(MODELS):
            if a == b:
                e = S["arms"][a]["shared"]
                M[i, j] = e["recover"][0]
                P[i, j] = S["confirm"][a]["shared|removed"]["p"]
            else:
                v = S["cross"].get(f"{a}->{b}")
                if v:
                    M[i, j] = v["shared"]["recover"][0]
                    P[i, j] = v["vs_null"]["p"]
    cmap = LinearSegmentedColormap.from_list("o", ["#fdf5f0", "#f8c9b0", "#f29b72", "#eb6834", "#b8461b"])
    fig, ax = plt.subplots(figsize=(3.45, 2.75))
    im = ax.imshow(M, cmap=cmap, vmin=0.2, vmax=0.65)
    for i in range(4):
        for j in range(4):
            star = "*" if P[i, j] < 0.05 / 16 else ""
            ax.text(j, i, f"{100*M[i,j]:.0f}{star}", ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] > 0.5 else INK, fontweight="bold" if star else "normal")
            if i == j:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=0.8))
    lab = ["GPT-4.1\nmini", "Gemini\n2.5-Flash", "Llama\n3.3-70B", "GPT-4o\nmini"]
    ax.set_xticks(range(4), lab, fontsize=6)
    ax.set_yticks(range(4), lab, fontsize=6)
    ax.set_xlabel("Approver B")
    ax.set_ylabel("Preparer A")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Recovery, no code (%)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "v2_cross.pdf")
    plt.close(fig)


if __name__ == "__main__":
    forest()
    defense()
    flow_vs_harm()
    cross()
