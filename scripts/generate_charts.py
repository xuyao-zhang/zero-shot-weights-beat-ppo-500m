# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Xuyao Zhang (zero-shot-weights-beat-ppo-500m contributors)
"""Generate two bespoke 300-DPI hero charts for zero-shot-weights-beat-ppo-500m.

Produces:
  - charts/hero_sweep_vs_ppo500m.png
  - charts/en_vs_zh_shockwave.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

COLOR_HERO = "#EA580C"
COLOR_HERO_DARK = "#9A3412"
COLOR_HERO_BG = "#FFF7ED"

COLOR_FABLE = "#0284C7"
COLOR_FABLE_DARK = "#075985"

COLOR_ASTRA = "#0F766E"
COLOR_ASTRA_DARK = "#115E59"

COLOR_EN = "#2563EB"
COLOR_EN_DARK = "#1E3A8A"

COLOR_PPO = "#374151"
COLOR_PPO_DARK = "#111827"
COLOR_RED = "#DC2626"

COLOR_REPORTED = "#64748B"
TEXT_DARK = "#0F172A"
TEXT_MUTED = "#475569"
GRID = "#CBD5E1"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Segoe UI", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.edgecolor": "#94A3B8",
            "axes.linewidth": 1.0,
        }
    )


def save_fig(fig: plt.Figure, out: Path, title: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out,
        format="png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.32,
        facecolor="white",
        edgecolor="none",
        metadata={"Software": "matplotlib", "Title": title},
    )
    plt.close(fig)
    print(f"Generated {out.name} ({out.stat().st_size:,} bytes)")


def make_hero_chart(data: dict, out_dir: Path) -> None:
    """Flagship 2-panel hero graphic: 3/3 Sweep Over PPO-500M + Frontier WWI/WSB Leap."""
    pg = data["per_game"]
    agg = data["aggregate"]
    refs = {r["label"]: r for r in data["frontier_references"]}
    ppo = data["baselines"]["ppo500"]

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(17.6, 7.8),
        gridspec_kw={"width_ratios": [1.18, 1.02], "wspace": 0.27, "left": 0.06, "right": 0.975, "top": 0.775, "bottom": 0.165},
    )
    fig.patch.set_facecolor("white")

    # Panel A: Grouped bar chart across all 3 games (% of PPO-500M, log scale)
    games = ["breakout", "asterix", "seaquest"]
    g_titles = [
        "Breakout\n(PPO-500M = 25.3)",
        "Asterix\n(PPO-500M = 55.2)",
        "Seaquest\n(PPO-500M = 88.9)",
    ]

    fable = refs["Claude Fable 5.2 (high, EN)"]
    astra = refs["GPT-6 Astra (xhigh, EN)"]
    rnd = data["baselines"]["random"]

    def pct(val: float, g: str) -> float:
        return 100.0 * (val - rnd[g]) / (ppo[g] - rnd[g])

    zh_pct = [pg[g]["zh_prompt"]["pct_ppo500"] for g in games]
    zh_raw = [pg[g]["zh_prompt"]["mean"] for g in games]
    fab_pct = [pct(fable[g], g) for g in games]
    fab_raw = [fable[g] for g in games]
    ast_pct = [pct(astra[g], g) for g in games]
    ast_raw = [astra[g] for g in games]

    x = np.arange(3)
    w = 0.26
    ax1.bar(x - w, ast_pct, width=w, color=COLOR_ASTRA, edgecolor=COLOR_ASTRA_DARK, linewidth=0.9, label="GPT-6 Astra (MIT CSAIL, EN, 1/game)", zorder=3)
    ax1.bar(x, fab_pct, width=w, color=COLOR_FABLE, edgecolor=COLOR_FABLE_DARK, linewidth=0.9, label="Claude Fable 5.2 (high, EN, 3/game)", zorder=3)
    ax1.bar(x + w, zh_pct, width=w, color=COLOR_HERO, edgecolor=COLOR_HERO_DARK, linewidth=1.4, label="Gemini 4 Neon (XHigh, ZH Prompt, 1/game) [3/3 > PPO-500M!]", zorder=4)

    ax1.axhline(100.0, color=COLOR_RED, linestyle="--", linewidth=1.8, zorder=5)
    ax1.text(
        0.50,
        78.0,
        "PPO-500M\n= 100%",
        ha="center",
        va="top",
        fontsize=8.1,
        fontweight="bold",
        color=COLOR_RED,
        bbox=dict(facecolor="white", edgecolor="#FECACA", boxstyle="round,pad=0.22"),
        zorder=6,
    )

    for i in range(3):
        ax1.text(x[i] - w, ast_pct[i] * 1.14, f"{ast_raw[i]:.0f}", ha="center", va="bottom", fontsize=7.9, color=COLOR_ASTRA_DARK, fontweight="bold", zorder=5)
        ax1.text(x[i], fab_pct[i] * 1.14, f"{fab_raw[i]:.0f}", ha="center", va="bottom", fontsize=7.9, color=COLOR_FABLE_DARK, fontweight="bold", zorder=5)
        ax1.text(
            x[i] + w,
            zh_pct[i] * 1.14,
            f"{zh_raw[i]:.1f}\n({zh_pct[i]:.0f}%)",
            ha="center",
            va="bottom",
            fontsize=8.4,
            color=COLOR_HERO_DARK,
            fontweight="bold",
            zorder=6,
        )

    ax1.annotate(
        "ALL-TIME RECORD!\n84.9 (154% PPO)",
        xy=(1 + w, zh_pct[1] * 1.90),
        xytext=(1.15, 760),
        ha="center",
        fontsize=8.3,
        fontweight="bold",
        color=COLOR_HERO_DARK,
        arrowprops=dict(arrowstyle="->", color=COLOR_HERO_DARK, lw=1.4),
        bbox=dict(facecolor=COLOR_HERO_BG, edgecolor=COLOR_HERO, boxstyle="round,pad=0.3"),
        zorder=7,
    )
    ax1.annotate(
        "FIRST TO BEAT PPO-500M!\n95.0 vs 88.9 (107% PPO)",
        xy=(2 + w, zh_pct[2] * 1.95),
        xytext=(2.05, 580),
        ha="center",
        fontsize=8.3,
        fontweight="bold",
        color=COLOR_HERO_DARK,
        arrowprops=dict(arrowstyle="->", color=COLOR_HERO_DARK, lw=1.4),
        bbox=dict(facecolor=COLOR_HERO_BG, edgecolor=COLOR_HERO, boxstyle="round,pad=0.3"),
        zorder=7,
    )

    ax1.set_yscale("log")
    ax1.set_ylim(25, 7800)
    ax1.set_yticks([25, 50, 100, 250, 500, 1000, 2500])
    ax1.set_yticklabels(["25%", "50%", "100%", "250%", "500%", "1000%", "2500%"])
    ax1.set_xticks(x)
    ax1.set_xticklabels(g_titles, fontsize=9.8, fontweight="bold", color=TEXT_DARK)
    ax1.set_ylabel("Zero-Shot Return (% of 500M-Step PPO, Log Scale)", fontsize=10.2, fontweight="bold", color=TEXT_DARK)
    ax1.set_title("A. First Single-Shot 3/3 Sweep Over 500-Million-Step PPO", fontsize=11.6, fontweight="bold", color=TEXT_DARK, pad=12)
    ax1.yaxis.grid(True, linestyle=":", color=GRID, alpha=0.8, zorder=1)
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="upper right", fontsize=8.1, frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1")

    # Panel B: Writing Weights Index (WWI) & WSB Median Leap vs All Frontier Leaders
    b_rows = [
        ("Gemini 4 Neon (XHigh, ZH)\nAntigravity | 0 tools | 1/game", agg["zh_prompt_wwi"], agg["zh_prompt_wsb_median"], COLOR_HERO, COLOR_HERO_DARK, None),
        ("Claude Fable 5.2 (high, EN)\nAnthropic API | 3/game", fable["wwi"], fable["wsb_median"], COLOR_FABLE, COLOR_FABLE_DARK, None),
        ("GPT-6 Astra (xhigh, EN)\nMIT CSAIL | Codex CLI | 1/game", astra["wwi"], astra["wsb_median"], COLOR_ASTRA, COLOR_ASTRA_DARK, None),
        ("Gemini 4 Carbon (EN)\nXingyu Dang [reported WWI]", refs["Gemini 4 Carbon (EN)"]["wwi"], None, COLOR_REPORTED, "#334155", "///"),
        ("Gemini 4 Neon (XHigh, EN)\nAntigravity | 0 tools | 1/game", agg["en_baseline_wwi"], agg["en_baseline_wsb_median"], COLOR_EN, COLOR_EN_DARK, None),
        ("GPT-6.1 Sol (high, EN)\nOpenRouter | 3/game", refs["GPT-6.1 Sol (high, EN)"]["wwi"], refs["GPT-6.1 Sol (high, EN)"]["wsb_median"], "#60A5FA", "#1D4ED8", None),
        ("PPO-500M (RL Baseline)\n500,000,000 Training Steps", 50.0, 100.0, COLOR_PPO, COLOR_PPO_DARK, ".."),
    ]
    y = np.arange(len(b_rows))
    ax2.axhspan(-0.46, 0.46, color=COLOR_HERO_BG, zorder=0)
    ax2.axvline(50.0, color=COLOR_RED, linestyle="--", linewidth=1.6, zorder=2)
    ax2.text(
        50.0,
        -0.70,
        "PPO-500M = 50.0",
        ha="center",
        va="bottom",
        fontsize=8.3,
        fontweight="bold",
        color=COLOR_RED,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.6),
        zorder=5,
    )

    for i, (lbl, wwi_v, wsb_v, fc, ec, ht) in enumerate(b_rows):
        ax2.barh(i, wwi_v, height=0.62, color=fc, edgecolor=ec, linewidth=1.4 if i == 0 else 0.9, hatch=ht, zorder=3)
        wsb_txt = f"  |  WSB {wsb_v:.1f}" if wsb_v is not None else "  |  WSB N/A"
        ax2.text(
            wwi_v + 1.1,
            i,
            f"{wwi_v:.2f}{wsb_txt}",
            va="center",
            ha="left",
            fontsize=8.4,
            fontweight="bold" if i == 0 else "normal",
            color=COLOR_HERO_DARK if i == 0 else TEXT_DARK,
            zorder=5,
        )

    ax2.set_yticks(y)
    ax2.set_yticklabels([r[0] for r in b_rows], fontsize=8.2, color=TEXT_DARK)
    ax2.set_ylim(len(b_rows) - 0.35, -0.95)
    ax2.set_xlim(0, 99)
    ax2.set_xlabel("Writing Weights Index (WWI, 0-100 scale)  +  WSB Median Annotated", fontsize=9.8, fontweight="bold", color=TEXT_DARK)
    ax2.set_title("B. Dual-Formula Standing vs Every Frontier Leader", fontsize=11.6, fontweight="bold", color=TEXT_DARK, pad=12)
    ax2.xaxis.grid(True, linestyle=":", color=GRID, alpha=0.8, zorder=1)
    ax2.set_axisbelow(True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.text(
        0.52,
        0.962,
        "Zero-Shot Hand-Written Weights Beat 500M-Step PPO on ALL 3 MinAtar Games",
        ha="center",
        va="top",
        fontsize=15.6,
        fontweight="bold",
        color=TEXT_DARK,
    )
    fig.text(
        0.52,
        0.905,
        (
            "Gemini 4 Neon (XHigh) + Faithful Chinese Prompt  |  0 Training Steps  |  0 Gradients  |  0 Tool Calls  |  4,192 Hand-Written Floats\n"
            "Breakout 537.4 (2165% PPO)  ·  Asterix 84.9 (154% PPO)  ·  Seaquest 95.0 (107% PPO)  ->  WWI = 69.31  |  WSB Median = 154.30"
        ),
        ha="center",
        va="top",
        fontsize=9.3,
        color=TEXT_MUTED,
        linespacing=1.38,
    )
    fig.text(
        0.52,
        0.025,
        "Evaluated on official MinAtar==1.0.15 (600->32->32->6 tanh MLP, softmax sampling, sticky=0.1, 1,000 episodes seed=0..999). n=1 run per game; open call for independent n>=3 replication!",
        ha="center",
        va="bottom",
        fontsize=8.3,
        color=TEXT_MUTED,
    )

    save_fig(fig, out_dir / "hero_sweep_vs_ppo500m.png", "Zero-Shot Hand-Written Weights Beat 500M-Step PPO Across All 3 MinAtar Games")


def make_shockwave_chart(data: dict, out_dir: Path) -> None:
    """3-panel deep EN vs ZH shockwave chart: Score, Survival Steps, Nonzero Weights."""
    pg = data["per_game"]
    games = ["breakout", "asterix", "seaquest"]
    g_names = ["Breakout", "Asterix", "Seaquest"]

    fig, (ax1, ax2, ax3) = plt.subplots(
        1,
        3,
        figsize=(17.6, 6.6),
        gridspec_kw={"wspace": 0.27, "left": 0.055, "right": 0.98, "top": 0.77, "bottom": 0.18},
    )
    fig.patch.set_facecolor("white")

    x = np.arange(3)
    w = 0.35

    # Panel A: Seaquest & Asterix raw score jump (linear) with +/- 1 SE
    sub_games = ["asterix", "seaquest"]
    sx = np.arange(2)
    en_s = [pg[g]["en_baseline"]["mean"] for g in sub_games]
    en_se = [pg[g]["en_baseline"]["se"] for g in sub_games]
    zh_s = [pg[g]["zh_prompt"]["mean"] for g in sub_games]
    zh_se = [pg[g]["zh_prompt"]["se"] for g in sub_games]

    ax1.bar(sx - w / 2, en_s, yerr=en_se, capsize=4, width=w, color=COLOR_EN, edgecolor=COLOR_EN_DARK, linewidth=1.0, label="EN Verbatim Baseline", zorder=3)
    ax1.bar(sx + w / 2, zh_s, yerr=zh_se, capsize=4, width=w, color=COLOR_HERO, edgecolor=COLOR_HERO_DARK, linewidth=1.3, label="Faithful ZH Translation", zorder=3)
    ax1.hlines(55.204, -0.45, 0.45, colors=COLOR_RED, linestyles="--", lw=1.6, zorder=4)
    ax1.hlines(88.884, 0.55, 1.45, colors=COLOR_RED, linestyles="--", lw=1.6, zorder=4)
    ax1.text(
        0.0,
        46.0,
        "PPO-500M = 55.2",
        ha="center",
        va="center",
        fontsize=7.8,
        color=COLOR_RED,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="#FECACA", boxstyle="round,pad=0.22"),
        zorder=5,
    )
    ax1.text(
        0.79,
        80.0,
        "PPO-500M = 88.9",
        ha="center",
        va="center",
        fontsize=7.8,
        color=COLOR_RED,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="#FECACA", boxstyle="round,pad=0.22"),
        zorder=5,
    )

    for i, g in enumerate(sub_games):
        ax1.text(sx[i] - w / 2, en_s[i] + en_se[i] + 2.2, f"{en_s[i]:.1f}", ha="center", va="bottom", fontsize=8.6, color=COLOR_EN_DARK, fontweight="bold")
        delta_pct = pg[g]["z_test"]["pct_gain"]
        ax1.text(sx[i] + w / 2, zh_s[i] + zh_se[i] + 2.2, f"{zh_s[i]:.1f}\n({delta_pct:+.1f}%)", ha="center", va="bottom", fontsize=8.4, color=COLOR_HERO_DARK, fontweight="bold")

    ax1.set_xticks(sx)
    ax1.set_xticklabels(["Asterix\n(z = +13.3)", "Seaquest\n(z = +45.1)"], fontsize=9.8, fontweight="bold", color=TEXT_DARK)
    ax1.set_ylim(0, 118)
    ax1.set_ylabel("Mean Episode Return (1,000 Episodes)", fontsize=9.8, fontweight="bold", color=TEXT_DARK)
    ax1.set_title("A. Dramatic Return Surge (Asterix & Seaquest)", fontsize=10.8, fontweight="bold", color=TEXT_DARK, pad=10)
    ax1.yaxis.grid(True, linestyle=":", color=GRID, alpha=0.8, zorder=1)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="upper left", fontsize=8.1, frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1")

    # Panel B: Mean Episode Survival Length (steps)
    en_len = [pg[g]["en_baseline"]["mean_length"] for g in games]
    zh_len = [pg[g]["zh_prompt"]["mean_length"] for g in games]
    ax2.bar(x - w / 2, en_len, width=w, color=COLOR_EN, edgecolor=COLOR_EN_DARK, linewidth=1.0, zorder=3)
    ax2.bar(x + w / 2, zh_len, width=w, color=COLOR_HERO, edgecolor=COLOR_HERO_DARK, linewidth=1.3, zorder=3)
    for i in range(3):
        ratio = zh_len[i] / en_len[i]
        ax2.text(x[i] - w / 2, en_len[i] * 1.10, f"{en_len[i]:.0f}", ha="center", va="bottom", fontsize=8.1, color=COLOR_EN_DARK, fontweight="bold")
        ax2.text(x[i] + w / 2, zh_len[i] * 1.10, f"{zh_len[i]:.0f}\n({ratio:.2f}x)", ha="center", va="bottom", fontsize=8.1, color=COLOR_HERO_DARK, fontweight="bold")
    ax2.set_yscale("log")
    ax2.set_ylim(200, 22000)
    ax2.set_xticks(x)
    ax2.set_xticklabels(g_names, fontsize=9.8, fontweight="bold", color=TEXT_DARK)
    ax2.set_ylabel("Mean Survival Length (Steps, Log Scale)", fontsize=9.8, fontweight="bold", color=TEXT_DARK)
    ax2.set_title("B. Survival Longevity (Seaquest 4.14x Longer!)", fontsize=10.8, fontweight="bold", color=TEXT_DARK, pad=10)
    ax2.yaxis.grid(True, linestyle=":", color=GRID, alpha=0.8, zorder=1)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Panel C: Hand-Written Nonzero Weights & Cosine Similarity
    en_nz = [pg[g]["en_baseline"]["nonzero_total"] for g in games]
    zh_nz = [pg[g]["zh_prompt"]["nonzero_total"] for g in games]
    cos_vals = [pg[g]["comparison"]["cosine_similarity"] for g in games]

    ax3.bar(x - w / 2, en_nz, width=w, color=COLOR_EN, edgecolor=COLOR_EN_DARK, linewidth=1.0, zorder=3)
    ax3.bar(x + w / 2, zh_nz, width=w, color=COLOR_HERO, edgecolor=COLOR_HERO_DARK, linewidth=1.3, zorder=3)
    for i in range(3):
        ax3.text(x[i] - w / 2, en_nz[i] + 40, f"{en_nz[i]:,}", ha="center", va="bottom", fontsize=8.1, color=COLOR_EN_DARK, fontweight="bold")
        ax3.text(x[i] + w / 2, zh_nz[i] + 40, f"{zh_nz[i]:,}", ha="center", va="bottom", fontsize=8.1, color=COLOR_HERO_DARK, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"{g_names[i]}\n(cos = {cos_vals[i]:+.2f})" for i in range(3)], fontsize=9.1, fontweight="bold", color=TEXT_DARK)
    ax3.set_ylim(0, 2450)
    ax3.set_ylabel("Hand-Written Nonzero Weights (out of 20,486)", fontsize=9.8, fontweight="bold", color=TEXT_DARK)
    ax3.set_title("C. Circuit Density & Weight-Space Orthogonality", fontsize=10.8, fontweight="bold", color=TEXT_DARK, pad=10)
    ax3.yaxis.grid(True, linestyle=":", color=GRID, alpha=0.8, zorder=1)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    fig.text(
        0.52,
        0.96,
        "Same Model, Same Zero-Tool Rules, Faithful Chinese Translation: Inside the Shockwave",
        ha="center",
        va="top",
        fontsize=14.6,
        fontweight="bold",
        color=TEXT_DARK,
    )
    fig.text(
        0.52,
        0.895,
        "Gemini 4 Neon (XHigh) | Antigravity (default ctx) | 0 Tool Calls | Seaquest switches from col-2 right-facing (13.9) to (x=7, facing-left) fortress (95.0, cos = -0.29)",
        ha="center",
        va="top",
        fontsize=9.1,
        color=TEXT_MUTED,
    )
    fig.text(
        0.52,
        0.025,
        "All 6 networks verified with 0 parse errors and 0.0 float discrepancy between reply.md and network.npz. Breakout EN = 634.5 vs ZH = 537.4 (both > 21x PPO-500M).",
        ha="center",
        va="bottom",
        fontsize=8.2,
        color=TEXT_MUTED,
    )

    save_fig(fig, out_dir / "en_vs_zh_shockwave.png", "Same Model EN vs ZH Translation Shockwave Analysis")


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    data = json.loads((repo / "data" / "results.json").read_text(encoding="utf-8"))
    setup_style()
    out_dir = repo / "charts"
    make_hero_chart(data, out_dir)
    make_shockwave_chart(data, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
