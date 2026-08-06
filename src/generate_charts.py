"""
Generates the static PNG charts embedded in README.md from
results/*.json. Re-run after any evaluation change so the charts stay in
sync with the numbers in the text.

Colors follow a validated categorical palette (fixed hue order, not
cycled) rather than default matplotlib colors -- see the dataviz
guidance this was built against for why: arbitrary/cycled hues aren't
colorblind-safe by construction, a fixed validated order is.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
CHARTS_DIR = RESULTS_DIR / "charts"

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

BLUE = "#2a78d6"    # categorical slot 1
ORANGE = "#eb6834"  # categorical slot 2
AQUA = "#1baf7a"    # categorical slot 3

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "text.color": INK_PRIMARY,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def _clean_axes(ax):
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(length=0)


def chart_recall_mrr_by_difficulty(summary):
    difficulties = ["easy", "medium", "hard"]
    recall = [summary["by_difficulty"][d]["recall_at_k"] * 100 for d in difficulties]
    mrr = [summary["by_difficulty"][d]["mrr"] * 100 for d in difficulties]

    x = range(len(difficulties))
    width = 0.32

    fig, ax = plt.subplots(figsize=(6, 3.8), dpi=150)
    bars1 = ax.bar([i - width / 2 for i in x], recall, width, label="Recall@5", color=BLUE)
    bars2 = ax.bar([i + width / 2 for i in x], mrr, width, label="MRR x100", color=ORANGE)

    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.0f}", (bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha="center",
                fontsize=9, color=INK_SECONDARY,
            )

    ax.set_xticks(list(x))
    ax.set_xticklabels([d.capitalize() for d in difficulties])
    ax.set_ylim(0, 112)
    ax.set_ylabel("Score (out of 100)")
    ax.set_title("Retrieval quality by question difficulty", loc="left", color=INK_PRIMARY, fontsize=12)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    # Placed above the axes entirely (bbox y > 1) so it can never overlap a
    # bar's value label regardless of bar heights.
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.14), ncol=2, fontsize=9)
    _clean_axes(ax)

    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "recall_mrr_by_difficulty.png")
    plt.close(fig)


def chart_before_after(baseline_summary, improved_summary):
    labels = ["Recall@5", "MRR x100"]
    before = [baseline_summary["recall_at_k"] * 100, baseline_summary["mrr"] * 100]
    after = [improved_summary["recall_at_k"] * 100, improved_summary["mrr"] * 100]

    x = range(len(labels))
    width = 0.32

    fig, ax = plt.subplots(figsize=(5, 3.8), dpi=150)
    bars1 = ax.bar([i - width / 2 for i in x], before, width, label="Before (pure reranker)", color=BLUE)
    bars2 = ax.bar([i + width / 2 for i in x], after, width, label="After (blended, weight=0.85)", color=AQUA)

    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}", (bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha="center",
                fontsize=9, color=INK_SECONDARY,
            )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Score (out of 100)")
    ax.set_title("Before / after the reranker-blend fix", loc="left", color=INK_PRIMARY, fontsize=12)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.16), ncol=1, fontsize=8)
    _clean_axes(ax)

    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "before_after.png")
    plt.close(fig)


def chart_failure_stages(diagnoses):
    stage_labels = {
        "not_found_by_either_search": "Not found by\neither search",
        "lost_in_fusion_pool_size": "Lost in fusion\npool size",
        "demoted_by_reranker": "Demoted by\nreranker",
    }
    counts = {key: 0 for key in stage_labels}
    for d in diagnoses:
        counts[d["diagnosis"]["stage"]] += 1

    labels = list(stage_labels.values())
    values = [counts[key] for key in stage_labels]

    fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=150)
    bars = ax.barh(labels, values, color=BLUE, height=0.55)

    for bar in bars:
        width = bar.get_width()
        ax.annotate(
            f"{int(width)}", (width, bar.get_y() + bar.get_height() / 2),
            xytext=(6, 0), textcoords="offset points", va="center",
            fontsize=10, color=INK_PRIMARY,
        )

    ax.set_xlim(0, max(values) + 1.5)
    ax.set_title("Root cause of the 6 retrieval misses (baseline)", loc="left", color=INK_PRIMARY, fontsize=12)
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    _clean_axes(ax)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])

    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "failure_stages.png")
    plt.close(fig)


def main():
    baseline = json.loads((RESULTS_DIR / "baseline.json").read_text(encoding="utf-8"))
    improved = json.loads((RESULTS_DIR / "improved.json").read_text(encoding="utf-8"))
    diagnoses = json.loads((RESULTS_DIR / "failure_analysis_retrieval.json").read_text(encoding="utf-8"))

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    chart_recall_mrr_by_difficulty(improved["summary"])
    chart_before_after(baseline["summary"], improved["summary"])
    chart_failure_stages(diagnoses)

    print(f"Charts written to {CHARTS_DIR}")


if __name__ == "__main__":
    main()
