"""Generate result figures directly into paper/figures/ for LaTeX inclusion."""
import pathlib, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = pathlib.Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

BG_DARK    = "#1B1B2F"
CYAN       = "#00B4D8"
GOLD       = "#FFD66B"
LIGHT_BLUE = "#90E0EF"
WHITE      = "#FFFFFF"
GRAY       = "#CCCCCC"

MODEL_COLORS = {"BERT": "#4BACC6", "MNLI-BERT": "#F79646", "RoBERTa": "#9BBB59"}
MODES  = ["baseline", "repeat", "template", "oracle", "llm"]
MODELS = ["BERT", "MNLI-BERT", "RoBERTa"]

MACRO_F1 = {
    "BERT":      [0.501, 0.536, 0.545, 1.000, 0.755],
    "MNLI-BERT": [0.515, 0.595, 0.518, 1.000, 0.782],
    "RoBERTa":   [0.511, 0.597, 0.546, 1.000, 0.754],
}

def _dark_fig(figsize=(10, 4.8)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=BG_DARK)
    ax.set_facecolor(BG_DARK)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRAY); spine.set_linewidth(0.6)
    ax.tick_params(colors=GRAY, labelsize=11)
    return fig, ax

def macro_f1_chart():
    fig, ax = _dark_fig()
    x     = np.arange(len(MODES))
    width = 0.25
    for i, (model, vals) in enumerate(MACRO_F1.items()):
        bars = ax.bar(x + (i-1)*width, vals, width, label=model,
                      color=MODEL_COLORS[model], edgecolor=BG_DARK, linewidth=0.8, zorder=3)
        for bar, val in zip(bars, vals):
            if val < 0.99:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                        f"{val:.2f}", ha="center", va="bottom",
                        fontsize=8, color=WHITE, fontweight="bold")
    # LLM highlight
    llm_i = MODES.index("llm")
    ax.axvspan(x[llm_i]-width*1.6, x[llm_i]+width*1.6, color=GOLD, alpha=0.09, zorder=1)
    ax.text(x[llm_i], 1.03, "★ LLM (ours)", ha="center", va="bottom", fontsize=9,
            color=GOLD, fontweight="bold", transform=ax.get_xaxis_transform())
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in MODES], color=WHITE, fontsize=11)
    ax.set_ylabel("Macro-F1", fontsize=12, color=LIGHT_BLUE, labelpad=8)
    ax.set_ylim(0, 1.12)
    ax.set_title("Macro-F1 by Augmentation Condition and Model", fontsize=13,
                 color=GOLD, fontweight="bold", pad=10)
    ax.yaxis.grid(True, color=GRAY, alpha=0.2, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, facecolor="#222A48", edgecolor=CYAN, labelcolor=WHITE, framealpha=0.9)
    fig.tight_layout()
    out = OUT / "macro_f1.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)
    print(f"Saved: {out}")

def per_class_chart():
    classes  = ["Yes", "Prob. Yes", "Yes Cond.", "No", "Prob. No", "In Middle"]
    baseline = [0.845, 0.541, 0.894, 0.812, 0.000, 0.000]
    llm      = [0.845, 0.763, 0.897, 0.749, 0.717, 0.719]

    fig, ax = _dark_fig(figsize=(10, 4.8))
    x     = np.arange(len(classes))
    width = 0.35
    b1 = ax.bar(x-width/2, baseline, width, label="Baseline (BERT-MNLI)",
                color="#4BACC6", edgecolor=BG_DARK, linewidth=0.8, zorder=3)
    b2 = ax.bar(x+width/2, llm,      width, label="LLM (BERT-MNLI)",
                color=GOLD,    edgecolor=BG_DARK, linewidth=0.8, zorder=3)
    for bar in list(b1)+list(b2):
        v = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, v+0.015,
                f"{v:.2f}", ha="center", va="bottom",
                fontsize=8.5, color=WHITE, fontweight="bold")
    # highlight hard classes
    ax.axvspan(3.5, 5.5, color="#FF4444", alpha=0.07, zorder=1)
    ax.text(4.5, 1.03, "Zero → recovered", ha="center", va="bottom",
            fontsize=9, color="#FF8888", transform=ax.get_xaxis_transform())
    ax.set_xticks(x)
    ax.set_xticklabels(classes, color=WHITE, fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=12, color=LIGHT_BLUE, labelpad=8)
    ax.set_ylim(0, 1.12)
    ax.set_title("Per-Class F1: Baseline vs. LLM Augmentation (BERT-MNLI)", fontsize=13,
                 color=GOLD, fontweight="bold", pad=10)
    ax.yaxis.grid(True, color=GRAY, alpha=0.2, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, facecolor="#222A48", edgecolor=CYAN, labelcolor=WHITE, framealpha=0.9)
    fig.tight_layout()
    out = OUT / "per_class_f1.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)
    print(f"Saved: {out}")

macro_f1_chart()
per_class_chart()
print("Done.")
