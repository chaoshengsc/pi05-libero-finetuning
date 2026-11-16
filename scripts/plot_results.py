"""Build publication figures from validated episode evidence."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter
from analyze_results import validate_repository, per_task, paired_counts

BG = "#FCFBF8"
INK = "#222E39"
MUTED = "#526173"
GRID = "#E2E5E6"
TEAL = "#007F78"
BLUE = "#376195"
LABELS = ["Between plate / ramekin", "Next to ramekin", "Table center",
          "On cookie box", "Inside top drawer", "On ramekin",
          "Next to cookie box", "On stove", "Next to plate", "On cabinet"]
STEPS = [800, 1000, 1250, 1500, 1750, 2000]


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "text.color": INK, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "svg.fonttype": "path", "svg.hashsalt": "pi05-libero",
        "savefig.facecolor": BG, "figure.facecolor": BG,
    })


def axis_style(ax):
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, pad=8)
    ax.set_axisbelow(True)


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(ROOT / f"assets/{name}.{ext}", dpi=160, metadata={"Date": None} if ext == "svg" else None)
    plt.close(fig)


def title(fig, heading, subtitle):
    fig.text(.065, .92, heading, fontsize=22, weight="bold")
    fig.text(.065, .864, subtitle, fontsize=11, color=MUTED)


def checkpoints(groups):
    fig, ax = plt.subplots(figsize=(11, 5.6))
    fig.subplots_adjust(left=.10, right=.95, top=.76, bottom=.19)
    axis_style(ax)
    rates = [sum(r["success"] for r in groups[(str(s), "development")]) for s in STEPS]
    title(fig, "Does more training improve task success?",
          "Development split · 100 fixed initial states per checkpoint · one training seed")
    ax.plot(STEPS, rates, color=TEAL, lw=2.4, marker="o", ms=7)
    ax.scatter([0], [0], color=MUTED, marker="D", s=42, zorder=5)
    ax.annotate("Pretrained baseline\n0 / 100", (0, 0), xytext=(12, 14),
                textcoords="offset points", fontsize=10, color=MUTED)
    for s, rate in zip(STEPS, rates):
        ax.annotate(f"{rate}%", (s, rate), xytext=(0, 11),
                    textcoords="offset points", ha="center", color=TEAL, weight="bold")
    ax.set(xlim=(-75, 2100), ylim=(-4, 112), xticks=[0] + STEPS,
           yticks=[0, 25, 50, 75, 100], xlabel="Optimizer updates")
    ax.set_xticklabels(["Base", "800", "1,000", "1,250", "1,500", "1,750", "2,000"], fontsize=10)
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.grid(axis="y", color=GRID)
    fig.text(.10, .055, "One fixed set of tasks and initial states. Held-out outcomes are reported separately.",
             fontsize=10, color=MUTED)
    save(fig, "checkpoint-results")


def heatmap(counts):
    data = [[counts[(str(s), "development", task)][0] * 10 for s in STEPS] for task in range(10)]
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.subplots_adjust(left=.32, right=.90, top=.77, bottom=.19)
    cmap = LinearSegmentedColormap.from_list("research_teal", [(0, BG), (.6, "#E0EDE7"), (.7, "#BCDAD2"), (.8, "#82B9B0"), (.9, "#21766F"), (1, "#005F59")])
    ax.imshow(data, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    title(fig, "Where do checkpoints gain — and lose?",
          "Development success by task · 10 fixed initial states per cell")
    ax.set_xticks(range(6), [f"{s:,}" for s in STEPS])
    ax.set_yticks(range(10), [f"{t:02d}  {label}" for t, label in enumerate(LABELS)], fontsize=10)
    ax.tick_params(length=0, pad=11)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i, row in enumerate(data):
        for j, value in enumerate(row):
            ax.text(j, i, f"{value // 10}/10", ha="center", va="center",
                    color="white" if value >= 85 else INK, fontsize=11, weight="bold")
    ax.set_xticks([x - .5 for x in range(7)], minor=True)
    ax.set_yticks([x - .5 for x in range(11)], minor=True)
    ax.grid(which="minor", color=BG, linewidth=3)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("Optimizer updates", labelpad=12)
    fig.text(.065, .04, "All tasks ask the robot to place the black bowl on the plate. Labels identify its starting location.",
             fontsize=10, color=MUTED)
    save(fig, "task-checkpoint-matrix")


def held_out(counts):
    data = [counts[("800", "held-out", t)][0] for t in range(10)]
    overall = sum(data) / 4
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.subplots_adjust(left=.32, right=.91, top=.77, bottom=.13)
    title(fig, "800 updates · held-out performance",
          f"{sum(data)} / 400 successes ({overall:.1f}%) · 40 different initial states per task")
    axis_style(ax)
    ax.barh(range(10), [x * 2.5 for x in data], height=.58, color=BLUE)
    ax.axvline(overall, color=INK, ls=(0, (3, 4)), lw=1)
    for task, wins in enumerate(data):
        ax.text(104, task, f"{wins}/40", va="center", fontsize=11)
    ax.set(xlim=(0, 115), xticks=[0, 25, 50, 75, 100], yticks=range(10), xlabel="Task success rate")
    ax.set_yticklabels([f"{t:02d}  {label}" for t, label in enumerate(LABELS)], fontsize=10)
    ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID)
    fig.text(.065, .04, "Dashed line = overall success. These initial states are disjoint from development.",
             fontsize=10, color=MUTED)
    save(fig, "held-out-tasks")


def pair_plot(rows):
    pair = paired_counts(rows, "1750", "2000", "development")
    fig, ax = plt.subplots(figsize=(11, 3.7))
    fig.subplots_adjust(left=.08, right=.94, top=.62, bottom=.30)
    fig.text(.065, .84, "One extra success. Seven changed outcomes.", fontsize=22, weight="bold")
    fig.text(.065, .74, "1,750 vs 2,000 updates · 100 identical task / initial-state pairs · development", fontsize=11, color=MUTED)
    vals = [pair["both_success"], pair["left_only"], pair["right_only"], pair["both_failure"]]
    labels = ["Both succeed", "Only 1,750", "Only 2,000", "Both fail"]
    colors = [TEAL, "#748395", BLUE, "#DEE2E4"]
    start = 0
    for v, color in zip(vals, colors):
        ax.barh(0, v, left=start, color=color, height=.48, edgecolor=BG, linewidth=2)
        start += v
    ax.set_xlim(0, 100)
    ax.set_ylim(-.45, .45)
    ax.axis("off")
    for x, value, label, color in zip([.08,.32,.56,.80], vals, labels, colors):
        fig.text(x, .24, str(value), fontsize=23, weight="bold", color=INK)
        fig.text(x, .15, label, fontsize=10, color=MUTED)
    fig.text(.08, .04, "A descriptive comparison on development data. The same paired analysis is planned for held-out episodes.",
             fontsize=9, color=MUTED)
    save(fig, "paired-development")


def main():
    rows, groups = validate_repository()
    style()
    counts = per_task(rows)
    checkpoints(groups)
    heatmap(counts)
    held_out(counts)
    pair_plot(rows)
    print("Built four figures as PNG and SVG from validated episode records.")


if __name__ == "__main__":
    main()
