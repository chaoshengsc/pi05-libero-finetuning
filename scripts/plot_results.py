"""Render academic figures from the validated public experiment snapshot."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from analyze_results import validate_repository, per_task, paired_counts

INK = "#202020"
BLUE = "#3B5B7A"
GRAY = "#666666"
GRID = "#DCDCDC"
STEPS = [800, 1000, 1250, 1500, 1750, 2000]
LABELS = ["Between plate and ramekin", "Next to ramekin", "Table center",
          "On cookie box", "Inside top drawer", "On ramekin",
          "Next to cookie box", "On stove", "Next to plate", "On cabinet"]
CMAP = LinearSegmentedColormap.from_list("task_counts", ["#FFFFFF", "#DDE5ED", BLUE])


def style():
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
        "axes.linewidth": .65, "xtick.color": INK, "ytick.color": INK,
        "xtick.major.width": .65, "ytick.major.width": .65,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.bbox": "tight",
        "savefig.pad_inches": .06, "svg.fonttype": "path",
        "svg.hashsalt": "pi05-libero-paper", "pdf.fonttype": 42,
    })


def axes_style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out")
    ax.grid(axis="y", color=GRID, linewidth=.45)


def save(fig, name):
    for ext in ("png", "svg", "pdf"):
        metadata = {"Date": None} if ext == "svg" else {"CreationDate": None, "ModDate": None} if ext == "pdf" else None
        path = ROOT / f"assets/{name}.{ext}"
        fig.savefig(path, dpi=240, metadata=metadata)
        if ext == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def draw_development(ax, groups):
    axes_style(ax)
    rates = [100 * sum(r["success"] for r in groups[(str(s), "development")]) /
             len(groups[(str(s), "development")]) for s in STEPS]
    ax.plot(STEPS, rates, color=BLUE, lw=1.25, marker="o", ms=4,
            markerfacecolor="white", markeredgewidth=1.1)
    for step, rate in zip(STEPS, rates):
        ax.annotate(f"{rate:g}", (step, rate), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.set(xlim=(720, 2080), ylim=(80, 101.5), xticks=STEPS,
           yticks=[80, 85, 90, 95, 100], xlabel="Optimizer updates",
           ylabel="Success rate (%)")
    ax.set_xticklabels(["800", "1,000", "1,250", "1,500", "1,750", "2,000"], rotation=25, ha="right")


def draw_held_out(ax, counts):
    axes_style(ax)
    values = [counts[("800", "held-out", t)][0] for t in range(10)]
    rates = [100 * count / 40 for count in values]
    ax.bar(range(10), rates, width=.58, facecolor="#DDE5ED", edgecolor=BLUE, linewidth=.65)
    overall = 100 * sum(values) / 400
    ax.axhline(overall, color=GRAY, lw=.9, ls=(0, (3, 3)), label=f"Overall: {overall:.1f}%")
    ax.set(xlim=(-.65, 9.65), ylim=(0, 110), xticks=range(10),
           yticks=[0, 25, 50, 75, 100], xlabel="Task ID", ylabel="Success rate (%)")
    ax.set_xticklabels([f"{t:02d}" for t in range(10)])
    for task, count in enumerate(values):
        ax.text(task, rates[task] + 2, str(count), fontsize=8, ha="center")


def main_results(groups, counts):
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.9))
    fig.subplots_adjust(left=.08, right=.99, bottom=.25, top=.85, wspace=.28)
    draw_development(axes[0], groups)
    axes[0].set_title("(a) Development: checkpoint comparison", pad=10)
    draw_held_out(axes[1], counts)
    axes[1].set_title("(b) Held-out: checkpoint 800", pad=10)
    save(fig, "evaluation-results")


def standalone_results(groups, counts):
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    draw_development(ax, groups)
    fig.subplots_adjust(left=.12, bottom=.24, right=.98, top=.94)
    save(fig, "checkpoint-results")
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    draw_held_out(ax, counts)
    fig.subplots_adjust(left=.12, bottom=.20, right=.98, top=.94)
    save(fig, "held-out-tasks")


def task_matrix(counts):
    data = [[counts[(str(step), "development", task)][0] for step in STEPS] for task in range(10)]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    fig.subplots_adjust(left=.37, right=.98, bottom=.16, top=.98)
    ax.imshow(data, cmap=CMAP, vmin=0, vmax=10, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(6), [f"{s:,}" for s in STEPS])
    ax.set_yticks(range(10), [f"{t:02d}  {label}" for t, label in enumerate(LABELS)])
    ax.set_xlabel("Optimizer updates")
    ax.tick_params(length=0, pad=6)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i, row in enumerate(data):
        for j, value in enumerate(row):
            ax.text(j, i, f"{value}/10", ha="center", va="center", fontsize=9,
                    color="white" if value >= 9 else INK)
    ax.set_xticks([x - .5 for x in range(7)], minor=True)
    ax.set_yticks([x - .5 for x in range(11)], minor=True)
    ax.grid(which="minor", color="white", linewidth=.7)
    ax.tick_params(which="minor", length=0)
    save(fig, "task-checkpoint-matrix")


def paired_matrix(rows):
    pair = paired_counts(rows, "1750", "2000", "development")
    values = [[pair["both_success"], pair["left_only"]],
              [pair["right_only"], pair["both_failure"]]]
    fig, ax = plt.subplots(figsize=(3.7, 2.9))
    fig.subplots_adjust(left=.27, right=.95, bottom=.23, top=.95)
    ax.imshow(values, cmap=CMAP, vmin=0, vmax=100)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Success", "Failure"],
           yticklabels=["Success", "Failure"], xlabel="Checkpoint 2,000", ylabel="Checkpoint 1,750")
    ax.tick_params(length=0, pad=7)
    for spine in ax.spines.values():
        spine.set_linewidth(.5)
    ax.set_xticks([.5], minor=True); ax.set_yticks([.5], minor=True)
    ax.grid(which="minor", color="white", linewidth=1)
    ax.tick_params(which="minor", length=0)
    for i, row in enumerate(values):
        for j, count in enumerate(row):
            ax.text(j, i, str(count), ha="center", va="center", fontsize=14,
                    color="white" if count > 80 else INK)
    save(fig, "paired-development")


def main():
    rows, groups = validate_repository()
    style()
    counts = per_task(rows)
    main_results(groups, counts)
    standalone_results(groups, counts)
    task_matrix(counts)
    paired_matrix(rows)
    print("Built five result figures (PNG, SVG, PDF) from validated episode records.")


if __name__ == "__main__":
    main()
