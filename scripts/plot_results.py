"""Rebuild the GitHub results figure from the published per-task counts."""
from collections import defaultdict
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
BACKGROUND = "#FCFBF8"
INK = "#222E39"
MUTED = "#526173"
GRID = "#E2E5E6"
TEAL = "#007F78"
BLUE = "#376195"


def read_results():
    dev = defaultdict(lambda: [0, 0])
    held_out = {}
    with (ROOT / "results/per_task.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            checkpoint = row["checkpoint"]
            wins, total = int(row["successes"]), int(row["episodes"])
            if row["split"] == "development":
                dev[checkpoint][0] += wins
                dev[checkpoint][1] += total
            elif checkpoint == "800" and row["split"] == "held-out":
                held_out[int(row["task_id"])] = (wins, total)
    return dev, held_out


def draw():
    dev, held_out = read_results()
    steps = sorted(int(step) for step in dev if step.isdigit())
    rates = [100 * dev[str(step)][0] / dev[str(step)][1] for step in steps]
    tasks = sorted(held_out)
    counts = [held_out[task] for task in tasks]
    task_rates = [100 * wins / total for wins, total in counts]
    all_wins = sum(wins for wins, _ in counts)
    all_episodes = sum(total for _, total in counts)
    held_out_rate = 100 * all_wins / all_episodes
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "text.color": INK, "axes.labelcolor": MUTED})
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.6, 4.5), facecolor=BACKGROUND)
    fig.subplots_adjust(left=.085, right=.94, top=.72, bottom=.19, wspace=.44)
    fig.text(.045, .916, "π0.5 fine-tuning on LIBERO-Spatial", weight="bold", size=19)
    fig.text(.045, .856, "Closed-loop task success  /  November 2026", color=MUTED, size=10)

    for axis in (left, right):
        axis.set_facecolor(BACKGROUND)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.tick_params(length=0, colors=MUTED, pad=6)
        axis.set_axisbelow(True)
    left.set_title("Development across checkpoints", loc="left", size=11, weight="bold", pad=20)
    left.plot(steps, rates, "o-", color=TEAL, linewidth=2, markersize=6)
    for x, y in zip(steps, rates):
        left.annotate(f"{y:.0f}%", (x, y), xytext=(0, 8),
                      textcoords="offset points", ha="center", color=TEAL, weight="bold")
    left.set(xlim=(700, 2100), ylim=(80, 101), xticks=steps, yticks=[80, 85, 90, 95, 100],
             xlabel="Optimizer updates")
    left.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    left.grid(axis="y", color=GRID, linewidth=.7)

    right.set_title("800 updates on held-out states", loc="left", size=11, weight="bold", pad=20)
    right.barh(tasks, task_rates, height=.61, color=BLUE, alpha=.84)
    right.axvline(held_out_rate, color=INK, linewidth=1.1, linestyle=(0, (3, 3)), zorder=3)
    for task, (wins, total) in zip(tasks, counts):
        right.text(103, task, f"{wins}/{total}", va="center", color=MUTED, size=8)
    right.set(xlim=(0, 113), yticks=tasks, xticks=[0, 25, 50, 75, 100], xlabel="Success rate")
    right.set_yticklabels([f"Task {task:02d}" for task in tasks])
    right.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    right.invert_yaxis()
    right.grid(axis="x", color=GRID, linewidth=.7)

    base_wins, base_count = dev["pi05_base"]
    fig.text(.085, .057, f"100 episodes per point · Base {base_wins}/{base_count} (outside y-axis range)",
             color=MUTED, size=8)
    fig.text(.591, .057, f"Dashed line: {held_out_rate:.1f}% overall ({all_wins}/{all_episodes})",
             color=MUTED, size=8)
    fig.savefig(ROOT / "assets/checkpoint-results.png", dpi=180, facecolor=BACKGROUND)
    plt.close(fig)


if __name__ == "__main__":
    draw()
