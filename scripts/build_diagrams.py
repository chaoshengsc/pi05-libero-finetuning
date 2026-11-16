"""Render an experimental-design schematic and original rollout frame sequences."""
import io
import json
import subprocess
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from plot_results import ROOT, INK, BLUE, GRAY, style, save
from analyze_results import validate_repository


def box(ax, x, y, w, h, lines, facecolor="white"):
    ax.add_patch(Rectangle((x, y), w, h, lw=.65, edgecolor=INK, facecolor=facecolor))
    ax.text(x+w/2, y+h/2, "\n".join(lines), ha="center", va="center", fontsize=9, linespacing=1.4)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9, lw=.7, color=INK))


def experiment():
    fig, ax = plt.subplots(figsize=(8.2, 3.35))
    fig.subplots_adjust(left=.01, right=.99, bottom=.03, top=.98)
    ax.set(xlim=(0, 10), ylim=(0, 4)); ax.axis("off")
    ax.text(.12, 3.78, "(a) Fine-tuning", fontsize=11)
    box(ax, .12, 2.55, 2.35, .93, ["LIBERO demonstrations", "1,608 training / 85 validation"])
    box(ax, 3.1, 2.55, 2.8, .93, [r"$\pi_{0.5}$ base (frozen) + LoRA", "rank 64; alpha 32"], "#F2F4F6")
    box(ax, 6.56, 2.55, 3.22, .93, ["Checkpoints (optimizer updates)", "800, 1,000, 1,250, 1,500, 1,750, 2,000"])
    arrow(ax, (2.47, 3.02), (3.1, 3.02))
    arrow(ax, (5.9, 3.02), (6.56, 3.02))
    ax.text(4.5, 2.29, "RTX A5000; effective batch 256; seed 0", ha="center", fontsize=9, color=GRAY)

    ax.text(.12, 1.83, "(b) Closed-loop evaluation", fontsize=11)
    box(ax, .12, .32, 2.35, 1.14, ["LIBERO-Spatial", "10 tasks × 50 initial states", "100 development / 400 held-out"])
    box(ax, 3.1, .32, 2.8, 1.14, ["Observation → policy → action", "Replan after 5 actions", "Task-success predicate"])
    box(ax, 6.56, .32, 3.22, 1.14, ["Paired checkpoint comparison", "Same task / initial-state pairs", "One scored rollout per pair"])
    arrow(ax, (2.47, .89), (3.1, .89))
    arrow(ax, (5.9, .89), (6.56, .89))
    ax.plot([8.17, 8.17, 4.5], [2.55, 1.95, 1.95], color=INK, lw=.7)
    arrow(ax, (4.5, 1.95), (4.5, 1.46))
    save(fig, "experiment-design")


def selected_frame(path, index):
    data = subprocess.check_output([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
        "-vf", f"select=eq(n\\,{index}),crop=iw/2:ih:0:0",
        "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-",
    ])
    return plt.imread(io.BytesIO(data), format="png")


def rollouts():
    media = json.loads((ROOT / "assets/demos/manifest.json").read_text())
    clips = [c for c in media["clips"] if c["task_id"] == 0]
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.65))
    fig.subplots_adjust(left=.01, right=.99, top=.90, bottom=.015, hspace=.38, wspace=.045)
    for row, clip in enumerate(clips):
        steps = clip["policy_steps"]
        frame_indices = [0, (steps-1)//2, steps-1]
        label = "(a) Success" if clip["success"] else "(b) Failure at the step limit"
        for col, index in enumerate(frame_indices):
            ax = axes[row, col]
            ax.imshow(selected_frame(ROOT / clip["mp4"], index))
            ax.set_title(f"Step {index+1}", fontsize=9, pad=3)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(.4); spine.set_edgecolor("#888888")
        y = axes[row, 0].get_position().y1 + .065
        fig.text(.013, y, f"{label} (initial state {clip['initial_state_id']:02d})", fontsize=10)
    save(fig, "qualitative-rollouts")


if __name__ == "__main__":
    validate_repository()
    style()
    experiment()
    rollouts()
    print("Built method schematic and rollout sequences (PNG, SVG, PDF).")
