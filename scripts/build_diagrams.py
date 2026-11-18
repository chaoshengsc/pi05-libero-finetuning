"""Render an experimental-design schematic and original rollout frame sequences."""
import io
import json
import subprocess
from plot_results import ROOT, INK, BLUE, GRAY, style, save
from analyze_results import validate_repository
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch


def box(ax, x, y, w, h, lines, facecolor="white"):
    ax.add_patch(Rectangle((x, y), w, h, lw=.65, edgecolor=INK, facecolor=facecolor))
    ax.text(x+w/2, y+h/2, "\n".join(lines), ha="center", va="center", fontsize=9, linespacing=1.4)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9, lw=.7, color=INK))


def experiment():
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    fig.subplots_adjust(left=.01, right=.99, bottom=.02, top=.99)
    ax.set(xlim=(0, 10), ylim=(0, 5.25))
    ax.axis("off")
    ax.text(.15, 4.98, "(a) Train one adapter trajectory on one RTX A5000", fontsize=11)
    box(ax, .15, 3.36, 2.84, 1.31,
        ["Demonstration sample", "Scene + wrist images + instruction", "10 consecutive demo actions"])
    box(ax, 3.58, 3.36, 2.84, 1.31,
        ["Frozen base + all-linear LoRA", "Flow matching loss", "8 samples × 32 microbatches / update"], "#F2F4F6")
    box(ax, 7.01, 3.36, 2.84, 1.31,
        ["A0: zero updates", "Save six checkpoints", "800 / 1k / 1,250 / 1,500 / 1,750 / 2k"])
    arrow(ax, (2.99, 4.02), (3.58, 4.02))
    arrow(ax, (6.42, 4.02), (7.01, 4.02))
    ax.text(.15, 2.99, "1,608 training demos / 85 offline validation demos", fontsize=9, color=GRAY)

    ax.text(.15, 2.47, "(b) Compare policies on matched simulator initial states", fontsize=11)
    ax.text(.15, 2.12, "10 LIBERO-Spatial tasks  ·  100 development states  ·  400 separate held-out states",
            fontsize=9, color=GRAY)
    box(ax, .15, .56, 3.10, 1.27,
        ["Fine-tuning gain", "A0 vs 2,000 updates", "400 states (399 strict pairs)"])
    box(ax, 3.45, .56, 3.10, 1.27,
        ["Checkpoint ranking", "Six trained checkpoints", "400 states at the original seed"], "#F2F4F6")
    box(ax, 6.75, .56, 3.10, 1.27,
        ["Inference seed sensitivity", "1,250 vs 2,000 updates", "400 shared states × 2 new seeds"])
    ax.text(.15, .20, "One score per checkpoint × inference seed × task and initial state. Replan every five actions.",
            fontsize=9, color=GRAY)
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
