"""Draw repository cover and experiment diagram using the shared figure palette."""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from plot_results import BG, INK, MUTED, GRID, TEAL, BLUE, style, save

def cover():
    fig = plt.figure(figsize=(12, 3.45), facecolor=BG)
    ax = fig.add_axes([0,0,1,1])
    ax.set_xlim(0,1200); ax.set_ylim(0,345); ax.axis("off")
    ax.text(58,188,"π0.5",fontsize=84,weight="bold",color=TEAL)
    ax.text(61,133,"on LIBERO",fontsize=30,weight="bold",color=INK)
    ax.plot([431,431],[130,277],color=GRID,lw=1)
    ax.text(473,252,"From pretrained policy",fontsize=25,color=INK,weight="bold")
    ax.text(473,205,"to closed-loop control.",fontsize=25,color=INK,weight="bold")
    ax.text(474,147,"A single-GPU fine-tuning study",fontsize=16,color=MUTED)
    ax.plot([61,1140],[92,92],color=GRID,lw=1)
    ax.text(61,42,"All-linear LoRA",fontsize=12,color=INK)
    ax.text(431,42,"One RTX A5000",fontsize=12,color=INK)
    ax.text(849,42,"Six saved checkpoints",fontsize=12,color=INK)
    save(fig,"cover")

def box(ax,x,y,w,h,title,lines,accent=INK):
    patch=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=5",
                        linewidth=1,edgecolor=GRID,facecolor="white")
    ax.add_patch(patch)
    ax.text(x+22,y+h-35,title,fontsize=14,weight="bold",color=accent)
    for i,line in enumerate(lines):
        ax.text(x+22,y+h-67-i*25,line,fontsize=11,color=MUTED)

def arrow(ax,x,y,to_x,to_y,color=MUTED):
    ax.add_patch(FancyArrowPatch((x,y),(to_x,to_y),arrowstyle="-|>",mutation_scale=13,
                                 linewidth=1.2,color=color))

def pipeline():
    fig=plt.figure(figsize=(12,6.6),facecolor=BG)
    ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,1200);ax.set_ylim(0,660);ax.axis("off")
    ax.text(60,606,"One training run. Two evaluation splits.",fontsize=25,weight="bold")
    ax.text(60,568,"Demonstrations train the policy. Simulator initial states measure task success.",
            fontsize=12,color=MUTED)
    box(ax,60,330,325,188,"Demonstrations",["40 tasks · 1,693 complete episodes",
          "1,608 train / 85 offline validation","Normalization fitted on training only"],TEAL)
    box(ax,440,330,325,188,"All-linear LoRA",["Official pi05_base + all-linear LoRA",
          "Rank 64 / alpha 32 · seed 0","Microbatch 8 × accumulation 32",
          "2,000 optimizer updates"],TEAL)
    box(ax,820,330,320,188,"Six complete checkpoints",["800 / 1,000 / 1,250",
          "1,500 / 1,750 / 2,000","Same training trajectory"],TEAL)
    arrow(ax,387,424,437,424); arrow(ax,767,424,817,424)
    ax.plot([60,1140],[288,288],color=GRID,lw=1)
    box(ax,60,66,325,174,"Simulator initial states",["10 tasks × 50 fixed states",
          "100 development + 400 held-out","Disjoint states within each task"],BLUE)
    box(ax,440,66,325,174,"Closed-loop evaluation",["Each checkpoint uses both splits",
          "Same initial states across checkpoints",
          "Observe → infer → act → observe",
          "Native task-success predicate"],BLUE)
    box(ax,820,66,320,174,"Compare outcomes",["Success per task and checkpoint",
          "Pair by task + initial-state ID",
          "Report successes and failures"],BLUE)
    arrow(ax,387,151,437,151);arrow(ax,767,151,817,151)
    ax.plot([980,980,600],[328,270,270],color=TEAL,lw=1.2)
    arrow(ax,600,270,600,242,TEAL)
    ax.text(1140,18,"Frozen protocol · one rollout per state and checkpoint",fontsize=10,
            color=MUTED,ha="right")
    save(fig,"experiment-design")

if __name__=="__main__":
    style();cover();pipeline()
    print("Built repository cover and experiment diagram.")
