# π0.5 fine-tuning on LIBERO

[简体中文](README_CN.md) · [Experiment protocol](docs/experiment.md) · [Results data](results/per_task.csv)

Parameter-efficient fine-tuning of the official `pi05_base` policy on a single RTX A5000. This project follows one training run across six checkpoints and measures how well each policy completes manipulation tasks in closed-loop simulation.

**First held-out result: 346 / 400 successful episodes (86.5%) at 800 updates.** Training reached 2,000 optimizer updates. The 2,000-update checkpoint scored 97 / 100 on the development split, where the pretrained baseline scored 0 / 100.

![Development success across six checkpoints and 800-update held-out success by task](assets/checkpoint-results.png)

## Results

| Policy checkpoint | Development · 100 episodes | Held-out · 400 episodes |
|:--|--:|--:|
| Pretrained `pi05_base` | 0 / 100 | — |
| 800 updates | 89 / 100 | 346 / 400 |
| 1,000 updates | 90 / 100 | In progress |
| 1,250 updates | 95 / 100 | In progress |
| 1,500 updates | 91 / 100 | In progress |
| 1,750 updates | 96 / 100 | In progress |
| 2,000 updates | 97 / 100 | In progress |

The development and held-out splits use disjoint initial states from the same ten LIBERO-Spatial tasks. All checkpoints use the same initial states within each split. The held-out sweep is still in progress as of November 16, 2026. Success is defined by the simulator's task predicate, and each task contributes equally.

## Experiment at a glance

| Stage | Design |
|:--|:--|
| Data | 1,693 LIBERO demonstrations across 40 tasks. 1,608 for training and 85 for offline validation. |
| Fine-tuning | Frozen pretrained weights with all-linear LoRA. Rank 64, alpha 32, training seed 0. |
| Compute | One RTX A5000. Microbatch 8 × 32 gradient accumulation steps = 256 samples per optimizer update. |
| Evaluation | 10 tasks × 10 development states. 10 tasks × 40 held-out states. One scored rollout per state and checkpoint. |

## Inspect the results

- [Experiment protocol](docs/experiment.md) describes the splits, version pins, and how to interpret the measurements.
- [Per-task counts](results/per_task.csv) contain the data behind the table and figure.
- [Plot script](scripts/plot_results.py) regenerates the figure from the CSV using Matplotlib.

The results describe one training seed and the LIBERO-Spatial simulation suite. The remaining held-out results will determine the six-checkpoint ranking.

## Upstream

[Physical Intelligence / OpenPI](https://github.com/Physical-Intelligence/openpi) · [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) · [Training dataset](https://huggingface.co/datasets/physical-intelligence/libero/tree/9dfa69510ea9e1613fc54112bc706444b686a231)
