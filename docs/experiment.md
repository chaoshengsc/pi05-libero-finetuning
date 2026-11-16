# Experiment protocol

## Training

The run starts from the official OpenPI `pi05_base` checkpoint and trains rank-64, alpha-32 LoRA adapters on the target linear layers with the base weights frozen. The run uses seed 0 on one RTX A5000, a microbatch of 8 and gradient accumulation of 32. The six saved checkpoints correspond to 800, 1,000, 1,250, 1,500, 1,750, and 2,000 optimizer updates.

The dataset contains 1,693 demonstrations across 40 tasks. A demonstration-level split allocates 1,608 to training and 85 to offline validation. Normalization statistics are fitted on the training split.

| Component | Pinned upstream revision |
|:--|:--|
| OpenPI | [`215abfb217db`](https://github.com/Physical-Intelligence/openpi/commit/215abfb217dbac7d5f1273282331b9b1866c0479) |
| LIBERO | [`8f1084e3132a`](https://github.com/Lifelong-Robot-Learning/LIBERO/commit/8f1084e3132a39270c3a13ebe37270a43ece2a01) |
| Demonstrations | [`physical-intelligence/libero@9dfa69510ea9`](https://huggingface.co/datasets/physical-intelligence/libero/tree/9dfa69510ea9e1613fc54112bc706444b686a231) |

## Closed-loop evaluation

| Split | Task coverage | Initial states per task | Rollouts per policy |
|:--|--:|--:|--:|
| Development | 10 LIBERO-Spatial tasks | 10 | 100 |
| Held-out | 10 LIBERO-Spatial tasks | 40 | 400 |

Each task has 50 fixed initial states partitioned into the two disjoint splits. Within a split, every checkpoint sees the same task and initial-state pairs. The evaluator rolls out the policy in simulation and records a Boolean success from the task predicate. A valid rollout contributes one outcome, and a technical failure may be retried once on the same initial state. Because every task has equal episode counts, the success rate is both the mean across tasks and the fraction of successful episodes.

The held-out protocol is fixed at six checkpoints × 400 episodes. Comparisons between checkpoints are paired by task and initial state. Development results guide inspection, while the final ranking uses the complete held-out run.

Evaluation protocol SHA-256: `7039fd18b555a239fc5505c50ff26468a246d61d78aef18fe77aef7e1b190e70`.

## First held-out result

The 800-update checkpoint completed 400 valid episodes with 346 successes. A workstation restart separated the first 352 episodes from the final 48, which ran with the updated Linux kernel and NVIDIA driver. The same checkpoint and evaluation protocol were used across the restart. This runtime change is part of the interpretation of the first held-out result.

The public CSV reports only completed splits, with one row per checkpoint and task. Its columns are the checkpoint label, split, simulator task index (0–9), successes, and valid episodes. The development baseline is the official pretrained `pi05_base` checkpoint. The table and figure describe simulation under a single training seed.

## Regenerate the figure

With Python and Matplotlib available, run from the repository root:

```sh
python scripts/plot_results.py
```
