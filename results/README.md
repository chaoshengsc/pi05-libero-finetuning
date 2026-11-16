# Public experiment data

[Project overview](../README.md) · [Reproduce the analysis](../docs/reproduce.md)

## Episode records

[`episodes.csv`](episodes.csv) contains 1,100 final valid outcomes. Only complete checkpoint/split groups are included.

| Column | Meaning |
|:--|:--|
| `checkpoint` | `pi05_base`, or optimizer update count |
| `split` | `development` or `held-out` |
| `task_id` | LIBERO-Spatial task index, 0–9 under task order 0 |
| `initial_state_id` | Index in the task's fixed 50-state initial-state set |
| `success` | Native simulator task success, 1 or 0 |
| `policy_steps` | Number of policy-controlled simulator steps, up to 220 |

Unique key: `(checkpoint, split, task_id, initial_state_id)`.

An initial-state ID is only meaningful within its task. To compare two checkpoints, match on **both** task and initial state, within the same split. These records do not include the full action trace or every technical retry.

## Per-task aggregates

[`per_task.csv`](per_task.csv) contains one row per completed checkpoint/split/task. Columns are `checkpoint`, `split`, `task_id`, `successes`, and `episodes`.

Development has 10 episodes per task. Held-out has 40. Aggregate success is the sum of successes divided by the sum of episodes, also equal to the unweighted average task rate.

## Task dictionary

All ten tasks ask the robot to place a black bowl on a plate.

| ID | Bowl starting location |
|:--|:--|
| 00 | Between the plate and the ramekin |
| 01 | Next to the ramekin |
| 02 | Table center |
| 03 | On the cookie box |
| 04 | Inside the top drawer of the wooden cabinet |
| 05 | On the ramekin |
| 06 | Next to the cookie box |
| 07 | On the stove |
| 08 | Next to the plate |
| 09 | On the wooden cabinet |

Exact task names, file identities, and initial-state lists appear in [evaluation.json](../configs/evaluation.json).

## Provenance

[`provenance.json`](provenance.json) records the export timestamp, published coverage, CSV checksum, and source-record digests. For each group, the source digest is SHA-256 of the concatenated binary SHA-256 digests of final record files, ordered lexicographically by task/initial-state directory.

The source digests identify the original files. Those full source files are not part of this public release. The public validator checks the exported CSV against its manifest and aggregate table.

The snapshot contains 100 baseline development episodes, 6 × 100 trained development episodes, and 400 held-out episodes at checkpoint 800. A pending group has no published rows and must not be assigned a zero score.
