# Public experiment data

[Project overview](../README.md) · [Reproduce the analysis](../docs/reproduce.md)

## Episode records

[`episodes.csv`](episodes.csv) contains 3,500 final valid outcomes under inference seed 20261104. A0 and the six trained checkpoints each contribute 100 development and 400 held-out outcomes.

[`inference_seeds.csv`](inference_seeds.csv) adds 1,600 episodes, covering checkpoints 1,250 and 2,000 at inference seeds 20261118 and 20261119. Each checkpoint–seed group uses the same 400 held-out initial states. The file adds the `inference_seed` column to the key below. Eighty scored pilot episodes are reused once, and the pilot's original-seed replays are outside the published counts.

| Column | Meaning |
|:--|:--|
| `checkpoint` | `pi05_base` for the matched step-zero A0, or the optimizer update count |
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

[`provenance.json`](provenance.json) records the export timestamp, published coverage, CSV checksums, and source-record digests. Development digests retain the prior snapshot's SHA-256 of concatenated binary final-record digests in task and initial-state order. The held-out re-export computes SHA-256 over the corresponding lowercase hexadecimal digests joined by LF, with no trailing LF. Source report digests identify the three completed evaluation reports. The manifest also records the single first-observation exception used for strict A0 pairing.

The public validator checks the projected outcomes and media against their manifests. The full workstation records are available only in the experiment archive.
