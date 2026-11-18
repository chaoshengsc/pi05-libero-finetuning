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

## Audit evidence

- [`pilot_replay.csv`](pilot_replay.csv) identifies the 20 pilot initial states by task and state ID and lists the 40 original-seed replay receipts at 1,250 and 2,000 updates. Five Boolean columns record agreement on first observation, first action block (`atol=1e-7`, `rtol=1e-5`), success, policy-step count, and policy-query count. Replay entries are diagnostics outside the 5,100 scored outcomes.
- The pilot chooses the two lowest SHA-256 ranks per task from the held-out split. The input string is `pi05-a5000-seed-check-v1|TT|II`, with zero-padded task and state IDs. Join the unique keys from the replay table to [`inference_seeds.csv`](inference_seeds.csv) to identify all 80 reused trials. The two tables let readers recompute the 380-state comparison with the pilot removed.
- [`step800_runtime.csv`](step800_runtime.csv) assigns each of the 400 scored held-out states to the evaluator batch before or after the kernel and driver update. Join by task and state ID to the 800-update held-out rows of [`episodes.csv`](episodes.csv). Assignments come from the two evaluator logs' acceptance events.
- [`audit_evidence.json`](audit_evidence.json) records checksums of the public supplemental CSVs, source manifests and acceptance-event logs, plus the action tolerance, archived pilot manifest modification time, first new-seed worker event, and batch time ranges.
