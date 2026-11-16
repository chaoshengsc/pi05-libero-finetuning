# Recorded policy episodes

[Project overview](../README.md) · [Episode data](../results/episodes.csv) · [Media manifest](../assets/demos/manifest.json)

All clips use the **800-update checkpoint** on the **held-out initial-state split**. They are original simulator recordings from the scored evaluation, showing the external camera on the left and wrist camera on the right. GIF previews run at approximately original speed.

## Task 00 — between the plate and the ramekin

Instruction: “pick up the black bowl between the plate and the ramekin and place it on the plate”.

| Initial state | Recorded result | Policy steps | Original |
|:--|:--|--:|:--|
| 02 | Success | 78 | [MP4](../assets/demos/step800-task00-init02.mp4) |
| 16 | Unsolved at the step limit | 220 | [MP4](../assets/demos/step800-task00-init16.mp4) |

![Successful task 00 episode](../assets/demos/step800-task00-init02.gif)

![Unsuccessful task 00 episode](../assets/demos/step800-task00-init16.gif)

## Task 09 — on the wooden cabinet

Instruction: “pick up the black bowl on the wooden cabinet and place it on the plate”.

Initial state **00**, success in **131** policy steps. [Original MP4](../assets/demos/step800-task09-init00.mp4)

![Successful task 09 episode](../assets/demos/step800-task09-init00.gif)

## Selection and interpretation

For task 00, these are the first valid success and first valid failure in ascending initial-state order. For task 09, this is the first valid success by the same ordering. The selection rule was applied to existing recordings after evaluation.

These examples illustrate behavior and include a failure. They are not randomly sampled estimates of overall performance, a comparison between checkpoints, or a demonstration of an added verifier. Success labels are taken from the native task predicate in the final episode record.

The [manifest](../assets/demos/manifest.json) includes the checkpoint, task, initial state, outcome, step count and SHA-256 of each original recording.
