# Reproduce the public analysis

[Project overview](../README.md) · [Protocol](experiment.md) · [Data dictionary](../results/README.md)

## Scope

This repository reproduces aggregate and per-task scores, the matched baseline, seed sensitivity, the original failures and their outcome flips, and the result figures without a GPU. To rerun policy evaluation itself, the full launcher and model weights are required separately.

## Check the evidence

Python 3.10 or later is sufficient for validation and paired counts. No third-party packages are needed.

```sh
git clone https://github.com/chaoshengsc/pi05-libero-finetuning.git
cd pi05-libero-finetuning
python scripts/analyze_results.py
```

Expected snapshot coverage:

```text
Verified 3,500 original-seed episodes in 14 complete groups and 1,600 additional-seed episodes.
```

The script verifies:

- Unique checkpoint/split/task/initial-state keys, with inference seed included for the extension.
- Exact membership of the frozen initial-state partition.
- Complete coverage for every published group.
- Equality between episode-derived counts and the public per-task CSV.
- Checksums and outcome labels for the original demos and all three failure comparisons.

The validator checks consistency of the published evidence. The original workstation logs are not all distributed, so source-record digests are provenance anchors rather than independent verification of the whole experiment.

## Pair two checkpoints

```sh
python scripts/analyze_results.py --pair 1750 2000 --split development
```

Expected outcomes are 93 shared successes, 3 successes unique to 1,750, 4 unique to 2,000, and 0 shared failures. The net difference is +1 percentage point.

Use `--split held-out` for the original 400-state comparison. The full report recomputes the strict 399-pair A0 result, paired bootstrap intervals, and the original 2,000-update failure list.

```sh
python scripts/analyze_results.py --report
```

The paired report conditions on the published weights, tasks and inference seeds. The observation exception used in the strict baseline analysis is recorded in the provenance manifest.

## Rebuild all figures

Use an isolated Python environment:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-viz.txt
python scripts/plot_results.py
python scripts/build_diagrams.py
```

The plotting script validates the data first and writes five result figures. The diagram script writes the method schematic and extracts original rollout frames using FFmpeg. All figures are exported as PNG, SVG, and PDF with a shared academic style. SVG fonts are converted to paths; PDF fonts are embedded. FFmpeg is required only for extracting rollout frames.

The main results figure includes the matched A0 baseline and all six held-out checkpoint scores. The three comparison recordings and their frame sequences are distributed as rendered media.

## Rebuild GIF previews

With FFmpeg available, this command preserves the source duration and samples the preview at 10 frames per second:

```sh
ffmpeg -i assets/demos/step800-task00-init02.mp4 \
  -filter_complex "fps=10,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer" \
  assets/demos/step800-task00-init02.gif
```

The GIF is a preview. The original MP4 remains unchanged and its checksum is recorded in the media manifest.

## Run the evidence-guard checks

```sh
python -m unittest discover -s tests
```

These checks inject duplicate episodes, incomplete groups, wrong-split states, unavailable held-out pairs, and mismatched pairing keys. Each must be rejected.
