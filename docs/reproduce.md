# Reproduce the public analysis

[Project overview](../README.md) · [Protocol](experiment.md) · [Data dictionary](../results/README.md)

## Scope

This repository can reproduce the published aggregate scores, per-task tables, development pairing, and figures without a GPU. It does not yet include the complete training/evaluation launcher or distribute model weights. The recorded configurations describe the completed experiment; they are not drop-in OpenPI training commands.

## Check the evidence

Python 3.10 or later is sufficient for validation and paired counts. No third-party packages are needed.

```sh
git clone https://github.com/chaoshengsc/pi05-libero-finetuning.git
cd pi05-libero-finetuning
python scripts/analyze_results.py
```

Expected snapshot coverage:

```text
Verified 1,100 episodes in 8 complete splits.
```

The script verifies:

- Unique checkpoint/split/task/initial-state keys.
- Exact membership of the frozen initial-state partition.
- Complete coverage for every published group.
- Equality between episode-derived counts and the public per-task CSV.
- Checksums, demo labels, and original MP4 integrity.

The validator checks consistency of the published evidence. The original workstation logs are not all distributed, so source-record digests are provenance anchors rather than independent verification of the whole experiment.

## Pair two checkpoints

```sh
python scripts/analyze_results.py --pair 1750 2000 --split development
```

Expected outcomes are 93 shared successes, 3 successes unique to 1,750, 4 unique to 2,000, and 0 shared failures. The net difference is +1 percentage point.

A request for an unpublished complete pair, such as the current 1,750/2,000 held-out comparison, exits with an error. Missing results are never treated as failures or zero scores.

## Rebuild all figures

Use an isolated Python environment:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-viz.txt
python scripts/plot_results.py
python scripts/build_diagrams.py
```

The plotting script validates the data first. It writes both PNG and SVG versions of four figures. The diagram script writes the cover and experiment diagram. Fonts are converted to paths in SVG exports for consistent appearance.

The present figures are authored for this snapshot, including the 800-update held-out result. When publishing further completed checkpoints, update the figures and accompanying narrative together.

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
