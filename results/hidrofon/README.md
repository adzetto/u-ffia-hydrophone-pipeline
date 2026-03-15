# Hydrophone Results

This folder contains the current inference plots generated from our hydrophone recordings.

## Data Source

- Our hydrophone data: https://drive.google.com/drive/folders/1qVZvUsLJxGaPP1cPbR4LjEqP2VsjgzeT?usp=drive_link

## Included Plots

- `mobilenet_hydrophone_summary.png`
  - Summary plot for the released MobileNetV2-based audio checkpoint
- `panns_hydrophone_summary.png`
  - Summary plot for the released PANNs CNN10 checkpoint
- `hydrophone_model_comparison.png`
  - Side-by-side comparison of the two checkpoints on the same hydrophone files

## Short Interpretation

- MobileNetV2 output:
  - `11` files predicted as `weak`
  - `2` files predicted as `strong`
  - `1` file predicted as `none`
  - `1` file predicted as `medium`
- PANNs CNN10 output:
  - `15` files predicted as `strong`

These outputs are useful as a reproducible baseline, but they should not yet be treated as validated scientific results for the hydrophone domain.
