# U-FFIA Hydrophone Pipeline

Hydrophone-oriented evaluation pipeline built on top of U-FFIA. This repo focuses on three things: raw hydrophone analysis, inference on the released audio checkpoints, and comparison of `raw`, `hum_filtered`, and `hum_filtered + adaptation` runs.

## Links

- Repository: https://github.com/adzetto/u-ffia-hydrophone-pipeline
- Paper: https://arxiv.org/abs/2309.05058
- Original U-FFIA repo: https://github.com/FishMaster93/U-FFIA
- Released weights: https://drive.google.com/drive/folders/1fh-Lo3S7-aTgfPni5-IeG5_-P7MBKBfL?usp=drive_link
- Our hydrophone data: https://drive.google.com/drive/folders/1qVZvUsLJxGaPP1cPbR4LjEqP2VsjgzeT?usp=drive_link

## Results Snapshot

Raw hydrophone analysis:

![Raw Hydrophone Analysis](results/hidrofon/raw_analysis/hydrophone_raw_analysis.png)

MobileNetV2 run comparison:

![MobileNetV2 Comparison](results/hidrofon/comparisons/mobilenet/comparison.png)

PANNs CNN10 run comparison:

![PANNs Comparison](results/hidrofon/comparisons/panns/comparison.png)

## Short Findings

| Item | Result |
| --- | --- |
| Raw-data structure | `15` files, all `48 kHz`, mono, `10 s`, no clipping |
| Raw-data issue | `10/15` files are very low energy, `5/15` are strongly `50 Hz` hum-heavy |
| MobileNetV2 | Changes under filtering/adaptation, but still unstable on hydrophone data |
| PANNs CNN10 | Collapses to `strong` for every tested file in all run modes |

## Docs

- Detailed pipeline and execution tables: [docs/PIPELINE.md](docs/PIPELINE.md)
- Detailed results, markdown tables, and plot references: [docs/RESULTS.md](docs/RESULTS.md)

## Quick Start

```bash
python infer_audio_folder.py path/to/audio_folder --output-csv results.csv
python infer_audio_folder.py path/to/audio_folder --preprocess-profile hydrophone --output-csv hum_filtered.csv
python infer_audio_folder.py path/to/audio_folder --preprocess-profile hydrophone --adaptation-profile hydrophone_v1 --output-csv hum_filtered_adapted.csv
```
