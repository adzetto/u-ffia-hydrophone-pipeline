# Google Colab

This repository can be run on Google Colab in two practical modes:

- Hydrophone-only audio path
- Local paired audio+video path with fast late fusion

Use a GPU runtime in Colab:

- `Runtime -> Change runtime type -> T4 GPU` or better

## 1. Environment Setup

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
cd /content
git clone https://github.com/adzetto/u-ffia-hydrophone-pipeline.git
cd u-ffia-hydrophone-pipeline
apt-get -qq update
apt-get -qq install -y ffmpeg
pip install -q librosa matplotlib numpy scipy scikit-learn joblib omegaconf soundfile torchlibrosa
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 2. Expected Folder Layout

Store your data and released weights on Google Drive, then either symlink or copy them into the repo session.

Minimal expected layout:

```text
/content/drive/MyDrive/omar_data/
  pretrained_models/
    audio-visual_pretrainedmodel/
      audio_best.pt
      video_best.pt
  PROJE1/
    hidrofon/
    voice_none1/
    voice_none2/
    voice_yem1/
    voice_yem2/
    video_none1/
    video_none2/
    video_yem1/
    video_yem2/
```

Optional symlink setup:

```bash
ln -s /content/drive/MyDrive/omar_data/pretrained_models pretrained_models
ln -s /content/drive/MyDrive/omar_data/PROJE1 PROJE1
```

## 3. Hydrophone-Only Path

Train the binary audio model:

```bash
python tools/finetune_binary_audio.py \
  --negative-dir PROJE1/voice_none1 \
  --negative-dir PROJE1/voice_none2 \
  --positive-dir PROJE1/voice_yem1 \
  --positive-dir PROJE1/voice_yem2 \
  --output-dir results/finetune_binary_none_yem/hydrophone_source_e8_thr \
  --epochs 8 \
  --batch-size 32 \
  --device cuda \
  --preprocess-profile hydrophone \
  --adaptation-profile hydrophone_v1
```

Run hydrophone inference:

```bash
python infer_audio_folder.py PROJE1/hidrofon \
  --config config/audio/exp_binary_none_yem.yaml \
  --checkpoint results/finetune_binary_none_yem/hydrophone_source_e8_thr/audio_binary_best.pt \
  --preprocess-profile hydrophone \
  --adaptation-profile hydrophone_v1 \
  --output-csv results/finetune_binary_none_yem/hydrophone_source_e8_thr/hidrofon_binary_hydrophone.csv
```

## 4. Local Paired Audio+Video Path

Step 1. Generate released video predictions:

```bash
python infer_video_folder.py PROJE1/video_none1 --output-csv results/video_none1_results.csv --device cuda
python infer_video_folder.py PROJE1/video_none2 --output-csv results/video_none2_results.csv --device cuda
python infer_video_folder.py PROJE1/video_yem1  --output-csv results/video_yem1_results.csv  --device cuda
python infer_video_folder.py PROJE1/video_yem2  --output-csv results/video_yem2_results.csv  --device cuda
```

Step 2. Train the fast late-fusion model:

```bash
python tools/train_fast_late_fusion.py --source-weight voice_yem2=0.8
```

This uses the already trained binary audio model predictions from:

- `results/finetune_binary_none_yem/hydrophone_source_e8_thr/train_predictions.csv`
- `results/finetune_binary_none_yem/hydrophone_source_e8_thr/val_predictions.csv`
- `results/finetune_binary_none_yem/hydrophone_source_e8_thr/test_predictions.csv`

## 5. Why `voice_yem2=0.8`

The local note for `yem2` is:

- feeding exists, but it is weaker than `yem1`
- fish ate less
- feed was given later

So `yem2` is treated as a weaker positive source during late fusion.

## 6. Key Colab Outputs

| Artifact | Meaning |
| --- | --- |
| `results/finetune_binary_none_yem/.../metrics.json` | Binary audio train/val/test metrics |
| `results/finetune_binary_none_yem/.../hidrofon_binary_hydrophone.csv` | Hydrophone file-level predictions |
| `results/video_*_results.csv` | Released video model outputs on local clips |
| `results/fusion_fast/summary.json` | Final fast late-fusion leaderboard and best model |
| `results/fusion_fast/comparison.png` | Audio vs video vs fusion metric comparison |

## 7. Practical Notes

- `infer_video_folder.py` is much slower than the fast late-fusion training step.
- Full end-to-end AV fine-tuning is available in `tools/finetune_binary_av.py`, but it is significantly slower than the late-fusion route.
- The most time-efficient Colab workflow is:
  1. train audio binary model
  2. run released video inference once
  3. train fast late fusion
