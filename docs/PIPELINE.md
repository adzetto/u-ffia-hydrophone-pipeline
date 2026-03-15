# Detailed Pipeline

This document expands the short repository README and records the execution structure used across the hydrophone, local video, and fast late-fusion experiments.

## Figure 1. Hydrophone Audio Stabilization Pipeline

```mermaid
flowchart TD
    RQ[Research question<br/>Can released U-FFIA models be transferred to our hydrophone domain?]
    RQ --> ACQ0

    subgraph Acquisition["Acquisition Layer"]
        ACQ0[Hydrophone in tank]
        ACQ1[Analog front-end<br/>Behringer UMC22]
        ACQ2[Capture host<br/>Raspberry Pi or PC]
        ACQ3[tools/record_hydrophone.py]
        ACQ4[Raw WAV export<br/>48 kHz / PCM16 / mono / 10 s]
        ACQ5[Instant FFT preview PNG]
        ACQ0 --> ACQ1 --> ACQ2 --> ACQ3 --> ACQ4
        ACQ3 --> ACQ5
    end

    ACQ4 --> QC0
    ACQ4 --> PRE0

    subgraph RawQC["Raw-Signal QC Layer"]
        QC0[tools/analyze_hydrophone_raw.py]
        QC1[Read metadata]
        QC2[Duration and sample-rate checks]
        QC3[Peak / RMS / crest factor]
        QC4[Zero-crossing rate]
        QC5[Welch PSD]
        QC6[Low-band energy ratios<br/>0-100 Hz / 0-300 Hz]
        QC7[Hum-band ratios<br/>45-55 / 95-105 / 145-155 / 195-205 Hz]
        QC8[Dominant low-frequency mode]
        QC9[Raw analysis report]
        QC10[raw_analysis/hydrophone_raw_metrics.csv]
        QC11[raw_analysis/hydrophone_raw_analysis.png]
        QC12[raw_analysis/hydrophone_rms_vs_lowband.png]
        QC13[raw_analysis/hydrophone_raw_analysis.md]
        QC0 --> QC1 --> QC2 --> QC3 --> QC4 --> QC5 --> QC6 --> QC7 --> QC8 --> QC9
        QC9 --> QC10
        QC9 --> QC11
        QC9 --> QC12
        QC9 --> QC13
    end

    subgraph Preprocess["Hydrophone Preprocessing And Adaptation"]
        PRE0[Waveform load]
        PRE1[Mono read]
        PRE2[Optional hydrophone profile]
        PRE3[High-pass 120 Hz]
        PRE4[Notch filters 50 / 100 / 150 / 200 Hz]
        PRE5[Mean removal]
        PRE6[Optional hydrophone_v1 adaptation]
        PRE7[RMS target normalization]
        PRE8[Gain clamp by max dB]
        PRE9[Resample to 64 kHz]
        PRE10[Chunk into 2 s windows]
        PRE0 --> PRE1 --> PRE2 --> PRE3 --> PRE4 --> PRE5 --> PRE6 --> PRE7 --> PRE8 --> PRE9 --> PRE10
    end

    PRE10 --> INF0

    subgraph AudioInference["Released Audio Inference"]
        INF0[infer_audio_folder.py]
        INF1[Load config/audio/pre_exp.yaml or exp_binary_none_yem.yaml]
        INF2[Sanitize invalid fmax so Nyquist is respected]
        INF3[Load checkpoint]
        INF4[Audio_Frontend<br/>STFT -> logmel -> batchnorm]
        INF5[Backbone]
        INF6[MobileNetV2 branch]
        INF7[PANNs CNN10 branch]
        INF8[Window logits]
        INF9[Softmax probabilities]
        INF10[File-level probability average]
        INF11[Base prediction]
        INF0 --> INF1 --> INF2 --> INF3 --> INF4 --> INF5
        INF5 --> INF6 --> INF8
        INF5 --> INF7 --> INF8
        INF8 --> INF9 --> INF10 --> INF11
    end

    PRE10 --> BIN0
    INF11 --> BIN0

    subgraph Stabilization["Binary Stabilization Gate"]
        BIN0[tools/train_hydrophone_binary_adapter.py]
        BIN1[Weak supervision from local folders]
        BIN2[Feature bank<br/>MFCC + spectral descriptors]
        BIN3[Logistic adapter]
        BIN4[Cross-validation report]
        BIN5[Feeding-like probability]
        BIN6[Gate none vs feed]
        BIN7[Redistribute positive mass across weak/medium/strong]
        BIN1 --> BIN0 --> BIN2 --> BIN3 --> BIN4
        BIN3 --> BIN5 --> BIN6 --> BIN7
    end

    INF11 --> CMP0
    BIN7 --> CMP0
    QC10 --> CMP0

    subgraph Comparison["Comparison And Interpretation"]
        CMP0[tools/compare_inference_runs.py]
        CMP1[raw]
        CMP2[hum_filtered]
        CMP3[hum_filtered + adaptation]
        CMP4[hum_filtered + binary_gate]
        CMP5[hum_filtered + adaptation + binary_gate]
        CMP6[Per-run CSVs]
        CMP7[Summary PNGs]
        CMP8[Comparison markdown]
        CMP9[Interpretation step<br/>Are predictions stable under hum removal and adaptation?]
        CMP0 --> CMP1 --> CMP6
        CMP0 --> CMP2 --> CMP6
        CMP0 --> CMP3 --> CMP6
        CMP0 --> CMP4 --> CMP6
        CMP0 --> CMP5 --> CMP6
        CMP6 --> CMP7 --> CMP8 --> CMP9
    end
```

## Figure 2. Local Video Evaluation And Fast Late Fusion Pipeline

```mermaid
flowchart TD
    D0[Local paired dataset<br/>voice_none1/2, voice_yem1/2,<br/>video_none1/2, video_yem1/2]
    D0 --> D1
    D0 --> D8

    subgraph Pairing["Pairing And Split Assumption"]
        D1[Clip stems match 1:1]
        D2[parca_xx_timestamp.wav <-> parca_xx_timestamp.mp4]
        D3[Audio split already exists from binary audio fine-tune]
        D1 --> D2 --> D3
    end

    subgraph AudioBranch["Audio Branch"]
        D8[Binary audio fine-tune outputs]
        D9[train_predictions.csv]
        D10[val_predictions.csv]
        D11[test_predictions.csv]
        D8 --> D9
        D8 --> D10
        D8 --> D11
    end

    subgraph VideoBranch["Video Branch"]
        V0[infer_video_folder.py]
        V1[Decode mp4 with ffmpeg]
        V2[Resize 250]
        V3[Center crop 196]
        V4[Uniformly sample 20 frames]
        V5[Normalize to -1..1]
        V6[S3D released checkpoint]
        V7[4-way probabilities<br/>none / strong / medium / weak]
        V8[results/video_none1_results.csv]
        V9[results/video_none2_results.csv]
        V10[results/video_yem1_results.csv]
        V11[results/video_yem2_results.csv]
        V0 --> V1 --> V2 --> V3 --> V4 --> V5 --> V6 --> V7
        V7 --> V8
        V7 --> V9
        V7 --> V10
        V7 --> V11
    end

    D9 --> F0
    D10 --> F0
    D11 --> F0
    V8 --> F0
    V9 --> F0
    V10 --> F0
    V11 --> F0

    subgraph FastFusion["Fast Late Fusion"]
        F0[tools/train_fast_late_fusion.py]
        F1[Join audio and video scores by stem]
        F2[Feature engineering]
        F3[audio_none / audio_yem / audio_margin]
        F4[video_none / strong / medium / weak]
        F5[cross features<br/>feedlike, active, agreement]
        F6[Weak-positive weighting]
        F7[voice_yem2 weight = 0.8]
        F8[Candidate models]
        F9[logistic regression]
        F10[random forest]
        F11[gradient boosting]
        F12[Threshold tuning on validation]
        F13[Test-set evaluation]
        F14[best_model.joblib]
        F15[summary.json]
        F16[test_predictions.csv]
        F17[comparison.png]
        F0 --> F1 --> F2
        F2 --> F3
        F2 --> F4
        F2 --> F5
        F2 --> F6 --> F7
        F5 --> F8
        F7 --> F8
        F8 --> F9
        F8 --> F10
        F8 --> F11
        F9 --> F12
        F10 --> F12
        F11 --> F12
        F12 --> F13 --> F14
        F13 --> F15
        F13 --> F16
        F13 --> F17
    end
```

## Figure 3. Google Colab Execution Flow

```mermaid
flowchart TD
    C0[Open Google Colab with GPU runtime]
    C0 --> C1

    subgraph Session["Runtime Preparation"]
        C1[Mount Google Drive]
        C2[git clone repository]
        C3[apt install ffmpeg]
        C4[pip install Python dependencies]
        C5[Expose pretrained_models]
        C6[Expose PROJE1 dataset]
        C1 --> C2 --> C3 --> C4 --> C5 --> C6
    end

    C6 --> C7

    subgraph Decision["Execution Decision"]
        C7{Which path?}
        C7 -->|Hydrophone-only| C8
        C7 -->|Paired audio+video| C9
    end

    subgraph HydrophoneOnly["Colab Hydrophone Path"]
        C8[Run tools/finetune_binary_audio.py]
        C8a[Write audio_binary_best.pt]
        C8b[Write train/val/test prediction CSVs]
        C8c[Run infer_audio_folder.py on PROJE1/hidrofon]
        C8d[Write hidrofon_binary_hydrophone.csv]
        C8 --> C8a
        C8 --> C8b
        C8a --> C8c --> C8d
    end

    subgraph PairedAV["Colab Local AV Path"]
        C9[Run infer_video_folder.py on 4 video folders]
        C9a[Write results/video_*_results.csv]
        C9b[Run tools/train_fast_late_fusion.py --source-weight voice_yem2=0.8]
        C9c[Write fusion summary, best model, comparison plot]
        C9 --> C9a --> C9b --> C9c
    end

    C8d --> C10
    C9c --> C10

    subgraph Export["Export And Review"]
        C10[Inspect CSVs, PNGs, and JSON reports]
        C11[Copy final artifacts back to Google Drive]
        C10 --> C11
    end
```

## Figure 4. Artifact Dependency Graph

```mermaid
flowchart LR
    A[Raw hydrophone WAVs] --> B[raw_analysis metrics CSV]
    A --> C[raw_analysis PNGs]
    A --> D[audio inference CSVs]
    D --> E[stabilized audio comparison PNGs]
    D --> F[docs/RESULTS.md]
    B --> F
    C --> F

    G[Binary audio fine-tune] --> H[train_predictions.csv]
    G --> I[val_predictions.csv]
    G --> J[test_predictions.csv]
    K[Released video inference] --> L[video_none1_results.csv]
    K --> M[video_none2_results.csv]
    K --> N[video_yem1_results.csv]
    K --> O[video_yem2_results.csv]
    H --> P[fast late fusion]
    I --> P
    J --> P
    L --> P
    M --> P
    N --> P
    O --> P
    P --> Q[results/fusion_fast/summary.json]
    P --> R[results/fusion_fast/comparison.png]
    P --> S[results/fusion_fast/test_predictions.csv]
    Q --> T[docs/FAST_FUSION.md]
    R --> T
    S --> T
```

## Execution Table

| Stage | Script / CLI | Main input | Main operations | Output artifacts |
| --- | --- | --- | --- | --- |
| Acquisition | `tools/record_hydrophone.py` | Live hydrophone stream | Record PCM16 wav, save FFT snapshot | raw `.wav`, spectrum `.png` |
| Raw analysis | `tools/analyze_hydrophone_raw.py` | Raw wav folder | Metadata, RMS, PSD, hum-band ratios, signal report | metrics `.csv`, analysis `.png`, summary `.md` |
| Hydrophone preprocessing | `tools/preprocess_hydrophone_audio.py` | Raw wav folder | high-pass, notch, resample, chunking | processed `.wav` chunks |
| Audio inference | `infer_audio_folder.py` | Raw or processed wav folder | frontend, Nyquist-safe mel band sanitization, backbone, softmax, file aggregation | prediction `.csv` |
| Binary gate training | `tools/train_hydrophone_binary_adapter.py` | weakly labeled local folders | MFCC and spectral features, logistic regression, CV evaluation | adapter `.joblib`, report `.md`, report `.png` |
| Stabilized inference | `infer_audio_folder.py --binary-adapter-model ... --stabilization-profile binary_gate` | wav folder + adapter | base model plus binary feeding gate | stabilized prediction `.csv` |
| Video inference | `infer_video_folder.py` | local `.mp4` folders | ffmpeg decode, resize, crop, frame sampling, S3D inference | `video_*_results.csv` |
| Fast late fusion | `tools/train_fast_late_fusion.py` | audio prediction CSVs + video prediction CSVs | join by stem, feature engineering, source weighting, model selection, threshold tuning | best model `.joblib`, summary `.json`, predictions `.csv`, comparison `.png` |
| End-to-end AV fine-tune | `tools/finetune_binary_av.py` | paired local audio+video clips | slow direct AV training with released initialization | AV checkpoint and metrics |

## Run Profile Table

| Profile | Preprocess profile | Adaptation profile | Intended purpose |
| --- | --- | --- | --- |
| `raw` | `none` | `none` | Baseline run on untreated hydrophone recordings |
| `hum_filtered` | `hydrophone` | `none` | Test whether electrical hum suppression changes model outputs |
| `hum_filtered_adapted` | `hydrophone` | `hydrophone_v1` | Test whether simple unsupervised RMS adaptation reduces domain mismatch |
| `hum_filtered_gated` | `hydrophone` | `none` | Use local binary gate to suppress unstable nonfeeding false positives |
| `hum_filtered_adapted_gated` | `hydrophone` | `hydrophone_v1` | Stabilized path combining local gate with amplitude adaptation |
| `video_released_eval` | N/A | N/A | Measure transferred video-only behaviour on local clips |
| `fast_late_fusion` | audio hydrophone profile optional | audio hydrophone_v1 optional | Combine local audio and video evidence with a cheap secondary model |

## Colab-Oriented Execution Table

| Colab step | Main command | Expected duration | Main artifact |
| --- | --- | --- | --- |
| Environment bootstrap | `apt-get install ffmpeg` and `pip install ...` | short | ready runtime |
| Audio fine-tune | `python tools/finetune_binary_audio.py ...` | medium | `audio_binary_best.pt` |
| Hydrophone inference | `python infer_audio_folder.py ...` | short | hydrophone prediction CSV |
| Video released inference | `python infer_video_folder.py ...` on 4 folders | medium to long | `results/video_*_results.csv` |
| Fast late fusion | `python tools/train_fast_late_fusion.py --source-weight voice_yem2=0.8` | very short | `results/fusion_fast/*` |

## Output Artifact Table

| Artifact group | Location |
| --- | --- |
| Raw analysis figures and tables | `results/hidrofon/raw_analysis/` |
| Binary adapter artifacts | `results/adapter/` |
| MobileNet comparison runs | `results/hidrofon/comparisons/mobilenet/` |
| PANNs comparison runs | `results/hidrofon/comparisons/panns/` |
| Binary audio fine-tune outputs | `results/finetune_binary_none_yem/` |
| Video evaluation outputs | `results/video_*_results.csv` and `results/video_eval/` |
| Fast late-fusion outputs | `results/fusion_fast/` |
| Short results overview | `docs/RESULTS.md` |
| Fast fusion detail | `docs/FAST_FUSION.md` |
| Colab guide | `docs/COLAB.md` |
