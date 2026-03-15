# Detailed Pipeline

This document expands the short repository README and records the execution structure used in the hydrophone experiments.

## Figure 1. End-to-End Hydrophone Execution Flow

```mermaid
flowchart TD
    A0[Research question:<br/>Can released U-FFIA audio models generalize to our hydrophone captures?]
    A0 --> A1

    subgraph Acquisition["Acquisition Layer"]
        A1[Hydrophone transducer in tank]
        A2[Behringer UMC22 analog front-end]
        A3[Capture host: Raspberry Pi / PC]
        A4[tools/record_hydrophone.py]
        A5[Raw WAV export<br/>48 kHz, PCM16, mono]
        A6[FFT spectrum PNG export]
        A1 --> A2 --> A3 --> A4 --> A5
        A4 --> A6
    end

    A5 --> B0

    subgraph QC["Signal Quality-Control Layer"]
        B0[Metadata validation]
        B1[Sample-rate / duration consistency]
        B2[Peak, RMS, crest factor]
        B3[Clipping count]
        B4[Zero-crossing and spectral descriptors]
        B5[Welch PSD estimation]
        B6[Low-band ratios:<br/>0-100 Hz, 0-300 Hz]
        B7[Hum-band ratios:<br/>45-55, 95-105, 145-155, 195-205 Hz]
        B8[Dominant low-frequency mode]
        B9[tools/analyze_hydrophone_raw.py]
        B0 --> B1 --> B2 --> B3 --> B4 --> B5 --> B6 --> B7 --> B8 --> B9
        B9 --> B10[raw_analysis/hydrophone_raw_metrics.csv]
        B9 --> B11[raw_analysis/hydrophone_raw_analysis.png]
        B9 --> B12[raw_analysis/hydrophone_rms_vs_lowband.png]
        B9 --> B13[raw_analysis/hydrophone_raw_analysis.md]
    end

    A5 --> C0

    subgraph Prep["Preprocessing And Adaptation Layer"]
        C0[Input waveform]
        C1[Mono loading]
        C2[Optional hydrophone preprocessing profile]
        C3[4th-order high-pass at 120 Hz]
        C4[Notch filtering at 50 / 100 / 150 / 200 Hz]
        C5[Optional unsupervised adaptation profile]
        C6[Mean removal]
        C7[RMS normalization toward target level]
        C8[Gain clipping by max dB constraint]
        C9[Resample to model rate]
        C10[Split into 2 s windows]
        C0 --> C1 --> C2 --> C3 --> C4 --> C5 --> C6 --> C7 --> C8 --> C9 --> C10
    end

    C10 --> D0

    subgraph Inference["Model Inference Layer"]
        D0[infer_audio_folder.py]
        D1[Load config/audio/pre_exp.yaml]
        D2[Load checkpoint]
        D3[Audio_Frontend mel pipeline]
        D4[Backbone branch]
        D5[MobileNetV2 released checkpoint]
        D6[PANNs CNN10 released checkpoint]
        D7[Window-level logits]
        D8[Softmax probabilities]
        D9[File-level probability average]
        D10[Predicted label:<br/>none / strong / medium / weak]
        D0 --> D1 --> D2 --> D3 --> D4
        D4 --> D5 --> D7
        D4 --> D6 --> D7
        D7 --> D8 --> D9 --> D10
    end

    D10 --> E0

    subgraph Comparison["Comparison And Reporting Layer"]
        E0[Run profile matrix]
        E1[raw]
        E2[hum_filtered]
        E3[hum_filtered + adaptation]
        E4[tools/plot_inference_results.py]
        E5[tools/compare_inference_runs.py]
        E6[Per-run summaries]
        E7[Cross-run comparison PNGs]
        E8[Cross-run comparison markdown]
        E0 --> E1
        E0 --> E2
        E0 --> E3
        E1 --> E4 --> E6
        E2 --> E4
        E3 --> E4
        E1 --> E5 --> E7
        E2 --> E5
        E3 --> E5
        E5 --> E8
    end

    B10 --> F0
    E6 --> F0
    E7 --> F0
    E8 --> F0

    subgraph Interpretation["Interpretation Layer"]
        F0[Joint review of raw-signal metrics and predicted labels]
        F1[Check whether hum removal changes labels]
        F2[Check whether adaptation reduces collapse]
        F3[Assess reliability of released checkpoints on hydrophone domain]
        F0 --> F1 --> F2 --> F3
    end
```

## Figure 2. Artifact Dependency Graph

```mermaid
flowchart LR
    A[Raw hydrophone WAVs] --> B[Raw analysis CSV]
    A --> C[Raw analysis PNGs]
    A --> D[MobileNet raw CSV]
    A --> E[MobileNet hum-filtered CSV]
    A --> F[MobileNet hum-filtered+adapted CSV]
    A --> G[PANNs raw CSV]
    A --> H[PANNs hum-filtered CSV]
    A --> I[PANNs hum-filtered+adapted CSV]
    D --> J[MobileNet comparison plot]
    E --> J
    F --> J
    G --> K[PANNs comparison plot]
    H --> K
    I --> K
    B --> L[docs/RESULTS.md tables]
    J --> L
    K --> L
    C --> L
```

## Execution Table

| Stage | Script / CLI | Main input | Main operations | Output artifacts |
| --- | --- | --- | --- | --- |
| Acquisition | `tools/record_hydrophone.py` | Live hydrophone stream | Record PCM16 wav, save FFT snapshot | raw `.wav`, spectrum `.png` |
| Raw analysis | `tools/analyze_hydrophone_raw.py` | Raw wav folder | Metadata, amplitude, Welch PSD, hum-band ratios, joined predictions | metrics `.csv`, analysis `.png`, summary `.md` |
| Preprocess only | `tools/preprocess_hydrophone_audio.py` | Raw wav folder | high-pass, notch, resample, chunking | processed `.wav` chunks |
| Inference | `infer_audio_folder.py` | Raw or processed wav folder | frontend, backbone, softmax, file aggregation | prediction `.csv` |
| Single-run plotting | `tools/plot_inference_results.py` | prediction `.csv` | probability heatmap, count plots | summary `.png` |
| Multi-run comparison | `tools/compare_inference_runs.py` | multiple prediction `.csv` files | cross-run count, class, confidence comparison | comparison `.png`, comparison `.md` |

## Run Profile Table

| Profile | Preprocess profile | Adaptation profile | Intended purpose |
| --- | --- | --- | --- |
| `raw` | `none` | `none` | Baseline run on untreated hydrophone recordings |
| `hum_filtered` | `hydrophone` | `none` | Test whether electrical hum suppression changes model outputs |
| `hum_filtered_adapted` | `hydrophone` | `hydrophone_v1` | Test whether simple unsupervised RMS adaptation reduces domain mismatch |

## Output Artifact Table

| Artifact group | Location |
| --- | --- |
| Raw analysis figures and tables | `results/hidrofon/raw_analysis/` |
| MobileNet comparison runs | `results/hidrofon/comparisons/mobilenet/` |
| PANNs comparison runs | `results/hidrofon/comparisons/panns/` |
| Short results overview | `docs/RESULTS.md` |
