# Hydrophone Raw Data Analysis

## Dataset Summary

- Files analyzed: 15
- Sample rate set: [48000]
- Channel count set: [1]
- Duration set: [10.0]
- Mean RMS: 0.014437
- Median RMS: 0.000301
- Max peak amplitude: 0.221863
- Files with RMS < 0.001: 10
- Files with strong 50 Hz band ratio >= 0.1: 5

## Main Findings

- The raw hydrophone files are structurally clean: consistent length, consistent sample rate, and no clipping spikes at full scale.
- A large portion of the recordings are very low energy, which increases the risk of unstable model behavior on domain-shifted data.
- Low-frequency energy is substantial in several files, especially the later captures.
- The most common dominant low-frequency pattern is: sub-10Hz drift/noise=10, ~50Hz hum=5
- The strongest likely electrical-hum files are:
  - kayit_20260313_052420.wav: 45-55 Hz ratio 0.369, 145-155 Hz ratio 0.090, dominant low frequency 52.73 Hz
  - kayit_20260313_052517.wav: 45-55 Hz ratio 0.383, 145-155 Hz ratio 0.080, dominant low frequency 52.73 Hz
  - kayit_20260313_052609.wav: 45-55 Hz ratio 0.324, 145-155 Hz ratio 0.118, dominant low frequency 52.73 Hz
  - kayit_20260313_052719.wav: 45-55 Hz ratio 0.348, 145-155 Hz ratio 0.108, dominant low frequency 52.73 Hz
  - kayit_20260313_052822.wav: 45-55 Hz ratio 0.400, 145-155 Hz ratio 0.068, dominant low frequency 52.73 Hz

## Joined Model Outputs

- MobileNetV2 prediction counts: {'weak': 11, 'strong': 2, 'none': 1, 'medium': 1}
- PANNs CNN10 prediction counts: {'strong': 15}
- These model outputs are included as joined columns in the CSV for easier correlation with the raw-data metrics.

## Output Files

- `hydrophone_raw_metrics.csv`: per-file numerical metrics
- `hydrophone_raw_analysis.png`: multi-panel summary figure
- `hydrophone_rms_vs_lowband.png`: RMS vs low-frequency energy scatter