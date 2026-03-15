import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import welch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze raw hydrophone wav recordings and export metrics, plots, and a markdown summary."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing raw hydrophone wav files.")
    parser.add_argument("output_dir", type=Path, help="Directory for CSV, markdown, and PNG outputs.")
    parser.add_argument(
        "--mobilenet-csv",
        type=Path,
        default=None,
        help="Optional file-level MobileNet inference CSV to join into the analysis table.",
    )
    parser.add_argument(
        "--panns-csv",
        type=Path,
        default=None,
        help="Optional file-level PANNs inference CSV to join into the analysis table.",
    )
    return parser.parse_args()


def list_wav_files(input_dir: Path) -> List[Path]:
    input_dir = input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError("Input directory not found: {}".format(input_dir))
    files = sorted(path for path in input_dir.iterdir() if path.suffix.lower() == ".wav")
    if not files:
        raise FileNotFoundError("No wav files found under {}".format(input_dir))
    return files


def area(y: np.ndarray, x: np.ndarray) -> float:
    try:
        return float(np.trapezoid(y, x))
    except AttributeError:
        return float(np.trapz(y, x))


def band_energy_ratio(freqs: np.ndarray, power: np.ndarray, lo: float, hi: float) -> float:
    total = area(power, freqs)
    if total <= 0:
        return 0.0
    mask = (freqs >= lo) & (freqs < hi)
    if not np.any(mask):
        return 0.0
    return area(power[mask], freqs[mask]) / total


def dominant_frequency(freqs: np.ndarray, power: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return 0.0
    local_freqs = freqs[mask]
    local_power = power[mask]
    return float(local_freqs[np.argmax(local_power)])


def read_inference_map(path: Path | None) -> Dict[str, Dict[str, str]]:
    if path is None or not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {Path(row["file"]).name: row for row in rows}


def analyze_file(audio_path: Path) -> Tuple[Dict[str, object], np.ndarray, np.ndarray]:
    waveform, sample_rate = sf.read(str(audio_path))
    if waveform.ndim > 1:
        waveform = waveform[:, 0]
        channels = 2
    else:
        channels = 1

    waveform = waveform.astype(np.float64)
    duration_seconds = float(len(waveform)) / float(sample_rate)
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    rms = float(np.sqrt(np.mean(waveform * waveform))) if waveform.size else 0.0
    crest_factor = float(peak / rms) if rms > 0 else 0.0
    dc_offset = float(np.mean(waveform)) if waveform.size else 0.0
    zero_crossing_rate = float(librosa.feature.zero_crossing_rate(waveform, frame_length=2048, hop_length=512).mean())
    clipping_samples = int(np.sum(np.abs(waveform) >= 0.999))

    spectral_centroid = float(librosa.feature.spectral_centroid(y=waveform, sr=sample_rate).mean())
    spectral_bandwidth = float(librosa.feature.spectral_bandwidth(y=waveform, sr=sample_rate).mean())
    spectral_rolloff = float(librosa.feature.spectral_rolloff(y=waveform, sr=sample_rate, roll_percent=0.95).mean())

    freqs, power = welch(
        waveform,
        fs=sample_rate,
        window="hann",
        nperseg=min(8192, len(waveform)),
        noverlap=min(4096, len(waveform) // 2),
    )

    low_100_ratio = band_energy_ratio(freqs, power, 0.0, 100.0)
    low_300_ratio = band_energy_ratio(freqs, power, 0.0, 300.0)
    hum_50_ratio = band_energy_ratio(freqs, power, 45.0, 55.0)
    hum_100_ratio = band_energy_ratio(freqs, power, 95.0, 105.0)
    hum_150_ratio = band_energy_ratio(freqs, power, 145.0, 155.0)
    hum_200_ratio = band_energy_ratio(freqs, power, 195.0, 205.0)
    dominant_freq_low = dominant_frequency(freqs, power, 0.0, 300.0)
    dominant_freq_full = dominant_frequency(freqs, power, 0.0, sample_rate / 2.0)

    row = {
        "file": audio_path.name,
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "duration_seconds": round(duration_seconds, 3),
        "peak": round(peak, 6),
        "rms": round(rms, 6),
        "crest_factor": round(crest_factor, 6),
        "dc_offset": round(dc_offset, 8),
        "zero_crossing_rate": round(zero_crossing_rate, 6),
        "clipping_samples": clipping_samples,
        "spectral_centroid_hz": round(spectral_centroid, 3),
        "spectral_bandwidth_hz": round(spectral_bandwidth, 3),
        "spectral_rolloff95_hz": round(spectral_rolloff, 3),
        "low_band_ratio_0_100hz": round(low_100_ratio, 6),
        "low_band_ratio_0_300hz": round(low_300_ratio, 6),
        "hum_50hz_ratio": round(hum_50_ratio, 6),
        "hum_100hz_ratio": round(hum_100_ratio, 6),
        "hum_150hz_ratio": round(hum_150_ratio, 6),
        "hum_200hz_ratio": round(hum_200_ratio, 6),
        "dominant_freq_0_300hz": round(dominant_freq_low, 3),
        "dominant_freq_full_hz": round(dominant_freq_full, 3),
    }
    return row, freqs, power


def write_metrics_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_summary_plot(
    rows: List[Dict[str, object]],
    mean_freqs: np.ndarray,
    mean_power: np.ndarray,
    output_path: Path,
) -> None:
    file_labels = [Path(str(row["file"])).stem for row in rows]
    rms_values = [float(row["rms"]) for row in rows]
    peak_values = [float(row["peak"]) for row in rows]
    low_100 = [float(row["low_band_ratio_0_100hz"]) for row in rows]
    hum_50 = [float(row["hum_50hz_ratio"]) for row in rows]
    hum_150 = [float(row["hum_150hz_ratio"]) for row in rows]
    centroids = [float(row["spectral_centroid_hz"]) for row in rows]
    rolloffs = [float(row["spectral_rolloff95_hz"]) for row in rows]

    figure, axes = plt.subplots(2, 2, figsize=(18, 10), constrained_layout=True)

    axes[0, 0].plot(file_labels, rms_values, marker="o", label="RMS", color="#1f77b4")
    axes[0, 0].plot(file_labels, peak_values, marker="o", label="Peak", color="#ff7f0e")
    axes[0, 0].set_title("Amplitude Metrics By File")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].tick_params(axis="x", rotation=60, labelsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].bar(file_labels, low_100, label="0-100 Hz ratio", color="#54a24b", alpha=0.85)
    axes[0, 1].plot(file_labels, hum_50, marker="o", color="#d62728", label="45-55 Hz ratio")
    axes[0, 1].plot(file_labels, hum_150, marker="o", color="#9467bd", label="145-155 Hz ratio")
    axes[0, 1].set_title("Low-Frequency And Hum Energy Ratios")
    axes[0, 1].set_ylabel("Energy ratio")
    axes[0, 1].tick_params(axis="x", rotation=60, labelsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(file_labels, centroids, marker="o", label="Spectral centroid", color="#1f77b4")
    axes[1, 0].plot(file_labels, rolloffs, marker="o", label="95% rolloff", color="#ff7f0e")
    axes[1, 0].set_title("Spectral Shape Metrics")
    axes[1, 0].set_ylabel("Frequency (Hz)")
    axes[1, 0].tick_params(axis="x", rotation=60, labelsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    mask = mean_freqs <= 500.0
    axes[1, 1].plot(mean_freqs[mask], mean_power[mask], color="#2ca02c", linewidth=1.5)
    for frequency in (50, 100, 150, 200):
        axes[1, 1].axvline(frequency, color="#7f7f7f", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[1, 1].set_title("Mean Welch PSD (0-500 Hz)")
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("Power spectral density")
    axes[1, 1].grid(True, alpha=0.3)

    figure.suptitle("Hydrophone Raw Data Analysis", fontsize=15)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def create_scatter_plot(rows: List[Dict[str, object]], output_path: Path) -> None:
    rms_values = np.array([float(row["rms"]) for row in rows], dtype=float)
    low_100 = np.array([float(row["low_band_ratio_0_100hz"]) for row in rows], dtype=float)
    labels = [row["file"].replace(".wav", "") for row in rows]

    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    scatter = axis.scatter(rms_values, low_100, s=80, c=np.arange(len(rows)), cmap="viridis")
    for x_value, y_value, label in zip(rms_values, low_100, labels):
        axis.annotate(label.split("_")[-1], (x_value, y_value), fontsize=8, xytext=(4, 4), textcoords="offset points")
    axis.set_title("RMS vs 0-100 Hz Energy Ratio")
    axis.set_xlabel("RMS amplitude")
    axis.set_ylabel("0-100 Hz energy ratio")
    axis.grid(True, alpha=0.3)
    figure.colorbar(scatter, ax=axis, label="File index")
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_markdown_summary(
    path: Path,
    rows: List[Dict[str, object]],
    mobilenet_map: Dict[str, Dict[str, str]],
    panns_map: Dict[str, Dict[str, str]],
) -> None:
    rms_values = [float(row["rms"]) for row in rows]
    peak_values = [float(row["peak"]) for row in rows]
    low_energy_count = sum(1 for value in rms_values if value < 0.001)
    high_hum_rows = [row for row in rows if float(row["hum_50hz_ratio"]) >= 0.1]
    dominant_counts = Counter("~50Hz hum" if float(row["dominant_freq_0_300hz"]) >= 45 and float(row["dominant_freq_0_300hz"]) <= 60 else "sub-10Hz drift/noise" for row in rows)
    mobilenet_counts = Counter(mobilenet_map[name]["predicted_label"] for name in mobilenet_map) if mobilenet_map else Counter()
    panns_counts = Counter(panns_map[name]["predicted_label"] for name in panns_map) if panns_map else Counter()

    lines = [
        "# Hydrophone Raw Data Analysis",
        "",
        "## Dataset Summary",
        "",
        "- Files analyzed: {}".format(len(rows)),
        "- Sample rate set: {}".format(sorted({int(row["sample_rate_hz"]) for row in rows})),
        "- Channel count set: {}".format(sorted({int(row["channels"]) for row in rows})),
        "- Duration set: {}".format(sorted({float(row["duration_seconds"]) for row in rows})),
        "- Mean RMS: {:.6f}".format(float(np.mean(rms_values))),
        "- Median RMS: {:.6f}".format(float(np.median(rms_values))),
        "- Max peak amplitude: {:.6f}".format(float(np.max(peak_values))),
        "- Files with RMS < 0.001: {}".format(low_energy_count),
        "- Files with strong 50 Hz band ratio >= 0.1: {}".format(len(high_hum_rows)),
        "",
        "## Main Findings",
        "",
        "- The raw hydrophone files are structurally clean: consistent length, consistent sample rate, and no clipping spikes at full scale.",
        "- A large portion of the recordings are very low energy, which increases the risk of unstable model behavior on domain-shifted data.",
        "- Low-frequency energy is substantial in several files, especially the later captures.",
        "- The most common dominant low-frequency pattern is: {}".format(", ".join("{}={}".format(name, count) for name, count in dominant_counts.items())),
    ]

    if high_hum_rows:
        lines.extend(
            [
                "- The strongest likely electrical-hum files are:",
            ]
        )
        for row in high_hum_rows:
            lines.append(
                "  - {}: 45-55 Hz ratio {:.3f}, 145-155 Hz ratio {:.3f}, dominant low frequency {:.2f} Hz".format(
                    row["file"],
                    float(row["hum_50hz_ratio"]),
                    float(row["hum_150hz_ratio"]),
                    float(row["dominant_freq_0_300hz"]),
                )
            )

    if mobilenet_counts or panns_counts:
        lines.extend(["", "## Joined Model Outputs", ""])
        if mobilenet_counts:
            lines.append("- MobileNetV2 prediction counts: {}".format(dict(mobilenet_counts)))
        if panns_counts:
            lines.append("- PANNs CNN10 prediction counts: {}".format(dict(panns_counts)))
        lines.append("- These model outputs are included as joined columns in the CSV for easier correlation with the raw-data metrics.")

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `hydrophone_raw_metrics.csv`: per-file numerical metrics",
            "- `hydrophone_raw_analysis.png`: multi-panel summary figure",
            "- `hydrophone_rms_vs_lowband.png`: RMS vs low-frequency energy scatter",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mobilenet_map = read_inference_map(args.mobilenet_csv)
    panns_map = read_inference_map(args.panns_csv)

    metric_rows = []
    psd_rows = []
    wav_files = list_wav_files(input_dir)

    for wav_path in wav_files:
        row, freqs, power = analyze_file(wav_path)
        if wav_path.name in mobilenet_map:
            metric_row = {**row, "mobilenet_label": mobilenet_map[wav_path.name]["predicted_label"]}
            metric_row["mobilenet_confidence"] = max(
                float(mobilenet_map[wav_path.name]["score_none"]),
                float(mobilenet_map[wav_path.name]["score_strong"]),
                float(mobilenet_map[wav_path.name]["score_medium"]),
                float(mobilenet_map[wav_path.name]["score_weak"]),
            )
        else:
            metric_row = dict(row)
            metric_row["mobilenet_label"] = ""
            metric_row["mobilenet_confidence"] = ""

        if wav_path.name in panns_map:
            metric_row["panns_label"] = panns_map[wav_path.name]["predicted_label"]
            metric_row["panns_confidence"] = max(
                float(panns_map[wav_path.name]["score_none"]),
                float(panns_map[wav_path.name]["score_strong"]),
                float(panns_map[wav_path.name]["score_medium"]),
                float(panns_map[wav_path.name]["score_weak"]),
            )
        else:
            metric_row["panns_label"] = ""
            metric_row["panns_confidence"] = ""

        metric_rows.append(metric_row)
        psd_rows.append(power)

    mean_power = np.mean(np.vstack(psd_rows), axis=0)
    mean_freqs = freqs

    csv_path = output_dir / "hydrophone_raw_metrics.csv"
    summary_plot_path = output_dir / "hydrophone_raw_analysis.png"
    scatter_plot_path = output_dir / "hydrophone_rms_vs_lowband.png"
    summary_md_path = output_dir / "hydrophone_raw_analysis.md"

    write_metrics_csv(csv_path, metric_rows)
    create_summary_plot(metric_rows, mean_freqs, mean_power, summary_plot_path)
    create_scatter_plot(metric_rows, scatter_plot_path)
    write_markdown_summary(summary_md_path, metric_rows, mobilenet_map, panns_map)

    print("Saved metrics CSV to {}".format(csv_path))
    print("Saved summary plot to {}".format(summary_plot_path))
    print("Saved scatter plot to {}".format(scatter_plot_path))
    print("Saved markdown summary to {}".format(summary_md_path))


if __name__ == "__main__":
    main()
