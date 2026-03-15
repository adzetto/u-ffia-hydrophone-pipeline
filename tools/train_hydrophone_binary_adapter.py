import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import joblib
import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_HYDROPHONE_NOTCHES = (50.0, 100.0, 150.0, 200.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a local binary hydrophone-domain adapter using weakly labeled folders."
    )
    parser.add_argument(
        "--positive-dir",
        action="append",
        required=True,
        help="Folder containing positive examples, e.g. feeding-like clips.",
    )
    parser.add_argument(
        "--negative-dir",
        action="append",
        required=True,
        help="Folder containing negative examples, e.g. non-feeding-like clips.",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        required=True,
        help="Path where the trained joblib model is saved.",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=None,
        help="Optional markdown report path.",
    )
    parser.add_argument(
        "--report-png",
        type=Path,
        default=None,
        help="Optional confusion-matrix plot path.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=48000,
        help="Sample rate used when loading training wav files.",
    )
    parser.add_argument(
        "--preprocess-profile",
        choices=("none", "hydrophone"),
        default="hydrophone",
        help="Optional preprocessing profile applied before feature extraction.",
    )
    parser.add_argument(
        "--adaptation-profile",
        choices=("none", "hydrophone_v1"),
        default="none",
        help="Optional domain-adaptation profile applied before feature extraction.",
    )
    parser.add_argument(
        "--adapt-target-rms",
        type=float,
        default=0.05,
        help="Target RMS for adaptation profile.",
    )
    parser.add_argument(
        "--adapt-max-gain-db",
        type=float,
        default=30.0,
        help="Maximum gain magnitude allowed during RMS adaptation.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Cross-validation folds used in the training report.",
    )
    return parser.parse_args()


def iter_wav_files(folder: Path) -> Iterable[Path]:
    if not folder.is_dir():
        raise FileNotFoundError("Directory not found: {}".format(folder))
    for path in sorted(folder.rglob("*.wav")):
        yield path


def apply_preprocessing(waveform: np.ndarray, sample_rate: int, profile: str) -> np.ndarray:
    if profile == "none":
        return waveform.astype(np.float32, copy=False)
    if profile != "hydrophone":
        raise ValueError("Unsupported preprocess profile: {}".format(profile))

    processed = waveform.astype(np.float64, copy=False)
    sos = butter(4, 120.0, btype="highpass", fs=sample_rate, output="sos")
    processed = sosfiltfilt(sos, processed)
    for frequency in DEFAULT_HYDROPHONE_NOTCHES:
        if frequency >= sample_rate / 2:
            continue
        b, a = iirnotch(frequency, Q=30.0, fs=sample_rate)
        processed = filtfilt(b, a, processed)
    peak = float(np.max(np.abs(processed))) if processed.size else 0.0
    if peak > 1.0:
        processed = processed / peak
    return processed.astype(np.float32)


def apply_domain_adaptation(
    waveform: np.ndarray,
    profile: str,
    target_rms: float,
    max_gain_db: float,
) -> np.ndarray:
    if profile == "none":
        return waveform.astype(np.float32, copy=False)
    if profile != "hydrophone_v1":
        raise ValueError("Unsupported adaptation profile: {}".format(profile))

    adapted = waveform.astype(np.float64, copy=False)
    adapted = adapted - float(np.mean(adapted))
    current_rms = float(np.sqrt(np.mean(adapted * adapted))) if adapted.size else 0.0
    if current_rms > 0 and target_rms > 0:
        max_gain = float(10 ** (max_gain_db / 20.0))
        gain = target_rms / current_rms
        gain = float(np.clip(gain, 1.0 / max_gain, max_gain))
        adapted = adapted * gain
    peak = float(np.max(np.abs(adapted))) if adapted.size else 0.0
    if peak > 1.0:
        adapted = adapted / peak
    return adapted.astype(np.float32)


def extract_features(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=20)
    features = []
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))
    features.extend(np.mean(librosa.feature.spectral_centroid(y=waveform, sr=sample_rate), axis=1))
    features.extend(np.mean(librosa.feature.spectral_bandwidth(y=waveform, sr=sample_rate), axis=1))
    features.extend(np.mean(librosa.feature.spectral_rolloff(y=waveform, sr=sample_rate), axis=1))
    features.extend(np.mean(librosa.feature.zero_crossing_rate(y=waveform), axis=1))
    features.extend(np.mean(librosa.feature.rms(y=waveform), axis=1))
    return np.asarray(features, dtype=np.float32)


def load_examples(
    folders: Sequence[Path],
    label: int,
    sample_rate: int,
    preprocess_profile: str,
    adaptation_profile: str,
    adapt_target_rms: float,
    adapt_max_gain_db: float,
) -> Tuple[List[np.ndarray], List[int], List[str]]:
    features = []
    labels = []
    names = []
    for folder in folders:
        for wav_path in iter_wav_files(folder):
            waveform, _ = librosa.load(str(wav_path), sr=sample_rate, mono=True)
            waveform = apply_preprocessing(waveform, sample_rate, preprocess_profile)
            waveform = apply_domain_adaptation(waveform, adaptation_profile, adapt_target_rms, adapt_max_gain_db)
            features.append(extract_features(waveform, sample_rate))
            labels.append(label)
            names.append(str(wav_path))
    return features, labels, names


def write_report(
    path: Path,
    positive_dirs: Sequence[Path],
    negative_dirs: Sequence[Path],
    accuracy: float,
    f1: float,
    confusion: np.ndarray,
    preprocess_profile: str,
    adaptation_profile: str,
) -> None:
    lines = [
        "# Hydrophone Binary Adapter Report",
        "",
        "## Configuration",
        "",
        "- Positive dirs: {}".format([str(path) for path in positive_dirs]),
        "- Negative dirs: {}".format([str(path) for path in negative_dirs]),
        "- Preprocess profile: `{}`".format(preprocess_profile),
        "- Adaptation profile: `{}`".format(adaptation_profile),
        "",
        "## Cross-Validation",
        "",
        "- Accuracy: {:.4f}".format(accuracy),
        "- F1: {:.4f}".format(f1),
        "- Confusion matrix:",
        "",
        "```text",
        str(confusion),
        "```",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def save_confusion_plot(path: Path, confusion: np.ndarray) -> None:
    figure, axis = plt.subplots(figsize=(5, 4), constrained_layout=True)
    image = axis.imshow(confusion, cmap="Blues")
    axis.set_title("Binary Adapter CV Confusion Matrix")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_xticks([0, 1])
    axis.set_xticklabels(["negative", "positive"])
    axis.set_yticks([0, 1])
    axis.set_yticklabels(["negative", "positive"])
    for row_index in range(confusion.shape[0]):
        for column_index in range(confusion.shape[1]):
            axis.text(column_index, row_index, int(confusion[row_index, column_index]), ha="center", va="center")
    figure.colorbar(image, ax=axis)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    positive_dirs = [Path(path).expanduser().resolve() for path in args.positive_dir]
    negative_dirs = [Path(path).expanduser().resolve() for path in args.negative_dir]

    positive_features, positive_labels, _ = load_examples(
        folders=positive_dirs,
        label=1,
        sample_rate=args.sample_rate,
        preprocess_profile=args.preprocess_profile,
        adaptation_profile=args.adaptation_profile,
        adapt_target_rms=args.adapt_target_rms,
        adapt_max_gain_db=args.adapt_max_gain_db,
    )
    negative_features, negative_labels, _ = load_examples(
        folders=negative_dirs,
        label=0,
        sample_rate=args.sample_rate,
        preprocess_profile=args.preprocess_profile,
        adaptation_profile=args.adaptation_profile,
        adapt_target_rms=args.adapt_target_rms,
        adapt_max_gain_db=args.adapt_max_gain_db,
    )

    X = np.vstack(positive_features + negative_features)
    y = np.asarray(positive_labels + negative_labels, dtype=np.int64)

    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    splitter = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    accuracy = float(cross_val_score(pipeline, X, y, cv=splitter, scoring="accuracy").mean())
    f1 = float(cross_val_score(pipeline, X, y, cv=splitter, scoring="f1").mean())
    predictions = cross_val_predict(pipeline, X, y, cv=splitter)
    confusion = confusion_matrix(y, predictions)

    pipeline.fit(X, y)

    payload = {
        "model": pipeline,
        "metadata": {
            "positive_dirs": [str(path) for path in positive_dirs],
            "negative_dirs": [str(path) for path in negative_dirs],
            "sample_rate": args.sample_rate,
            "preprocess_profile": args.preprocess_profile,
            "adaptation_profile": args.adaptation_profile,
            "adapt_target_rms": args.adapt_target_rms,
            "adapt_max_gain_db": args.adapt_max_gain_db,
            "cv_accuracy": accuracy,
            "cv_f1": f1,
        },
    }
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, args.output_model)
    print("Saved binary adapter model to {}".format(args.output_model))
    print("Cross-validation accuracy {:.4f}, f1 {:.4f}".format(accuracy, f1))

    if args.report_md is not None:
        args.report_md.parent.mkdir(parents=True, exist_ok=True)
        write_report(
            path=args.report_md,
            positive_dirs=positive_dirs,
            negative_dirs=negative_dirs,
            accuracy=accuracy,
            f1=f1,
            confusion=confusion,
            preprocess_profile=args.preprocess_profile,
            adaptation_profile=args.adaptation_profile,
        )
        print("Saved markdown report to {}".format(args.report_md))

    if args.report_png is not None:
        args.report_png.parent.mkdir(parents=True, exist_ok=True)
        save_confusion_plot(args.report_png, confusion)
        print("Saved confusion plot to {}".format(args.report_png))


if __name__ == "__main__":
    main()
