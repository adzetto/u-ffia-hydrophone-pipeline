import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import joblib
import librosa
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

from models.Audio_model import Audio_Frontend
from models.model_zoo.MobileNetV2 import MobileNetV2
from models.model_zoo.panns import PANNS_Cnn10


CLASS_NAMES = ["none", "strong", "medium", "weak"]
DEFAULT_HYDROPHONE_NOTCHES = (50.0, 100.0, 150.0, 200.0)
REPO_ROOT = Path(__file__).resolve().parent


class AudioInferenceModel(nn.Module):
    """Wrapper matching checkpoints saved with frontend/backbone module names."""

    def __init__(self, frontend: nn.Module, backbone: nn.Module) -> None:
        super().__init__()
        self.frontend = frontend
        self.backbone = backbone

    def forward(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        backbone_output = self.backbone(self.frontend(waveform))
        if isinstance(backbone_output, tuple):
            clipwise_output = backbone_output[0]
        else:
            clipwise_output = backbone_output
        return {"clipwise_output": clipwise_output}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run U-FFIA audio inference on a wav file or a folder of wav files."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Single wav file or a folder containing wav files.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/audio/pre_exp.yaml"),
        help="Training config used for the audio checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to the trained audio checkpoint. If omitted, common pretrained_models locations are searched.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("audio_inference_results.csv"),
        help="CSV file for aggregated file-level predictions.",
    )
    parser.add_argument(
        "--window-csv",
        type=Path,
        default=None,
        help="Optional CSV file for per-window predictions.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=2.0,
        help="Window length in seconds. Training data used 2-second clips.",
    )
    parser.add_argument(
        "--hop-seconds",
        type=float,
        default=2.0,
        help="Hop length in seconds between windows.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of windows processed together.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device, for example cpu or cuda. Defaults to cuda when available.",
    )
    parser.add_argument(
        "--preprocess-profile",
        choices=("none", "hydrophone"),
        default="none",
        help="Optional input preprocessing profile applied before windowing.",
    )
    parser.add_argument(
        "--highpass-hz",
        type=float,
        default=None,
        help="Optional Butterworth high-pass cutoff. Overrides the selected profile.",
    )
    parser.add_argument(
        "--notch-freqs",
        type=str,
        default=None,
        help="Comma-separated notch filter frequencies in Hz. Overrides the selected profile.",
    )
    parser.add_argument(
        "--notch-q",
        type=float,
        default=30.0,
        help="Quality factor used by the IIR notch filters.",
    )
    parser.add_argument(
        "--adaptation-profile",
        choices=("none", "hydrophone_v1"),
        default="none",
        help="Optional unsupervised domain-adaptation profile applied after preprocessing.",
    )
    parser.add_argument(
        "--adapt-target-rms",
        type=float,
        default=0.05,
        help="Target RMS used by adaptation profiles that normalize capture loudness.",
    )
    parser.add_argument(
        "--adapt-max-gain-db",
        type=float,
        default=30.0,
        help="Maximum absolute gain applied by RMS adaptation, in dB.",
    )
    parser.add_argument(
        "--binary-adapter-model",
        type=Path,
        default=None,
        help="Optional joblib model trained on local hydrophone-domain labels.",
    )
    parser.add_argument(
        "--stabilization-profile",
        choices=("none", "binary_gate"),
        default="none",
        help="Optional post-model stabilization profile.",
    )
    parser.add_argument(
        "--binary-feed-threshold",
        type=float,
        default=0.5,
        help="Threshold used by the binary gate to decide whether the clip is feeding-like.",
    )
    return parser.parse_args()


def checkpoint_candidates() -> List[Path]:
    return [
        REPO_ROOT / "pretrained_models" / "audio-visual_pretrainedmodel" / "audio_best.pt",
        REPO_ROOT.parent / "pretrained_models" / "audio-visual_pretrainedmodel" / "audio_best.pt",
        REPO_ROOT / "pretrained_models" / "audio_best.pt",
        REPO_ROOT.parent / "pretrained_models" / "audio_best.pt",
    ]


def resolve_checkpoint_path(checkpoint_arg: Path | None) -> Path:
    if checkpoint_arg is not None:
        return checkpoint_arg.expanduser().resolve()

    for candidate in checkpoint_candidates():
        if candidate.exists():
            return candidate

    searched = "\n".join(str(path) for path in checkpoint_candidates())
    raise FileNotFoundError(
        "Audio checkpoint not found. Searched:\n{}".format(searched)
    )


def find_audio_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".wav":
            raise ValueError("Only .wav files are supported.")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError("Input path does not exist: {}".format(input_path))

    files = sorted(input_path.rglob("*.wav"))
    if not files:
        raise FileNotFoundError("No .wav files found under {}".format(input_path))
    return files


def load_config(config_path: Path) -> Dict:
    if not config_path.exists():
        raise FileNotFoundError("Config file not found: {}".format(config_path))
    return OmegaConf.to_container(OmegaConf.load(str(config_path)), resolve=True)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint_state(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Checkpoint not found: {}. Place the Google Drive weights under "
            "pretrained_models/ and try again.".format(checkpoint_path)
        )

    # The released checkpoints were saved as full PyTorch objects before
    # torch.load switched to weights_only=True by default in PyTorch 2.6.
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise RuntimeError("Unsupported checkpoint format in {}".format(checkpoint_path))

    return {
        key.replace("module.", ""): value
        for key, value in state_dict.items()
    }


def infer_backbone_name(state_dict: Dict[str, torch.Tensor]) -> str:
    if any(key.startswith("backbone.features.") for key in state_dict):
        return "mobilenetv2"
    if any(key.startswith("backbone.conv_block1.conv1.weight") for key in state_dict):
        return "panns_cnn10"
    raise RuntimeError("Unsupported checkpoint architecture.")


def build_backbone(backbone_name: str, classes_num: int) -> nn.Module:
    if backbone_name == "mobilenetv2":
        return MobileNetV2(classes_num=classes_num)
    if backbone_name == "panns_cnn10":
        return PANNS_Cnn10(classes_num=classes_num)
    raise RuntimeError("Unsupported backbone: {}".format(backbone_name))


def sanitize_audio_features(audio_features: Dict) -> Dict:
    sanitized = dict(audio_features)
    sample_rate = int(sanitized["sample_rate"])
    nyquist = sample_rate / 2.0
    fmax = float(sanitized["fmax"])
    fmin = float(sanitized["fmin"])

    # Several released configs set fmax above Nyquist, which produces empty
    # mel filters during inference. Clamp it defensively to a valid range.
    safe_fmax = min(fmax, max(fmin + 1.0, nyquist - 1.0))
    safe_fmin = min(fmin, safe_fmax - 1.0)
    if safe_fmax != fmax or safe_fmin != fmin:
        print(
            "Adjusted audio frontend band limits from fmin={}Hz, fmax={}Hz to "
            "fmin={}Hz, fmax={}Hz for sample_rate={}Hz.".format(
                fmin,
                fmax,
                safe_fmin,
                safe_fmax,
                sample_rate,
            )
        )
    sanitized["fmin"] = safe_fmin
    sanitized["fmax"] = safe_fmax
    return sanitized


def build_model(config: Dict, checkpoint_path: Path, device: torch.device) -> Tuple[nn.Module, int, str]:
    audio_features = sanitize_audio_features(config["Audio_features"])
    classes_num = int(config["Training"]["classes_num"])
    state_dict = load_checkpoint_state(checkpoint_path)
    backbone_name = infer_backbone_name(state_dict)

    frontend = Audio_Frontend(**audio_features)
    backbone = build_backbone(backbone_name, classes_num=classes_num)
    model = AudioInferenceModel(frontend=frontend, backbone=backbone)

    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint and model do not match.\nMissing keys: {}\nUnexpected keys: {}".format(
                incompatible.missing_keys, incompatible.unexpected_keys
            )
        )

    model.to(device)
    model.eval()
    return model, int(audio_features["sample_rate"]), backbone_name


def pad_or_trim(waveform: np.ndarray, target_samples: int) -> np.ndarray:
    if len(waveform) >= target_samples:
        return waveform[:target_samples]
    return np.pad(waveform, (0, target_samples - len(waveform)), mode="constant")


def parse_notch_frequencies(value: str | None) -> List[float]:
    if value is None:
        return []
    items = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        items.append(float(token))
    return items


def resolve_preprocess_settings(
    profile: str,
    highpass_hz: float | None,
    notch_freqs: str | None,
) -> Tuple[float, List[float]]:
    if profile == "hydrophone":
        resolved_highpass = 120.0
        resolved_notches = list(DEFAULT_HYDROPHONE_NOTCHES)
    else:
        resolved_highpass = 0.0
        resolved_notches = []

    if highpass_hz is not None:
        resolved_highpass = highpass_hz
    if notch_freqs is not None:
        resolved_notches = parse_notch_frequencies(notch_freqs)

    return resolved_highpass, resolved_notches


def apply_preprocessing(
    waveform: np.ndarray,
    sample_rate: int,
    highpass_hz: float,
    notch_freqs: Sequence[float],
    notch_q: float,
) -> np.ndarray:
    processed = waveform.astype(np.float64, copy=False)

    if highpass_hz > 0:
        if highpass_hz >= sample_rate / 2:
            raise ValueError("High-pass cutoff must be below the Nyquist frequency.")
        # A gentle high-pass helps remove drift and sub-audio structure from hydrophone captures.
        sos = butter(4, highpass_hz, btype="highpass", fs=sample_rate, output="sos")
        processed = sosfiltfilt(sos, processed)

    for frequency in notch_freqs:
        if frequency <= 0 or frequency >= sample_rate / 2:
            continue
        b, a = iirnotch(frequency, Q=notch_q, fs=sample_rate)
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


def load_processed_waveform(
    audio_path: Path,
    target_sr: int,
    highpass_hz: float,
    notch_freqs: Sequence[float],
    notch_q: float,
    adaptation_profile: str,
    adapt_target_rms: float,
    adapt_max_gain_db: float,
) -> Tuple[np.ndarray, float]:
    waveform, source_sr = librosa.load(str(audio_path), sr=None, mono=True)
    if waveform.size == 0:
        raise RuntimeError("Failed to read audio samples from {}".format(audio_path))

    duration_seconds = float(len(waveform)) / float(source_sr)
    waveform = apply_preprocessing(
        waveform=waveform,
        sample_rate=source_sr,
        highpass_hz=highpass_hz,
        notch_freqs=notch_freqs,
        notch_q=notch_q,
    )
    waveform = apply_domain_adaptation(
        waveform=waveform,
        profile=adaptation_profile,
        target_rms=adapt_target_rms,
        max_gain_db=adapt_max_gain_db,
    )

    if source_sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=source_sr, target_sr=target_sr)

    return waveform.astype(np.float32), duration_seconds


def split_windows_from_waveform(
    waveform: np.ndarray,
    target_sr: int,
    window_seconds: float,
    hop_seconds: float,
) -> List[np.ndarray]:

    target_samples = int(round(window_seconds * target_sr))
    hop_samples = int(round(hop_seconds * target_sr))
    if target_samples <= 0 or hop_samples <= 0:
        raise ValueError("Window and hop sizes must be positive.")

    if len(waveform) <= target_samples:
        return [pad_or_trim(waveform, target_samples)]

    windows = []
    start = 0
    while start < len(waveform):
        chunk = waveform[start : start + target_samples]
        if len(chunk) < target_samples:
            chunk = pad_or_trim(chunk, target_samples)
            windows.append(chunk.astype(np.float32))
            break
        windows.append(chunk.astype(np.float32))
        if start + target_samples >= len(waveform):
            break
        start += hop_samples

    return windows


def extract_binary_adapter_features(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
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


def load_binary_adapter_bundle(model_path: Path | None) -> Tuple[object | None, Dict[str, object]]:
    if model_path is None:
        return None, {}
    resolved = model_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError("Binary adapter model not found: {}".format(resolved))
    payload = joblib.load(resolved)
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    model = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    if not hasattr(model, "predict_proba"):
        raise RuntimeError("Binary adapter model does not expose predict_proba().")
    return model, metadata


def stabilize_probabilities_with_binary_gate(
    mean_probabilities: np.ndarray,
    binary_feed_probability: float,
) -> np.ndarray:
    stabilized = np.zeros_like(mean_probabilities, dtype=np.float32)
    stabilized[0] = float(np.clip(1.0 - binary_feed_probability, 0.0, 1.0))
    non_none = np.clip(mean_probabilities[1:], 0.0, None)
    total = float(np.sum(non_none))
    if total <= 0:
        non_none = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        total = 1.0
    stabilized[1:] = float(binary_feed_probability) * (non_none / total)
    return stabilized.astype(np.float32)


def batched(items: Sequence[np.ndarray], batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(items), batch_size):
        yield np.stack(items[start : start + batch_size])


def infer_windows(
    model: nn.Module,
    windows: Sequence[np.ndarray],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    probabilities = []

    with torch.no_grad():
        for batch in batched(windows, batch_size=batch_size):
            batch_tensor = torch.from_numpy(batch).float().to(device)
            logits = model(batch_tensor)["clipwise_output"]
            probs = torch.softmax(logits, dim=1)
            probabilities.append(probs.cpu().numpy())

    return np.concatenate(probabilities, axis=0)


def aggregate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    return probabilities.mean(axis=0)


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    device = resolve_device(args.device)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    model, target_sr, backbone_name = build_model(config, checkpoint_path, device)
    binary_adapter_model, binary_adapter_metadata = load_binary_adapter_bundle(args.binary_adapter_model)
    audio_files = find_audio_files(args.input_path)
    highpass_hz, notch_frequencies = resolve_preprocess_settings(
        profile=args.preprocess_profile,
        highpass_hz=args.highpass_hz,
        notch_freqs=args.notch_freqs,
    )

    if highpass_hz > 0 or notch_frequencies:
        print(
            "Preprocessing enabled: profile={}, highpass={}Hz, notch_freqs={}, notch_q={}".format(
                args.preprocess_profile,
                highpass_hz,
                ",".join(str(int(freq)) if float(freq).is_integer() else str(freq) for freq in notch_frequencies) or "none",
                args.notch_q,
            )
        )
    if args.adaptation_profile != "none":
        print(
            "Domain adaptation enabled: profile={}, target_rms={}, max_gain_db={}".format(
                args.adaptation_profile,
                args.adapt_target_rms,
                args.adapt_max_gain_db,
            )
        )
    if args.stabilization_profile != "none":
        if binary_adapter_model is None:
            raise ValueError("stabilization-profile={} requires --binary-adapter-model.".format(args.stabilization_profile))
        print(
            "Stabilization enabled: profile={}, threshold={}, adapter={}".format(
                args.stabilization_profile,
                args.binary_feed_threshold,
                args.binary_adapter_model,
            )
        )

    file_rows = []
    window_rows = []

    for audio_path in audio_files:
        processed_waveform, duration_seconds = load_processed_waveform(
            audio_path=audio_path,
            target_sr=target_sr,
            highpass_hz=highpass_hz,
            notch_freqs=notch_frequencies,
            notch_q=args.notch_q,
            adaptation_profile=args.adaptation_profile,
            adapt_target_rms=args.adapt_target_rms,
            adapt_max_gain_db=args.adapt_max_gain_db,
        )
        windows = split_windows_from_waveform(
            waveform=processed_waveform,
            target_sr=target_sr,
            window_seconds=args.window_seconds,
            hop_seconds=args.hop_seconds,
        )
        probabilities = infer_windows(
            model=model,
            windows=windows,
            batch_size=args.batch_size,
            device=device,
        )
        mean_probabilities = aggregate_probabilities(probabilities)
        base_probabilities = mean_probabilities.copy()
        binary_feed_probability = None
        adapter_gate_label = ""
        if binary_adapter_model is not None:
            adapter_sample_rate = int(binary_adapter_metadata.get("sample_rate", target_sr))
            adapter_waveform = processed_waveform
            if adapter_sample_rate != target_sr:
                adapter_waveform = librosa.resample(processed_waveform, orig_sr=target_sr, target_sr=adapter_sample_rate)
            adapter_features = extract_binary_adapter_features(adapter_waveform, sample_rate=adapter_sample_rate)
            binary_feed_probability = float(binary_adapter_model.predict_proba(adapter_features.reshape(1, -1))[0, 1])
            adapter_gate_label = "feeding_like" if binary_feed_probability >= args.binary_feed_threshold else "nonfeeding_like"
            if args.stabilization_profile == "binary_gate":
                mean_probabilities = stabilize_probabilities_with_binary_gate(
                    mean_probabilities=mean_probabilities,
                    binary_feed_probability=binary_feed_probability,
                )
        predicted_index = int(np.argmax(mean_probabilities))
        predicted_label = CLASS_NAMES[predicted_index]

        row = {
            "file": str(audio_path),
            "duration_seconds": round(duration_seconds, 3),
            "num_windows": int(len(windows)),
            "predicted_index": predicted_index,
            "predicted_label": predicted_label,
        }
        for class_index, class_name in enumerate(CLASS_NAMES):
            row["score_{}".format(class_name)] = round(float(mean_probabilities[class_index]), 6)
            row["base_score_{}".format(class_name)] = round(float(base_probabilities[class_index]), 6)
        row["binary_feed_probability"] = "" if binary_feed_probability is None else round(binary_feed_probability, 6)
        row["binary_gate_label"] = adapter_gate_label
        file_rows.append(row)

        for window_index, prob_vector in enumerate(probabilities):
            window_prediction = int(np.argmax(prob_vector))
            window_row = {
                "file": str(audio_path),
                "window_index": window_index,
                "predicted_index": window_prediction,
                "predicted_label": CLASS_NAMES[window_prediction],
            }
            for class_index, class_name in enumerate(CLASS_NAMES):
                window_row["score_{}".format(class_name)] = round(float(prob_vector[class_index]), 6)
            window_rows.append(window_row)

        print(
            "{} -> {} ({:.4f}) [{}]".format(
                audio_path,
                predicted_label,
                float(mean_probabilities[predicted_index]),
                "{} | {}".format(checkpoint_path, backbone_name),
            )
        )

    file_fieldnames = [
        "file",
        "duration_seconds",
        "num_windows",
        "predicted_index",
        "predicted_label",
        "score_none",
        "score_strong",
        "score_medium",
        "score_weak",
        "base_score_none",
        "base_score_strong",
        "base_score_medium",
        "base_score_weak",
        "binary_feed_probability",
        "binary_gate_label",
    ]
    write_csv(args.output_csv, file_rows, file_fieldnames)

    if args.window_csv is not None:
        window_fieldnames = [
            "file",
            "window_index",
            "predicted_index",
            "predicted_label",
            "score_none",
            "score_strong",
            "score_medium",
            "score_weak",
        ]
        write_csv(args.window_csv, window_rows, window_fieldnames)

    print("Saved aggregated results to {}".format(args.output_csv))
    if args.window_csv is not None:
        print("Saved per-window results to {}".format(args.window_csv))


if __name__ == "__main__":
    main()
