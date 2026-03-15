import argparse
import csv
import json
import random
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infer_audio_folder import (
    apply_domain_adaptation,
    apply_preprocessing,
    load_checkpoint_state,
    parse_notch_frequencies,
    resolve_checkpoint_path,
    resolve_device,
    sanitize_audio_features,
)
from models.Audio_model import Audio_Frontend
from models.model_zoo.MobileNetV2 import MobileNetV2


DEFAULT_HYDROPHONE_NOTCHES = (50.0, 100.0, 150.0, 200.0)


@dataclass(frozen=True)
class AudioExample:
    path: Path
    label: int
    label_name: str
    source_dir: str


class BinaryAudioModel(nn.Module):
    def __init__(self, frontend: nn.Module, backbone: nn.Module) -> None:
        super().__init__()
        self.frontend = frontend
        self.backbone = backbone

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        output = self.backbone(self.frontend(waveform))
        if isinstance(output, tuple):
            return output[0]
        return output


class BinaryAudioDataset(Dataset):
    def __init__(
        self,
        examples: Sequence[AudioExample],
        target_sample_rate: int,
        clip_seconds: float,
        preprocess_profile: str,
        adaptation_profile: str,
        adapt_target_rms: float,
        adapt_max_gain_db: float,
        highpass_hz: float,
        notch_freqs: Sequence[float],
        notch_q: float,
        training: bool,
        cache_waveforms: bool,
        seed: int,
    ) -> None:
        self.examples = list(examples)
        self.target_sample_rate = target_sample_rate
        self.clip_samples = int(round(target_sample_rate * clip_seconds))
        self.preprocess_profile = preprocess_profile
        self.adaptation_profile = adaptation_profile
        self.adapt_target_rms = adapt_target_rms
        self.adapt_max_gain_db = adapt_max_gain_db
        self.highpass_hz = highpass_hz
        self.notch_freqs = tuple(notch_freqs)
        self.notch_q = notch_q
        self.training = training
        self.cache_waveforms = cache_waveforms
        self.seed = seed
        self.epoch = 0
        self.cached_waveforms: Dict[Path, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.examples)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _load_waveform(self, audio_path: Path) -> np.ndarray:
        waveform, source_sample_rate = librosa.load(str(audio_path), sr=None, mono=True)
        waveform = apply_preprocessing(
            waveform=waveform,
            sample_rate=source_sample_rate,
            highpass_hz=self.highpass_hz,
            notch_freqs=self.notch_freqs,
            notch_q=self.notch_q,
        )
        waveform = apply_domain_adaptation(
            waveform=waveform,
            profile=self.adaptation_profile,
            target_rms=self.adapt_target_rms,
            max_gain_db=self.adapt_max_gain_db,
        )
        if source_sample_rate != self.target_sample_rate:
            waveform = librosa.resample(
                waveform,
                orig_sr=source_sample_rate,
                target_sr=self.target_sample_rate,
            )
        return waveform.astype(np.float32, copy=False)

    def _get_waveform(self, example: AudioExample) -> np.ndarray:
        if self.cache_waveforms:
            cached = self.cached_waveforms.get(example.path)
            if cached is None:
                cached = self._load_waveform(example.path)
                self.cached_waveforms[example.path] = cached
            return cached.copy()
        return self._load_waveform(example.path)

    def _fit_clip_length(self, waveform: np.ndarray, rng: random.Random) -> np.ndarray:
        if len(waveform) == self.clip_samples:
            return waveform.astype(np.float32, copy=False)
        if len(waveform) > self.clip_samples:
            if self.training:
                max_offset = len(waveform) - self.clip_samples
                offset = rng.randint(0, max_offset)
            else:
                offset = (len(waveform) - self.clip_samples) // 2
            return waveform[offset : offset + self.clip_samples].astype(np.float32, copy=False)
        return np.pad(waveform, (0, self.clip_samples - len(waveform)), mode="constant").astype(np.float32, copy=False)

    def _augment(self, waveform: np.ndarray, rng: random.Random, np_rng: np.random.Generator) -> np.ndarray:
        if not self.training:
            return waveform
        gain_db = rng.uniform(-4.0, 4.0)
        waveform = waveform * float(10 ** (gain_db / 20.0))
        noise_scale = rng.uniform(0.0, 0.003)
        if noise_scale > 0:
            waveform = waveform + np_rng.normal(0.0, noise_scale, size=waveform.shape).astype(np.float32)
        peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
        if peak > 1.0:
            waveform = waveform / peak
        return waveform.astype(np.float32, copy=False)

    def __getitem__(self, index: int) -> Dict[str, object]:
        example = self.examples[index]
        item_seed = self.seed + self.epoch * 1000003 + index
        rng = random.Random(item_seed)
        np_rng = np.random.default_rng(item_seed)
        waveform = self._get_waveform(example)
        waveform = self._fit_clip_length(waveform, rng)
        waveform = self._augment(waveform, rng, np_rng)
        return {
            "waveform": torch.from_numpy(waveform).float(),
            "target": torch.tensor(example.label, dtype=torch.long),
            "path": str(example.path),
            "label_name": example.label_name,
            "source_dir": example.source_dir,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a binary none-vs-yem audio model on local data.")
    parser.add_argument("--positive-dir", action="append", required=True, help="Folder containing positive yem wav files.")
    parser.add_argument("--negative-dir", action="append", required=True, help="Folder containing negative none wav files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for checkpoints, reports, and plots.")
    parser.add_argument("--config", type=Path, default=Path("config/audio/exp_binary_none_yem.yaml"), help="Binary training config.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Initial pretrained checkpoint. Defaults to released audio_best.pt.")
    parser.add_argument("--epochs", type=int, default=12, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--freeze-encoder-epochs", type=int, default=2, help="Warmup epochs where only the final classifier is trainable.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default=None, help="Torch device. Defaults to cuda when available.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers. Keep 0 when waveform caching is enabled.")
    parser.add_argument("--clip-seconds", type=float, default=2.0, help="Clip duration seen by the model.")
    parser.add_argument("--cache-waveforms", action="store_true", help="Cache processed waveforms in memory after first load.")
    parser.add_argument("--no-cache-waveforms", action="store_true", help="Disable waveform caching.")
    parser.add_argument("--preprocess-profile", choices=("none", "hydrophone"), default="hydrophone", help="Optional preprocessing profile.")
    parser.add_argument("--adaptation-profile", choices=("none", "hydrophone_v1"), default="hydrophone_v1", help="Optional domain adaptation profile.")
    parser.add_argument("--adapt-target-rms", type=float, default=0.05, help="Target RMS for RMS adaptation.")
    parser.add_argument("--adapt-max-gain-db", type=float, default=30.0, help="Maximum gain used by RMS adaptation.")
    parser.add_argument("--highpass-hz", type=float, default=None, help="Optional manual high-pass cutoff.")
    parser.add_argument("--notch-freqs", type=str, default=None, help="Optional comma-separated notch frequencies.")
    parser.add_argument("--notch-q", type=float, default=30.0, help="Quality factor for notch filters.")
    return parser.parse_args()


def iter_wav_files(folder: Path) -> Iterable[Path]:
    if not folder.is_dir():
        raise FileNotFoundError("Directory not found: {}".format(folder))
    for path in sorted(folder.rglob("*.wav")):
        yield path


def resolve_class_names(config: Dict) -> List[str]:
    class_names = config.get("Class_names")
    if isinstance(class_names, list) and len(class_names) == 2:
        return [str(item) for item in class_names]
    return ["none", "yem"]


def resolve_preprocessing(profile: str, highpass_hz: float | None, notch_freqs: str | None) -> tuple[float, List[float]]:
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


def collect_examples(negative_dirs: Sequence[Path], positive_dirs: Sequence[Path], class_names: Sequence[str]) -> List[AudioExample]:
    examples: List[AudioExample] = []
    for folder in negative_dirs:
        for path in iter_wav_files(folder):
            examples.append(AudioExample(path=path, label=0, label_name=class_names[0], source_dir=folder.name))
    for folder in positive_dirs:
        for path in iter_wav_files(folder):
            examples.append(AudioExample(path=path, label=1, label_name=class_names[1], source_dir=folder.name))
    if not examples:
        raise RuntimeError("No wav files found in the provided training folders.")
    return examples


def split_examples(
    examples: Sequence[AudioExample],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[List[AudioExample], List[AudioExample], List[AudioExample]]:
    if val_ratio <= 0 or test_ratio <= 0 or (val_ratio + test_ratio) >= 1:
        raise ValueError("val_ratio and test_ratio must be positive and sum to less than 1.")

    indices = np.arange(len(examples))
    labels = np.asarray([example.label for example in examples], dtype=np.int64)
    train_indices, temp_indices = train_test_split(
        indices,
        test_size=val_ratio + test_ratio,
        random_state=seed,
        stratify=labels,
    )
    temp_labels = labels[temp_indices]
    val_fraction_of_temp = val_ratio / (val_ratio + test_ratio)
    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=val_fraction_of_temp,
        random_state=seed,
        stratify=temp_labels,
    )
    train_examples = [examples[index] for index in train_indices]
    val_examples = [examples[index] for index in val_indices]
    test_examples = [examples[index] for index in test_indices]
    return train_examples, val_examples, test_examples


def make_dataloader(
    examples: Sequence[AudioExample],
    target_sample_rate: int,
    clip_seconds: float,
    preprocess_profile: str,
    adaptation_profile: str,
    adapt_target_rms: float,
    adapt_max_gain_db: float,
    highpass_hz: float,
    notch_freqs: Sequence[float],
    notch_q: float,
    batch_size: int,
    training: bool,
    cache_waveforms: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    dataset = BinaryAudioDataset(
        examples=examples,
        target_sample_rate=target_sample_rate,
        clip_seconds=clip_seconds,
        preprocess_profile=preprocess_profile,
        adaptation_profile=adaptation_profile,
        adapt_target_rms=adapt_target_rms,
        adapt_max_gain_db=adapt_max_gain_db,
        highpass_hz=highpass_hz,
        notch_freqs=notch_freqs,
        notch_q=notch_q,
        training=training,
        cache_waveforms=cache_waveforms,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers > 0),
    )


def initialize_from_checkpoint(model: nn.Module, checkpoint_path: Path) -> Dict[str, object]:
    checkpoint_state = load_checkpoint_state(checkpoint_path)
    model_state = model.state_dict()
    loaded_keys = []
    skipped_keys = []
    for key, value in checkpoint_state.items():
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape):
            model_state[key] = value
            loaded_keys.append(key)
        else:
            skipped_keys.append(key)
    model.load_state_dict(model_state)
    return {
        "loaded_keys": loaded_keys,
        "skipped_keys": skipped_keys,
    }


def set_trainable_parameters(model: BinaryAudioModel, train_full_model: bool) -> None:
    for param in model.parameters():
        param.requires_grad = train_full_model
    if not train_full_model:
        for name, param in model.named_parameters():
            if name.startswith("backbone.fc_audioset"):
                param.requires_grad = True


def run_epoch(
    model: BinaryAudioModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler,
    loss_fn: nn.Module,
    device: torch.device,
) -> Dict[str, object]:
    training = optimizer is not None
    model.train(training)
    all_targets: List[int] = []
    all_predictions: List[int] = []
    all_probabilities: List[np.ndarray] = []
    all_paths: List[str] = []
    all_sources: List[str] = []
    total_loss = 0.0
    total_items = 0

    autocast_enabled = False

    for batch in dataloader:
        waveforms = batch["waveform"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
            logits = model(waveforms)
            loss = loss_fn(logits, targets)

        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        probabilities = torch.softmax(logits.detach(), dim=1).cpu().numpy()
        predictions = np.argmax(probabilities, axis=1)
        batch_targets = targets.detach().cpu().numpy()

        batch_size = int(targets.shape[0])
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total_items += batch_size

        all_targets.extend(batch_targets.tolist())
        all_predictions.extend(predictions.tolist())
        all_probabilities.extend(probabilities.tolist())
        all_paths.extend(batch["path"])
        all_sources.extend(batch["source_dir"])

    mean_loss = total_loss / max(total_items, 1)
    accuracy = accuracy_score(all_targets, all_predictions)
    macro_f1 = f1_score(all_targets, all_predictions, average="macro", zero_division=0)
    positive_f1 = f1_score(all_targets, all_predictions, average="binary", zero_division=0)
    confusion = confusion_matrix(all_targets, all_predictions)
    return {
        "loss": mean_loss,
        "accuracy": float(accuracy),
        "f1": float(macro_f1),
        "positive_f1": float(positive_f1),
        "targets": all_targets,
        "predictions": all_predictions,
        "probabilities": all_probabilities,
        "paths": all_paths,
        "sources": all_sources,
        "confusion": confusion.tolist(),
    }


def apply_binary_threshold(metrics: Dict[str, object], threshold: float) -> Dict[str, object]:
    probabilities = np.asarray(metrics["probabilities"], dtype=np.float32)
    targets = np.asarray(metrics["targets"], dtype=np.int64)
    predictions = (probabilities[:, 1] >= threshold).astype(np.int64)
    accuracy = accuracy_score(targets, predictions)
    macro_f1 = f1_score(targets, predictions, average="macro", zero_division=0)
    positive_f1 = f1_score(targets, predictions, average="binary", zero_division=0)
    confusion = confusion_matrix(targets, predictions)

    updated = dict(metrics)
    updated["predictions"] = predictions.tolist()
    updated["accuracy"] = float(accuracy)
    updated["f1"] = float(macro_f1)
    updated["positive_f1"] = float(positive_f1)
    updated["confusion"] = confusion.tolist()
    updated["decision_threshold"] = float(threshold)
    return updated


def find_best_binary_threshold(metrics: Dict[str, object]) -> float:
    best_threshold = 0.5
    best_macro_f1 = -1.0
    best_accuracy = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        candidate = apply_binary_threshold(metrics, float(threshold))
        if candidate["f1"] > best_macro_f1 or (
            np.isclose(candidate["f1"], best_macro_f1) and candidate["accuracy"] > best_accuracy
        ):
            best_threshold = float(threshold)
            best_macro_f1 = float(candidate["f1"])
            best_accuracy = float(candidate["accuracy"])
    return best_threshold


def save_split_csv(output_path: Path, split_name: str, examples: Sequence[AudioExample]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if output_path.exists() else "w"
    with output_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "file", "label_index", "label_name", "source_dir"])
        if mode == "w":
            writer.writeheader()
        for example in examples:
            writer.writerow(
                {
                    "split": split_name,
                    "file": str(example.path),
                    "label_index": example.label,
                    "label_name": example.label_name,
                    "source_dir": example.source_dir,
                }
            )


def save_prediction_csv(output_path: Path, split_name: str, metrics: Dict[str, object], class_names: Sequence[str]) -> None:
    rows = []
    for index, file_path in enumerate(metrics["paths"]):
        row = {
            "split": split_name,
            "file": file_path,
            "source_dir": metrics["sources"][index],
            "target_index": metrics["targets"][index],
            "target_label": class_names[metrics["targets"][index]],
            "predicted_index": metrics["predictions"][index],
            "predicted_label": class_names[metrics["predictions"][index]],
        }
        probabilities = metrics["probabilities"][index]
        for class_index, class_name in enumerate(class_names):
            row["score_{}".format(class_name)] = round(float(probabilities[class_index]), 6)
        rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["split", "file"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_training_plot(output_path: Path, history: List[Dict[str, float]]) -> None:
    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    val_loss = [item["val_loss"] for item in history]
    train_f1 = [item["train_f1"] for item in history]
    val_f1 = [item["val_f1"] for item in history]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].plot(epochs, train_loss, label="train_loss", color="#4c78a8")
    axes[0].plot(epochs, val_loss, label="val_loss", color="#e45756")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_f1, label="train_f1", color="#54a24b")
    axes[1].plot(epochs, val_f1, label="val_f1", color="#f58518")
    axes[1].set_title("F1 Score")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_confusion_plot(output_path: Path, confusion: np.ndarray, class_names: Sequence[str], title: str) -> None:
    figure, axis = plt.subplots(figsize=(5, 4.5), constrained_layout=True)
    image = axis.imshow(confusion, cmap="Blues")
    axis.set_title(title)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_xticks(range(len(class_names)))
    axis.set_xticklabels(class_names)
    axis.set_yticks(range(len(class_names)))
    axis.set_yticklabels(class_names)
    for row_index in range(confusion.shape[0]):
        for column_index in range(confusion.shape[1]):
            axis.text(column_index, row_index, int(confusion[row_index, column_index]), ha="center", va="center")
    figure.colorbar(image, ax=axis)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = OmegaConf.to_container(OmegaConf.load(str(args.config)), resolve=True)
    audio_features = sanitize_audio_features(config["Audio_features"])
    class_names = resolve_class_names(config)
    if int(config["Training"]["classes_num"]) != 2:
        raise ValueError("Binary fine-tuning requires classes_num=2 in {}".format(args.config))

    negative_dirs = [Path(path).expanduser().resolve() for path in args.negative_dir]
    positive_dirs = [Path(path).expanduser().resolve() for path in args.positive_dir]
    examples = collect_examples(negative_dirs=negative_dirs, positive_dirs=positive_dirs, class_names=class_names)
    train_examples, val_examples, test_examples = split_examples(
        examples=examples,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_csv = output_dir / "dataset_split.csv"
    if split_csv.exists():
        split_csv.unlink()
    save_split_csv(split_csv, "train", train_examples)
    save_split_csv(split_csv, "val", val_examples)
    save_split_csv(split_csv, "test", test_examples)

    highpass_hz, notch_freqs = resolve_preprocessing(
        profile=args.preprocess_profile,
        highpass_hz=args.highpass_hz,
        notch_freqs=args.notch_freqs,
    )
    cache_waveforms = True
    if args.no_cache_waveforms:
        cache_waveforms = False
    if args.cache_waveforms:
        cache_waveforms = True
    device = resolve_device(args.device)

    frontend = Audio_Frontend(**audio_features)
    backbone = MobileNetV2(classes_num=2)
    model = BinaryAudioModel(frontend=frontend, backbone=backbone).to(device)

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    checkpoint_summary = initialize_from_checkpoint(model, checkpoint_path)

    train_loader = make_dataloader(
        examples=train_examples,
        target_sample_rate=int(audio_features["sample_rate"]),
        clip_seconds=args.clip_seconds,
        preprocess_profile=args.preprocess_profile,
        adaptation_profile=args.adaptation_profile,
        adapt_target_rms=args.adapt_target_rms,
        adapt_max_gain_db=args.adapt_max_gain_db,
        highpass_hz=highpass_hz,
        notch_freqs=notch_freqs,
        notch_q=args.notch_q,
        batch_size=args.batch_size,
        training=True,
        cache_waveforms=cache_waveforms,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    val_loader = make_dataloader(
        examples=val_examples,
        target_sample_rate=int(audio_features["sample_rate"]),
        clip_seconds=args.clip_seconds,
        preprocess_profile=args.preprocess_profile,
        adaptation_profile=args.adaptation_profile,
        adapt_target_rms=args.adapt_target_rms,
        adapt_max_gain_db=args.adapt_max_gain_db,
        highpass_hz=highpass_hz,
        notch_freqs=notch_freqs,
        notch_q=args.notch_q,
        batch_size=args.batch_size,
        training=False,
        cache_waveforms=cache_waveforms,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    test_loader = make_dataloader(
        examples=test_examples,
        target_sample_rate=int(audio_features["sample_rate"]),
        clip_seconds=args.clip_seconds,
        preprocess_profile=args.preprocess_profile,
        adaptation_profile=args.adaptation_profile,
        adapt_target_rms=args.adapt_target_rms,
        adapt_max_gain_db=args.adapt_max_gain_db,
        highpass_hz=highpass_hz,
        notch_freqs=notch_freqs,
        notch_q=args.notch_q,
        batch_size=args.batch_size,
        training=False,
        cache_waveforms=cache_waveforms,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    train_labels = np.asarray([example.label for example in train_examples], dtype=np.int64)
    class_counts = np.bincount(train_labels, minlength=2)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1) / len(class_counts)
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    best_state = deepcopy(model.state_dict())
    best_summary = {"epoch": 0, "val_f1": -1.0, "val_accuracy": -1.0}
    history: List[Dict[str, float]] = []

    print("Training on device:", device)
    print("Initial checkpoint:", checkpoint_path)
    print("Loaded keys:", len(checkpoint_summary["loaded_keys"]))
    print("Skipped keys:", len(checkpoint_summary["skipped_keys"]))
    print(
        "Split sizes | train={} val={} test={}".format(
            len(train_examples),
            len(val_examples),
            len(test_examples),
        )
    )
    print(
        "Preprocess profile={} highpass={} notch_freqs={} adaptation={}".format(
            args.preprocess_profile,
            highpass_hz,
            ",".join(str(value) for value in notch_freqs) or "none",
            args.adaptation_profile,
        )
    )

    for epoch in range(1, args.epochs + 1):
        train_full_model = epoch > args.freeze_encoder_epochs
        set_trainable_parameters(model, train_full_model=train_full_model)
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)

        train_metrics = run_epoch(model, train_loader, optimizer, scaler, loss_fn, device)
        val_metrics = run_epoch(model, val_loader, None, scaler, loss_fn, device)

        history_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
        "train_f1": train_metrics["f1"],
        "val_loss": val_metrics["loss"],
        "val_accuracy": val_metrics["accuracy"],
        "val_f1": val_metrics["f1"],
        }
        history.append(history_row)

        improved = (
            val_metrics["f1"] > best_summary["val_f1"]
            or (
                np.isclose(val_metrics["f1"], best_summary["val_f1"])
                and val_metrics["accuracy"] > best_summary["val_accuracy"]
            )
        )
        if improved:
            best_summary = {
                "epoch": epoch,
                "val_f1": val_metrics["f1"],
                "val_accuracy": val_metrics["accuracy"],
            }
            best_state = deepcopy(model.state_dict())

        print(
            "epoch={:02d} train_loss={:.4f} train_macro_f1={:.4f} val_loss={:.4f} val_macro_f1={:.4f} val_acc={:.4f} mode={}".format(
                epoch,
                train_metrics["loss"],
                train_metrics["f1"],
                val_metrics["loss"],
                val_metrics["f1"],
                val_metrics["accuracy"],
                "full" if train_full_model else "head-only",
            )
        )

    model.load_state_dict(best_state)
    final_train_metrics = run_epoch(model, train_loader, None, scaler, loss_fn, device)
    final_val_metrics = run_epoch(model, val_loader, None, scaler, loss_fn, device)
    final_test_metrics = run_epoch(model, test_loader, None, scaler, loss_fn, device)
    best_threshold = find_best_binary_threshold(final_val_metrics)
    final_train_metrics = apply_binary_threshold(final_train_metrics, best_threshold)
    final_val_metrics = apply_binary_threshold(final_val_metrics, best_threshold)
    final_test_metrics = apply_binary_threshold(final_test_metrics, best_threshold)

    checkpoint_payload = {
        "model_state_dict": best_state,
        "class_names": class_names,
        "config": config,
        "metrics": {
            "best_epoch": best_summary["epoch"],
            "train_accuracy": final_train_metrics["accuracy"],
            "train_macro_f1": final_train_metrics["f1"],
            "train_positive_f1": final_train_metrics["positive_f1"],
            "val_accuracy": final_val_metrics["accuracy"],
            "val_macro_f1": final_val_metrics["f1"],
            "val_positive_f1": final_val_metrics["positive_f1"],
            "test_accuracy": final_test_metrics["accuracy"],
            "test_macro_f1": final_test_metrics["f1"],
            "test_positive_f1": final_test_metrics["positive_f1"],
            "decision_threshold": best_threshold,
        },
        "preprocess": {
            "profile": args.preprocess_profile,
            "adaptation_profile": args.adaptation_profile,
            "adapt_target_rms": args.adapt_target_rms,
            "adapt_max_gain_db": args.adapt_max_gain_db,
            "highpass_hz": highpass_hz,
            "notch_freqs": list(notch_freqs),
            "notch_q": args.notch_q,
        },
    }
    checkpoint_out = output_dir / "audio_binary_best.pt"
    torch.save(checkpoint_payload, checkpoint_out)

    save_prediction_csv(output_dir / "train_predictions.csv", "train", final_train_metrics, class_names)
    save_prediction_csv(output_dir / "val_predictions.csv", "val", final_val_metrics, class_names)
    save_prediction_csv(output_dir / "test_predictions.csv", "test", final_test_metrics, class_names)
    save_training_plot(output_dir / "training_curves.png", history)
    save_confusion_plot(
        output_dir / "test_confusion.png",
        np.asarray(final_test_metrics["confusion"]),
        class_names=class_names,
        title="Binary Audio Fine-Tune Test Confusion",
    )

    report = {
        "checkpoint": str(checkpoint_out),
        "initial_checkpoint": str(checkpoint_path),
        "class_names": class_names,
        "dataset_sizes": {
            "train": len(train_examples),
            "val": len(val_examples),
            "test": len(test_examples),
        },
        "best_epoch": best_summary["epoch"],
        "decision_threshold": best_threshold,
        "train": {
            "accuracy": final_train_metrics["accuracy"],
            "macro_f1": final_train_metrics["f1"],
            "positive_f1": final_train_metrics["positive_f1"],
            "confusion": final_train_metrics["confusion"],
        },
        "val": {
            "accuracy": final_val_metrics["accuracy"],
            "macro_f1": final_val_metrics["f1"],
            "positive_f1": final_val_metrics["positive_f1"],
            "confusion": final_val_metrics["confusion"],
        },
        "test": {
            "accuracy": final_test_metrics["accuracy"],
            "macro_f1": final_test_metrics["f1"],
            "positive_f1": final_test_metrics["positive_f1"],
            "confusion": final_test_metrics["confusion"],
            "classification_report": classification_report(
                final_test_metrics["targets"],
                final_test_metrics["predictions"],
                target_names=list(class_names),
                output_dict=True,
                zero_division=0,
            ),
        },
        "preprocess": checkpoint_payload["preprocess"],
        "loaded_key_count": len(checkpoint_summary["loaded_keys"]),
        "skipped_key_count": len(checkpoint_summary["skipped_keys"]),
    }
    report_path = output_dir / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved checkpoint to {}".format(checkpoint_out))
    print("Saved metrics to {}".format(report_path))
    print(
        "Final metrics | val_acc={:.4f} val_macro_f1={:.4f} test_acc={:.4f} test_macro_f1={:.4f}".format(
            final_val_metrics["accuracy"],
            final_val_metrics["f1"],
            final_test_metrics["accuracy"],
            final_test_metrics["f1"],
        )
    )


if __name__ == "__main__":
    main()
