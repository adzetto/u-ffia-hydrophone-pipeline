import argparse
import csv
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from models.model_zoo.S3D import S3D


DEFAULT_CLASS_NAMES_4 = ["none", "strong", "medium", "weak"]
DEFAULT_CLASS_NAMES_2 = ["none", "yem"]
REPO_ROOT = Path(__file__).resolve().parent


class VideoInferenceModel(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, video: torch.Tensor) -> Dict[str, torch.Tensor]:
        backbone_output = self.backbone(video)
        if isinstance(backbone_output, tuple):
            clipwise_output = backbone_output[0]
        else:
            clipwise_output = backbone_output
        return {"clipwise_output": clipwise_output}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run U-FFIA video inference on an mp4 file or a folder of mp4 files."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Single mp4 file or a folder containing mp4 files.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to the trained video checkpoint. If omitted, common pretrained_models locations are searched.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("video_inference_results.csv"),
        help="CSV file for file-level predictions.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device, for example cpu or cuda. Defaults to cuda when available.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=20,
        help="Number of frames sampled uniformly from each video clip.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=250,
        help="Square resize used before center cropping.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=196,
        help="Center crop size fed into S3D.",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        type=str,
        default="ffmpeg",
        help="Path to ffmpeg executable.",
    )
    parser.add_argument(
        "--binary-positive-threshold",
        type=float,
        default=None,
        help="Optional decision threshold for the positive class when using a 2-class checkpoint.",
    )
    return parser.parse_args()


def checkpoint_candidates() -> List[Path]:
    return [
        REPO_ROOT / "pretrained_models" / "audio-visual_pretrainedmodel" / "video_best.pt",
        REPO_ROOT.parent / "pretrained_models" / "audio-visual_pretrainedmodel" / "video_best.pt",
        REPO_ROOT / "pretrained_models" / "video_best.pt",
        REPO_ROOT.parent / "pretrained_models" / "video_best.pt",
    ]


def resolve_checkpoint_path(checkpoint_arg: Path | None) -> Path:
    if checkpoint_arg is not None:
        return checkpoint_arg.expanduser().resolve()

    for candidate in checkpoint_candidates():
        if candidate.exists():
            return candidate

    searched = "\n".join(str(path) for path in checkpoint_candidates())
    raise FileNotFoundError(
        "Video checkpoint not found. Searched:\n{}".format(searched)
    )


def find_video_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".mp4":
            raise ValueError("Only .mp4 files are supported.")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError("Input path does not exist: {}".format(input_path))

    files = sorted(input_path.rglob("*.mp4"))
    if not files:
        raise FileNotFoundError("No .mp4 files found under {}".format(input_path))
    return files


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint_state(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    if not checkpoint_path.exists():
        raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint_path))

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise RuntimeError("Unsupported checkpoint format in {}".format(checkpoint_path))

    return {key.replace("module.", ""): value for key, value in state_dict.items()}


def load_checkpoint_metadata(checkpoint_path: Path) -> Dict[str, object]:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        return {}
    metadata = {}
    if "class_names" in checkpoint:
        metadata["class_names"] = checkpoint["class_names"]
    if "metrics" in checkpoint:
        metadata["metrics"] = checkpoint["metrics"]
    if "preprocess" in checkpoint:
        metadata["preprocess"] = checkpoint["preprocess"]
    return metadata


def infer_classes_num(state_dict: Dict[str, torch.Tensor]) -> int:
    classifier_weight = state_dict.get("backbone.fc.0.weight")
    if classifier_weight is None:
        raise RuntimeError("Could not infer classes_num from checkpoint.")
    return int(classifier_weight.shape[0])


def resolve_class_names(classes_num: int, metadata: Dict[str, object]) -> List[str]:
    names = metadata.get("class_names")
    if isinstance(names, list) and len(names) == classes_num:
        return [str(name) for name in names]
    if classes_num == 4:
        return list(DEFAULT_CLASS_NAMES_4)
    if classes_num == 2:
        return list(DEFAULT_CLASS_NAMES_2)
    return ["class_{}".format(index) for index in range(classes_num)]


def build_model(checkpoint_path: Path, device: torch.device) -> Tuple[nn.Module, List[str], Dict[str, object]]:
    state_dict = load_checkpoint_state(checkpoint_path)
    metadata = load_checkpoint_metadata(checkpoint_path)
    classes_num = infer_classes_num(state_dict)
    model = VideoInferenceModel(backbone=S3D(classes_num=classes_num))
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint and model do not match.\nMissing keys: {}\nUnexpected keys: {}".format(
                incompatible.missing_keys,
                incompatible.unexpected_keys,
            )
        )
    model.to(device)
    model.eval()
    return model, resolve_class_names(classes_num, metadata), metadata


def decode_video_frames(
    video_path: Path,
    ffmpeg_bin: str,
    resize: int,
) -> np.ndarray:
    command = [
        ffmpeg_bin,
        "-v",
        "error",
        "-i",
        str(video_path),
        "-an",
        "-sn",
        "-vf",
        "scale={0}:{0}".format(resize),
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, check=True)
    frame_size = resize * resize * 3
    if len(result.stdout) == 0:
        raise RuntimeError("ffmpeg returned no frames for {}".format(video_path))
    if len(result.stdout) % frame_size != 0:
        raise RuntimeError("Unexpected rawvideo byte count for {}".format(video_path))
    frame_count = len(result.stdout) // frame_size
    frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(frame_count, resize, resize, 3)
    return frames


def center_crop_frames(frames: np.ndarray, crop_size: int) -> np.ndarray:
    height = frames.shape[1]
    width = frames.shape[2]
    if crop_size > min(height, width):
        raise ValueError("crop_size must be smaller than decoded frame size.")
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    return frames[:, top : top + crop_size, left : left + crop_size, :]


def sample_frames_uniformly(frames: np.ndarray, num_frames: int) -> np.ndarray:
    if len(frames) == 0:
        raise RuntimeError("No frames available after decoding.")
    if len(frames) >= num_frames:
        indices = np.linspace(0, len(frames) - 1, num_frames).round().astype(int)
        return frames[indices]
    indices = np.linspace(0, len(frames) - 1, num_frames).round().astype(int)
    indices = np.clip(indices, 0, len(frames) - 1)
    return frames[indices]


def normalize_frames(frames: np.ndarray) -> np.ndarray:
    normalized = frames.astype(np.float32) / 255.0
    normalized = (normalized - 0.5) / 0.5
    return np.transpose(normalized, (0, 3, 1, 2))


def load_video_clip(
    video_path: Path,
    ffmpeg_bin: str,
    resize: int,
    crop_size: int,
    num_frames: int,
) -> Tuple[np.ndarray, int]:
    decoded_frames = decode_video_frames(
        video_path=video_path,
        ffmpeg_bin=ffmpeg_bin,
        resize=resize,
    )
    sampled = sample_frames_uniformly(decoded_frames, num_frames=num_frames)
    cropped = center_crop_frames(sampled, crop_size=crop_size)
    normalized = normalize_frames(cropped)
    return normalized, int(decoded_frames.shape[0])


def infer_video(
    model: nn.Module,
    clip: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    batch = torch.from_numpy(clip[np.newaxis, ...]).float().to(device)
    with torch.no_grad():
        logits = model(batch)["clipwise_output"]
        probabilities = torch.softmax(logits, dim=1)
    return probabilities.squeeze(0).cpu().numpy()


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    model, class_names, metadata = build_model(checkpoint_path, device)
    video_files = find_video_files(args.input_path)
    preprocess_metadata = metadata.get("preprocess", {}) if isinstance(metadata, dict) else {}
    if args.num_frames == 20 and preprocess_metadata.get("num_frames") is not None:
        args.num_frames = int(preprocess_metadata["num_frames"])
    if args.resize == 250 and preprocess_metadata.get("resize") is not None:
        args.resize = int(preprocess_metadata["resize"])
    if args.crop_size == 196 and preprocess_metadata.get("crop_size") is not None:
        args.crop_size = int(preprocess_metadata["crop_size"])
    binary_positive_threshold = args.binary_positive_threshold
    checkpoint_metrics = metadata.get("metrics", {}) if isinstance(metadata, dict) else {}
    if binary_positive_threshold is None and isinstance(checkpoint_metrics, dict):
        stored_threshold = checkpoint_metrics.get("decision_threshold")
        if stored_threshold is not None:
            binary_positive_threshold = float(stored_threshold)
    if binary_positive_threshold is None:
        binary_positive_threshold = 0.5
    if len(class_names) == 2:
        print("Binary checkpoint decision threshold={}".format(binary_positive_threshold))

    rows = []
    for video_path in video_files:
        clip, decoded_frame_count = load_video_clip(
            video_path=video_path,
            ffmpeg_bin=args.ffmpeg_bin,
            resize=args.resize,
            crop_size=args.crop_size,
            num_frames=args.num_frames,
        )
        probabilities = infer_video(model=model, clip=clip, device=device)
        if len(class_names) == 2:
            predicted_index = 1 if float(probabilities[1]) >= float(binary_positive_threshold) else 0
        else:
            predicted_index = int(np.argmax(probabilities))
        predicted_label = class_names[predicted_index]

        row = {
            "file": str(video_path),
            "decoded_frame_count": decoded_frame_count,
            "sampled_frame_count": int(args.num_frames),
            "predicted_index": predicted_index,
            "predicted_label": predicted_label,
        }
        for class_index, class_name in enumerate(class_names):
            row["score_{}".format(class_name)] = round(float(probabilities[class_index]), 6)
        rows.append(row)

        print(
            "{} -> {} ({:.4f}) [{}]".format(
                video_path,
                predicted_label,
                float(probabilities[predicted_index]),
                checkpoint_path,
            )
        )

    fieldnames = [
        "file",
        "decoded_frame_count",
        "sampled_frame_count",
        "predicted_index",
        "predicted_label",
    ]
    fieldnames.extend("score_{}".format(class_name) for class_name in class_names)
    write_csv(args.output_csv, rows, fieldnames)
    print("Saved video inference results to {}".format(args.output_csv))


if __name__ == "__main__":
    main()
