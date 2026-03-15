import argparse
import sys
from pathlib import Path
from typing import Iterable, List

import librosa
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infer_audio_folder import apply_preprocessing, pad_or_trim, parse_notch_frequencies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert hydrophone wav files into model-ready mono chunks."
    )
    parser.add_argument("input_path", type=Path, help="Single wav file or a folder of wav files.")
    parser.add_argument("output_dir", type=Path, help="Directory where processed wav chunks are written.")
    parser.add_argument(
        "--target-sr",
        type=int,
        default=64000,
        help="Target sample rate expected by the released U-FFIA configs.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=2.0,
        help="Chunk length in seconds.",
    )
    parser.add_argument(
        "--hop-seconds",
        type=float,
        default=2.0,
        help="Hop length in seconds between adjacent chunks.",
    )
    parser.add_argument(
        "--highpass-hz",
        type=float,
        default=120.0,
        help="Butterworth high-pass cutoff applied before chunking.",
    )
    parser.add_argument(
        "--notch-freqs",
        type=str,
        default="50,100,150,200",
        help="Comma-separated notch filter frequencies in Hz.",
    )
    parser.add_argument(
        "--notch-q",
        type=float,
        default=30.0,
        help="Quality factor for the IIR notch filters.",
    )
    parser.add_argument(
        "--tail-mode",
        choices=("pad", "drop"),
        default="pad",
        help="How to handle the last partial chunk.",
    )
    return parser.parse_args()


def find_audio_files(input_path: Path) -> List[Path]:
    input_path = input_path.expanduser().resolve()
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


def split_into_chunks(
    waveform,
    sample_rate: int,
    chunk_seconds: float,
    hop_seconds: float,
    tail_mode: str,
) -> Iterable:
    target_samples = int(round(chunk_seconds * sample_rate))
    hop_samples = int(round(hop_seconds * sample_rate))
    if target_samples <= 0 or hop_samples <= 0:
        raise ValueError("Chunk and hop durations must be positive.")

    if len(waveform) <= target_samples:
        if tail_mode == "drop" and len(waveform) < target_samples:
            return
        yield pad_or_trim(waveform, target_samples)
        return

    start = 0
    while start < len(waveform):
        chunk = waveform[start : start + target_samples]
        if len(chunk) < target_samples:
            if tail_mode == "drop":
                break
            chunk = pad_or_trim(chunk, target_samples)
            yield chunk
            break
        yield chunk
        if start + target_samples >= len(waveform):
            break
        start += hop_samples


def relative_output_base(audio_path: Path, input_path: Path) -> Path:
    if input_path.is_file():
        return Path(audio_path.stem)
    return audio_path.relative_to(input_path).with_suffix("")


def main() -> None:
    args = parse_args()
    input_path = args.input_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    notch_freqs = parse_notch_frequencies(args.notch_freqs)
    audio_files = find_audio_files(input_path)
    total_chunks = 0

    for audio_path in audio_files:
        waveform, source_sr = librosa.load(str(audio_path), sr=None, mono=True)
        if waveform.size == 0:
            print("Skipping empty file {}".format(audio_path))
            continue

        waveform = apply_preprocessing(
            waveform=waveform,
            sample_rate=source_sr,
            highpass_hz=args.highpass_hz,
            notch_freqs=notch_freqs,
            notch_q=args.notch_q,
        )
        if source_sr != args.target_sr:
            waveform = librosa.resample(waveform, orig_sr=source_sr, target_sr=args.target_sr)

        chunk_base = relative_output_base(audio_path, input_path)
        chunk_dir = output_dir / chunk_base.parent
        chunk_dir.mkdir(parents=True, exist_ok=True)

        written = 0
        for chunk_index, chunk in enumerate(
            split_into_chunks(
                waveform=waveform,
                sample_rate=args.target_sr,
                chunk_seconds=args.chunk_seconds,
                hop_seconds=args.hop_seconds,
                tail_mode=args.tail_mode,
            )
        ):
            output_path = chunk_dir / "{}__chunk_{:03d}.wav".format(chunk_base.name, chunk_index)
            sf.write(str(output_path), chunk, args.target_sr, subtype="PCM_16")
            written += 1

        total_chunks += written
        print("{} -> {} chunk(s)".format(audio_path, written))

    print("Saved {} processed chunk(s) under {}".format(total_chunks, output_dir))


if __name__ == "__main__":
    main()
