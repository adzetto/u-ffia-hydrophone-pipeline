import argparse
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record a hydrophone wav clip and save an FFT spectrum image."
    )
    parser.add_argument("--duration", type=float, default=10.0, help="Recording duration in seconds.")
    parser.add_argument("--sample-rate", type=int, default=48000, help="Input sample rate.")
    parser.add_argument("--channels", type=int, default=1, help="Number of audio channels.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "Desktop" / "PROJE1" / "hidrofon",
        help="Directory where wav and spectrum files are written.",
    )
    parser.add_argument(
        "--max-frequency",
        type=float,
        default=20000.0,
        help="Upper x-axis bound used in the saved FFT plot.",
    )
    return parser.parse_args()


def find_umc22() -> int | None:
    keywords = ("umc22", "umc 22", "behringer", "umc", "usb audio codec", "usb audio")
    for index, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] <= 0:
            continue
        name = device["name"].lower()
        if any(keyword in name for keyword in keywords):
            return index
    return None


def save_spectrum_plot(waveform: np.ndarray, sample_rate: int, output_path: Path, max_frequency: float) -> None:
    signal = waveform.astype(np.float64)
    if signal.ndim > 1:
        signal = signal[:, 0]

    fft_values = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(len(signal), d=1.0 / sample_rate)
    magnitudes = np.abs(fft_values)

    plt.figure(figsize=(12, 5))
    plt.plot(frequencies, magnitudes, linewidth=1.0)
    plt.xlim(0, max_frequency)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Hydrophone FFT Spectrum")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device_index = find_umc22()
    if device_index is not None:
        device_info = sd.query_devices(device_index)
        print("Using input device #{}: {}".format(device_index, device_info["name"]))
    else:
        print("UMC22 not found explicitly; using the default input device.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = output_dir / "kayit_{}.wav".format(timestamp)
    plot_path = output_dir / "spektrum_{}.png".format(timestamp)

    frames = int(round(args.duration * args.sample_rate))
    print("Recording {} seconds...".format(args.duration))
    recording = sd.rec(
        frames,
        samplerate=args.sample_rate,
        channels=args.channels,
        dtype="int16",
        device=device_index,
        blocking=True,
    )

    sf.write(str(wav_path), recording, args.sample_rate, subtype="PCM_16")
    save_spectrum_plot(recording, args.sample_rate, plot_path, args.max_frequency)

    print("Saved wav to {}".format(wav_path))
    print("Saved spectrum to {}".format(plot_path))


if __name__ == "__main__":
    main()
