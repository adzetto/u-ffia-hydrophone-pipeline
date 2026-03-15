import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


CLASS_NAMES = ["none", "strong", "medium", "weak"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create matplotlib summary plots from U-FFIA inference CSV outputs."
    )
    parser.add_argument("aggregated_csv", type=Path, help="File-level inference CSV.")
    parser.add_argument("--window-csv", type=Path, default=None, help="Optional per-window inference CSV.")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path.")
    parser.add_argument("--title", type=str, default="Inference Summary", help="Figure title.")
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def short_name(path_value: str) -> str:
    return Path(path_value).stem


def build_probability_matrix(rows: List[Dict[str, str]]) -> np.ndarray:
    matrix = np.zeros((len(CLASS_NAMES), len(rows)), dtype=float)
    for column, row in enumerate(rows):
        for class_index, class_name in enumerate(CLASS_NAMES):
            matrix[class_index, column] = float(row["score_{}".format(class_name)])
    return matrix


def build_window_matrix(window_rows: List[Dict[str, str]], ordered_files: List[str]) -> np.ndarray:
    grouped = defaultdict(list)
    for row in window_rows:
        grouped[row["file"]].append(row)

    max_windows = max((len(grouped[file_path]) for file_path in ordered_files), default=0)
    matrix = np.full((len(ordered_files), max_windows), np.nan, dtype=float)

    for row_index, file_path in enumerate(ordered_files):
        file_windows = sorted(grouped[file_path], key=lambda row: int(row["window_index"]))
        for item in file_windows:
            window_index = int(item["window_index"])
            matrix[row_index, window_index] = CLASS_TO_INDEX[item["predicted_label"]]

    return matrix


def main() -> None:
    args = parse_args()
    aggregated_rows = read_csv(args.aggregated_csv)
    if not aggregated_rows:
        raise RuntimeError("No rows found in {}".format(args.aggregated_csv))

    aggregated_rows = sorted(aggregated_rows, key=lambda row: row["file"])
    ordered_files = [row["file"] for row in aggregated_rows]
    file_labels = [short_name(row["file"]) for row in aggregated_rows]
    predicted_labels = [row["predicted_label"] for row in aggregated_rows]
    confidence_values = [
        max(float(row["score_none"]), float(row["score_strong"]), float(row["score_medium"]), float(row["score_weak"]))
        for row in aggregated_rows
    ]
    probability_matrix = build_probability_matrix(aggregated_rows)
    label_counts = Counter(predicted_labels)

    window_rows = read_csv(args.window_csv) if args.window_csv is not None and args.window_csv.exists() else []
    has_windows = bool(window_rows)

    if has_windows:
        figure = plt.figure(figsize=(max(14, len(aggregated_rows) * 0.7), 11), constrained_layout=True)
        grid = figure.add_gridspec(3, 2, height_ratios=[2.6, 2.0, 1.4], hspace=0.5, wspace=0.3)
        axis_heatmap = figure.add_subplot(grid[0, :])
        axis_windows = figure.add_subplot(grid[1, :])
        axis_counts = figure.add_subplot(grid[2, 0])
        axis_conf = figure.add_subplot(grid[2, 1])
    else:
        figure = plt.figure(figsize=(max(14, len(aggregated_rows) * 0.7), 8), constrained_layout=True)
        grid = figure.add_gridspec(2, 2, height_ratios=[2.8, 1.4], hspace=0.45, wspace=0.3)
        axis_heatmap = figure.add_subplot(grid[0, :])
        axis_counts = figure.add_subplot(grid[1, 0])
        axis_conf = figure.add_subplot(grid[1, 1])
        axis_windows = None

    image = axis_heatmap.imshow(probability_matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    axis_heatmap.set_title("{}\nFile-Level Class Probabilities".format(args.title))
    axis_heatmap.set_yticks(range(len(CLASS_NAMES)))
    axis_heatmap.set_yticklabels(CLASS_NAMES)
    axis_heatmap.set_xticks(range(len(file_labels)))
    axis_heatmap.set_xticklabels(file_labels, rotation=60, ha="right", fontsize=8)
    axis_heatmap.set_xlabel("Audio file")
    axis_heatmap.set_ylabel("Class")
    colorbar = figure.colorbar(image, ax=axis_heatmap, fraction=0.025, pad=0.02)
    colorbar.set_label("Probability")

    for column, label in enumerate(predicted_labels):
        axis_heatmap.text(
            column,
            -0.55,
            label,
            ha="center",
            va="center",
            fontsize=8,
            rotation=60,
        )

    if has_windows and axis_windows is not None:
        window_matrix = build_window_matrix(window_rows, ordered_files)
        cmap = ListedColormap(["#4c78a8", "#f58518", "#54a24b", "#b279a2"])
        norm = BoundaryNorm(np.arange(-0.5, len(CLASS_NAMES) + 0.5, 1.0), cmap.N)
        masked_matrix = np.ma.masked_invalid(window_matrix)
        image_windows = axis_windows.imshow(masked_matrix, aspect="auto", cmap=cmap, norm=norm)
        axis_windows.set_title("Per-Window Dominant Class")
        axis_windows.set_yticks(range(len(file_labels)))
        axis_windows.set_yticklabels(file_labels, fontsize=8)
        axis_windows.set_xticks(range(window_matrix.shape[1]))
        axis_windows.set_xlabel("Window index")
        axis_windows.set_ylabel("Audio file")
        colorbar_windows = figure.colorbar(
            image_windows,
            ax=axis_windows,
            ticks=range(len(CLASS_NAMES)),
            fraction=0.025,
            pad=0.02,
        )
        colorbar_windows.ax.set_yticklabels(CLASS_NAMES)

    axis_counts.bar(CLASS_NAMES, [label_counts.get(name, 0) for name in CLASS_NAMES], color=["#4c78a8", "#f58518", "#54a24b", "#b279a2"])
    axis_counts.set_title("Predicted Label Counts")
    axis_counts.set_ylabel("File count")
    axis_counts.set_ylim(0, max(1, len(aggregated_rows)))

    dominant_indices = [CLASS_TO_INDEX[label] for label in predicted_labels]
    axis_conf.scatter(range(len(confidence_values)), confidence_values, c=dominant_indices, cmap=ListedColormap(["#4c78a8", "#f58518", "#54a24b", "#b279a2"]), s=50)
    axis_conf.plot(range(len(confidence_values)), confidence_values, color="#444444", linewidth=1.0, alpha=0.7)
    axis_conf.set_title("Top-Class Confidence By File")
    axis_conf.set_xlabel("File index")
    axis_conf.set_ylabel("Confidence")
    axis_conf.set_ylim(0.0, 1.05)
    axis_conf.grid(True, alpha=0.3)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print("Saved plot to {}".format(args.output))


if __name__ == "__main__":
    main()
