import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


CLASS_NAMES = ["none", "strong", "medium", "weak"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare multiple file-level inference CSV runs and render a matplotlib summary."
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("LABEL", "CSV"),
        required=True,
        help="Comparison run in the form: --run label path/to/results.csv",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path.")
    parser.add_argument("--title", type=str, default="Inference Run Comparison", help="Figure title.")
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=None,
        help="Optional markdown summary path for label changes and confidence trends.",
    )
    return parser.parse_args()


def read_run(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return sorted(list(csv.DictReader(handle)), key=lambda row: row["file"])


def top_confidence(row: Dict[str, str]) -> float:
    return max(float(row["score_none"]), float(row["score_strong"]), float(row["score_medium"]), float(row["score_weak"]))


def plot_runs(run_data: List[Tuple[str, List[Dict[str, str]]]], output_path: Path, title: str) -> None:
    file_labels = [Path(row["file"]).stem for row in run_data[0][1]]
    n_runs = len(run_data)
    x_files = np.arange(len(file_labels))

    figure = plt.figure(figsize=(max(16, len(file_labels) * 0.7), 10), constrained_layout=True)
    grid = figure.add_gridspec(3, 2, height_ratios=[2.2, 1.5, 1.2], hspace=0.35, wspace=0.25)
    axis_counts = figure.add_subplot(grid[0, 0])
    axis_conf = figure.add_subplot(grid[0, 1])
    axis_heatmap = figure.add_subplot(grid[1, :])
    axis_delta = figure.add_subplot(grid[2, :])

    width = 0.8 / max(1, n_runs)
    x_classes = np.arange(len(CLASS_NAMES))
    for run_index, (label, rows) in enumerate(run_data):
        counts = Counter(row["predicted_label"] for row in rows)
        axis_counts.bar(
            x_classes + (run_index - (n_runs - 1) / 2.0) * width,
            [counts.get(name, 0) for name in CLASS_NAMES],
            width=width,
            label=label,
            alpha=0.85,
        )

        confidences = [top_confidence(row) for row in rows]
        axis_conf.plot(x_files, confidences, marker="o", linewidth=1.5, label=label)

    axis_counts.set_xticks(x_classes)
    axis_counts.set_xticklabels(CLASS_NAMES)
    axis_counts.set_ylabel("File count")
    axis_counts.set_title("Predicted Label Counts")
    axis_counts.legend()

    axis_conf.set_title("Top-Class Confidence By File")
    axis_conf.set_xlabel("File index")
    axis_conf.set_ylabel("Confidence")
    axis_conf.set_ylim(0.0, 1.05)
    axis_conf.grid(True, alpha=0.3)
    axis_conf.legend()

    class_matrix = np.full((n_runs, len(file_labels)), np.nan, dtype=float)
    for run_index, (_, rows) in enumerate(run_data):
        for file_index, row in enumerate(rows):
            class_matrix[run_index, file_index] = CLASS_TO_INDEX[row["predicted_label"]]
    cmap = ListedColormap(["#4c78a8", "#f58518", "#54a24b", "#b279a2"])
    norm = BoundaryNorm(np.arange(-0.5, len(CLASS_NAMES) + 0.5, 1.0), cmap.N)
    image = axis_heatmap.imshow(class_matrix, aspect="auto", cmap=cmap, norm=norm)
    axis_heatmap.set_title("Predicted Class Per File")
    axis_heatmap.set_yticks(range(n_runs))
    axis_heatmap.set_yticklabels([label for label, _ in run_data])
    axis_heatmap.set_xticks(x_files)
    axis_heatmap.set_xticklabels(file_labels, rotation=60, ha="right", fontsize=8)
    axis_heatmap.set_xlabel("Audio file")
    colorbar = figure.colorbar(image, ax=axis_heatmap, ticks=range(len(CLASS_NAMES)), fraction=0.025, pad=0.02)
    colorbar.ax.set_yticklabels(CLASS_NAMES)

    baseline_conf = np.array([top_confidence(row) for row in run_data[0][1]], dtype=float)
    for label, rows in run_data[1:]:
        confidences = np.array([top_confidence(row) for row in rows], dtype=float)
        axis_delta.plot(x_files, confidences - baseline_conf, marker="o", linewidth=1.5, label="{} - {}".format(label, run_data[0][0]))
    axis_delta.axhline(0.0, color="#555555", linewidth=1.0, linestyle="--")
    axis_delta.set_title("Confidence Delta Versus Baseline")
    axis_delta.set_xlabel("File index")
    axis_delta.set_ylabel("Delta confidence")
    axis_delta.grid(True, alpha=0.3)
    if n_runs > 1:
        axis_delta.legend()

    figure.suptitle(title, fontsize=15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_markdown(run_data: List[Tuple[str, List[Dict[str, str]]]], output_path: Path) -> None:
    baseline_label, baseline_rows = run_data[0]
    baseline_map = {Path(row["file"]).name: row for row in baseline_rows}
    lines = [
        "# Inference Run Comparison",
        "",
        "- Baseline run: `{}`".format(baseline_label),
        "",
    ]
    for label, rows in run_data:
        counts = Counter(row["predicted_label"] for row in rows)
        mean_conf = np.mean([top_confidence(row) for row in rows])
        lines.append("- {}: counts={}, mean_confidence={:.4f}".format(label, dict(counts), float(mean_conf)))

    for label, rows in run_data[1:]:
        changed = []
        for row in rows:
            file_name = Path(row["file"]).name
            before = baseline_map[file_name]
            if before["predicted_label"] != row["predicted_label"]:
                changed.append((file_name, before["predicted_label"], row["predicted_label"], top_confidence(before), top_confidence(row)))
        lines.extend(["", "## Label Changes: {} vs {}".format(label, baseline_label), ""])
        if not changed:
            lines.append("- No file-level label changes.")
        else:
            for item in changed:
                lines.append(
                    "- {}: {} ({:.4f}) -> {} ({:.4f})".format(
                        item[0],
                        item[1],
                        item[3],
                        item[2],
                        item[4],
                    )
                )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_data = [(label, read_run(Path(path).expanduser().resolve())) for label, path in args.run]
    plot_runs(run_data, args.output.expanduser().resolve(), args.title)
    print("Saved comparison plot to {}".format(args.output.expanduser().resolve()))
    if args.markdown_output is not None:
        write_markdown(run_data, args.markdown_output.expanduser().resolve())
        print("Saved markdown summary to {}".format(args.markdown_output.expanduser().resolve()))


if __name__ == "__main__":
    main()
