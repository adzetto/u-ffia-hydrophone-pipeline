import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a fast late-fusion model from existing audio/video prediction CSVs.")
    parser.add_argument(
        "--audio-prediction-dir",
        type=Path,
        default=Path("results/finetune_binary_none_yem/hydrophone_source_e8_thr"),
        help="Directory containing train/val/test audio prediction CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/fusion_fast"),
        help="Directory for fusion artifacts.",
    )
    parser.add_argument(
        "--source-weight",
        action="append",
        default=[],
        help="Optional source_dir=weight rule, for example voice_yem2=0.8.",
    )
    parser.add_argument(
        "--video-result",
        action="append",
        default=[],
        help="Optional source_dir=csv_path mapping. Defaults to results/video_*_results.csv.",
    )
    return parser.parse_args()


def default_video_csvs(repo_root: Path) -> Dict[str, Path]:
    return {
        "video_none1": repo_root / "results" / "video_none1_results.csv",
        "video_none2": repo_root / "results" / "video_none2_results.csv",
        "video_yem1": repo_root / "results" / "video_yem1_results.csv",
        "video_yem2": repo_root / "results" / "video_yem2_results.csv",
    }


def parse_mapping_rules(values: Sequence[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("Invalid mapping rule: {}".format(value))
        key, raw_value = value.split("=", 1)
        mapping[key.strip()] = raw_value.strip()
    return mapping


def load_video_lookup(video_csvs: Dict[str, Path]) -> Dict[tuple[str, str], Dict[str, float]]:
    lookup: Dict[tuple[str, str], Dict[str, float]] = {}
    for source, path in video_csvs.items():
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                stem = Path(row["file"]).stem
                lookup[(source, stem)] = {
                    "video_none": float(row["score_none"]),
                    "video_strong": float(row["score_strong"]),
                    "video_medium": float(row["score_medium"]),
                    "video_weak": float(row["score_weak"]),
                }
    return lookup


def feature_names() -> List[str]:
    return [
        "audio_none",
        "audio_yem",
        "audio_margin",
        "audio_entropy",
        "video_none",
        "video_strong",
        "video_medium",
        "video_weak",
        "video_feedlike",
        "video_active",
        "video_strong_minus_none",
        "cross_audio_video",
        "agree_feed",
    ]


def resolve_sample_weight(source_dir: str, rules: Dict[str, str]) -> float:
    if source_dir in rules:
        return float(rules[source_dir])
    return 1.0


def build_rows(
    prediction_csv: Path,
    video_lookup: Dict[tuple[str, str], Dict[str, float]],
    source_weight_rules: Dict[str, str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with prediction_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            audio_path = Path(row["file"])
            stem = audio_path.stem
            source_dir = row["source_dir"]
            video_dir = source_dir.replace("voice_", "video_")
            video_scores = video_lookup[(video_dir, stem)]
            audio_none = float(row["score_none"])
            audio_yem = float(row["score_yem"])
            feature_row = {
                "audio_none": audio_none,
                "audio_yem": audio_yem,
                "audio_margin": audio_yem - audio_none,
                "audio_entropy": -(audio_none * np.log(max(audio_none, 1e-8)) + audio_yem * np.log(max(audio_yem, 1e-8))),
                "video_none": video_scores["video_none"],
                "video_strong": video_scores["video_strong"],
                "video_medium": video_scores["video_medium"],
                "video_weak": video_scores["video_weak"],
                "video_feedlike": 1.0 - video_scores["video_none"],
                "video_active": video_scores["video_strong"] + video_scores["video_medium"],
                "video_strong_minus_none": video_scores["video_strong"] - video_scores["video_none"],
                "cross_audio_video": audio_yem * (1.0 - video_scores["video_none"]),
                "agree_feed": float((audio_yem >= 0.5) and ((1.0 - video_scores["video_none"]) >= 0.5)),
            }
            rows.append(
                {
                    "split": row["split"],
                    "stem": stem,
                    "source_dir": source_dir,
                    "target": int(row["target_index"]),
                    "audio_predicted_index": int(row["predicted_index"]),
                    "sample_weight": resolve_sample_weight(source_dir, source_weight_rules),
                    "features": feature_row,
                }
            )
    return rows


def rows_to_matrix(rows: Sequence[Dict[str, object]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = feature_names()
    features = np.asarray([[row["features"][name] for name in names] for row in rows], dtype=np.float32)
    targets = np.asarray([row["target"] for row in rows], dtype=np.int64)
    weights = np.asarray([row["sample_weight"] for row in rows], dtype=np.float32)
    return features, targets, weights


def evaluate_predictions(targets: np.ndarray, predictions: np.ndarray) -> Dict[str, object]:
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "positive_f1": float(f1_score(targets, predictions, average="binary", zero_division=0)),
        "confusion": confusion_matrix(targets, predictions).tolist(),
    }


def tune_threshold(probabilities: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    best = None
    for threshold in np.linspace(0.1, 0.9, 81):
        predictions = (probabilities >= threshold).astype(np.int64)
        macro_f1 = f1_score(targets, predictions, average="macro", zero_division=0)
        accuracy = accuracy_score(targets, predictions)
        if best is None or macro_f1 > best["val_macro_f1"] or (
            np.isclose(macro_f1, best["val_macro_f1"]) and accuracy > best["val_accuracy"]
        ):
            best = {
                "threshold": float(threshold),
                "val_macro_f1": float(macro_f1),
                "val_accuracy": float(accuracy),
            }
    return best


def save_comparison_plot(output_path: Path, audio_metrics: Dict[str, object], video_metrics: Dict[str, object], fusion_metrics: Dict[str, object]) -> None:
    labels = ["Audio", "Video", "Fusion"]
    accuracy_values = [audio_metrics["accuracy"], video_metrics["accuracy"], fusion_metrics["accuracy"]]
    macro_f1_values = [audio_metrics["macro_f1"], video_metrics["macro_f1"], fusion_metrics["macro_f1"]]
    positive_f1_values = [audio_metrics["positive_f1"], video_metrics["positive_f1"], fusion_metrics["positive_f1"]]

    x = np.arange(len(labels))
    width = 0.24
    figure, axis = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    axis.bar(x - width, accuracy_values, width=width, label="Accuracy", color="#4c78a8")
    axis.bar(x, macro_f1_values, width=width, label="Macro-F1", color="#54a24b")
    axis.bar(x + width, positive_f1_values, width=width, label="Positive-F1", color="#e45756")
    axis.set_xticks(x, labels)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Score")
    axis.set_title("Fast Late Fusion vs Baselines")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    video_csvs = default_video_csvs(repo_root)
    for key, path_text in parse_mapping_rules(args.video_result).items():
        video_csvs[key] = Path(path_text).expanduser().resolve()
    source_weight_rules = parse_mapping_rules(args.source_weight)
    video_lookup = load_video_lookup(video_csvs)

    train_rows = build_rows(args.audio_prediction_dir / "train_predictions.csv", video_lookup, source_weight_rules)
    val_rows = build_rows(args.audio_prediction_dir / "val_predictions.csv", video_lookup, source_weight_rules)
    test_rows = build_rows(args.audio_prediction_dir / "test_predictions.csv", video_lookup, source_weight_rules)

    X_train, y_train, w_train = rows_to_matrix(train_rows)
    X_val, y_val, _ = rows_to_matrix(val_rows)
    X_test, y_test, _ = rows_to_matrix(test_rows)

    candidates = {
        "logreg": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced")),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=42,
            n_estimators=250,
            learning_rate=0.04,
            max_depth=3,
        ),
    }

    leaderboard = []
    best_model_name = None
    best_model = None
    best_summary = None

    for name, model in candidates.items():
        if isinstance(model, Pipeline):
            model.fit(X_train, y_train, clf__sample_weight=w_train)
        else:
            model.fit(X_train, y_train, sample_weight=w_train)
        val_probabilities = model.predict_proba(X_val)[:, 1]
        threshold_summary = tune_threshold(val_probabilities, y_val)
        test_probabilities = model.predict_proba(X_test)[:, 1]
        test_predictions = (test_probabilities >= threshold_summary["threshold"]).astype(np.int64)
        test_metrics = evaluate_predictions(y_test, test_predictions)
        summary = {
            "model": name,
            **threshold_summary,
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_positive_f1": test_metrics["positive_f1"],
            "test_confusion": test_metrics["confusion"],
        }
        leaderboard.append(summary)

        if best_summary is None or threshold_summary["val_macro_f1"] > best_summary["val_macro_f1"] or (
            np.isclose(threshold_summary["val_macro_f1"], best_summary["val_macro_f1"])
            and threshold_summary["val_accuracy"] > best_summary["val_accuracy"]
        ):
            best_model_name = name
            best_model = model
            best_summary = summary

    audio_predictions = np.asarray([row["audio_predicted_index"] for row in test_rows], dtype=np.int64)
    audio_metrics = evaluate_predictions(y_test, audio_predictions)

    best_video_threshold = tune_threshold(1.0 - np.asarray([row["features"]["video_none"] for row in val_rows]), y_val)
    video_test_predictions = (
        (1.0 - np.asarray([row["features"]["video_none"] for row in test_rows])) >= best_video_threshold["threshold"]
    ).astype(np.int64)
    video_metrics = evaluate_predictions(y_test, video_test_predictions)

    fusion_test_probabilities = best_model.predict_proba(X_test)[:, 1]
    fusion_test_predictions = (fusion_test_probabilities >= best_summary["threshold"]).astype(np.int64)
    fusion_metrics = evaluate_predictions(y_test, fusion_test_predictions)

    prediction_rows = []
    for index, row in enumerate(test_rows):
        prediction_rows.append(
            {
                "split": row["split"],
                "stem": row["stem"],
                "source_dir": row["source_dir"],
                "target_index": row["target"],
                "audio_predicted_index": row["audio_predicted_index"],
                "video_predicted_index": int(video_test_predictions[index]),
                "fusion_predicted_index": int(fusion_test_predictions[index]),
                "fusion_score_yem": round(float(fusion_test_probabilities[index]), 6),
            }
        )

    leaderboard.sort(key=lambda item: (item["val_macro_f1"], item["val_accuracy"]), reverse=True)
    summary = {
        "feature_names": feature_names(),
        "source_weight_rules": source_weight_rules,
        "leaderboard": leaderboard,
        "best_model": best_model_name,
        "best_summary": best_summary,
        "test_baselines": {
            "audio": audio_metrics,
            "video_feedlike": video_metrics,
            "fusion": fusion_metrics,
        },
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0].keys()))
        writer.writeheader()
        writer.writerows(prediction_rows)
    joblib.dump(
        {
            "model_name": best_model_name,
            "model": best_model,
            "threshold": best_summary["threshold"],
            "feature_names": feature_names(),
            "source_weight_rules": source_weight_rules,
        },
        output_dir / "best_model.joblib",
    )
    save_comparison_plot(output_dir / "comparison.png", audio_metrics, video_metrics, fusion_metrics)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
