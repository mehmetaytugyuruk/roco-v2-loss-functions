from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import build_thresholds  # noqa: E402
from evaluator import (  # noqa: E402
    SECONDARY_ALLOWED_CONCEPTS,
    probabilities_to_binary_predictions,
)


REQUIRED_PROBABILITY_FILES = ("val_probabilities.csv", "test_probabilities.csv")
BUCKET_ORDER = ["head", "mid", "tail", "other"]


@dataclass
class ProbabilityTable:
    ids: List[str]
    labels: List[str]
    probabilities: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate threshold strategies and compute bucket diagnostics for every experiment."
    )
    parser.add_argument("--runs-root", type=str, default="runs")
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Optional explicit experiment directories. If omitted, auto-discovers under runs-root.",
    )
    parser.add_argument("--annotations-dir", type=str, default="data/annotations")
    parser.add_argument("--val-primary-gt", type=str, default="data/annotations/valid_concepts.csv")
    parser.add_argument("--val-secondary-gt", type=str, default="data/annotations/valid_concepts_manual.csv")
    parser.add_argument("--test-primary-gt", type=str, default="data/annotations/test_concepts.csv")
    parser.add_argument("--test-secondary-gt", type=str, default="data/annotations/test_concepts_manual.csv")
    parser.add_argument("--global-threshold-start", type=float, default=0.0)
    parser.add_argument("--global-threshold-end", type=float, default=1.0)
    parser.add_argument("--global-threshold-step", type=float, default=0.01)
    parser.add_argument("--classwise-threshold-start", type=float, default=0.0)
    parser.add_argument("--classwise-threshold-end", type=float, default=1.0)
    parser.add_argument("--classwise-threshold-step", type=float, default=0.02)
    parser.add_argument("--max-labels-per-image", type=int, default=100)
    parser.add_argument("--classwise-chunk-size", type=int, default=256)
    parser.add_argument("--output-json-name", type=str, default="results.json")
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def load_probability_table(path: Path) -> ProbabilityTable:
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"Probability CSV must have at least 2 columns (ID + labels): {path}")
    if df.columns[0] != "ID":
        raise ValueError(f"First column must be 'ID': {path}")

    ids = df.iloc[:, 0].astype(str).tolist()
    labels = df.columns[1:].astype(str).tolist()
    probabilities = df.iloc[:, 1:].to_numpy(dtype=np.float32)

    if probabilities.ndim != 2:
        raise ValueError(f"Probability matrix must be 2D: {path}")
    if probabilities.shape[0] != len(ids):
        raise ValueError(f"Row mismatch in probability CSV: {path}")
    if probabilities.min() < 0.0 or probabilities.max() > 1.0:
        raise ValueError(f"Probability values must be within [0, 1]: {path}")

    return ProbabilityTable(ids=ids, labels=labels, probabilities=probabilities)


def parse_cuis(raw_value: str | float | None) -> List[str]:
    if raw_value is None:
        return []
    text = str(raw_value).strip()
    if not text or text.lower() == "nan":
        return []
    return [token.strip() for token in text.split(";") if token.strip()]


def assign_bucket(freq: int) -> str:
    if freq >= 1000:
        return "head"
    if 100 <= freq <= 999:
        return "mid"
    return "tail"


def load_train_frequencies(train_concepts_csv: Path) -> Dict[str, int]:
    df = pd.read_csv(train_concepts_csv, dtype={"ID": str, "CUIs": str})
    frequencies: Dict[str, int] = {}
    for value in df["CUIs"].fillna(""):
        for cui in parse_cuis(value):
            frequencies[cui.upper()] = frequencies.get(cui.upper(), 0) + 1
    return frequencies


class FastOfficialScorer:
    """Vectorized scorer with the same formula as the official evaluator."""

    def __init__(self, *, ids: Sequence[str], labels: Sequence[str], primary_gt_csv: Path, secondary_gt_csv: Path) -> None:
        self.ids = list(ids)
        self.labels = list(labels)
        self.id_to_index = {image_id: idx for idx, image_id in enumerate(self.ids)}
        self.label_to_index = {label.upper(): idx for idx, label in enumerate(self.labels)}

        self.primary_gt = self._load_binary_matrix(primary_gt_csv)
        self.secondary_gt = self._load_binary_matrix(secondary_gt_csv)

        self.primary_gt_count = self.primary_gt.sum(axis=1, dtype=np.int32)

        self.secondary_allowed_mask = np.array(
            [label.upper() in SECONDARY_ALLOWED_CONCEPTS for label in self.labels],
            dtype=bool,
        )
        self.secondary_gt_allowed = self.secondary_gt[:, self.secondary_allowed_mask]
        self.secondary_gt_count = self.secondary_gt_allowed.sum(axis=1, dtype=np.int32)

    def _load_binary_matrix(self, gt_csv: Path) -> np.ndarray:
        matrix = np.zeros((len(self.ids), len(self.labels)), dtype=bool)
        gt_df = pd.read_csv(gt_csv, dtype={"ID": str, "CUIs": str})

        for row in gt_df.itertuples(index=False):
            image_id = str(getattr(row, "ID", "")).strip()
            row_idx = self.id_to_index.get(image_id)
            if row_idx is None:
                continue

            raw_cuis = getattr(row, "CUIs", "")
            for cui in parse_cuis(raw_cuis):
                col_idx = self.label_to_index.get(cui.upper())
                if col_idx is not None:
                    matrix[row_idx, col_idx] = True
        return matrix

    @staticmethod
    def _score_block(pred: np.ndarray, gt: np.ndarray, gt_count: np.ndarray) -> float:
        pred_count = pred.sum(axis=1, dtype=np.int32)
        tp = np.logical_and(pred, gt).sum(axis=1, dtype=np.int32)

        denom = (pred_count + gt_count).astype(np.float32)
        f1 = np.divide(
            2.0 * tp.astype(np.float32),
            denom,
            out=np.zeros_like(denom, dtype=np.float32),
            where=denom > 0,
        )
        f1[tp == 0] = 0.0

        valid = gt_count > 0
        if not np.any(valid):
            return 0.0
        return float(f1[valid].mean())

    def score(self, binary_predictions: np.ndarray) -> Dict[str, float]:
        if binary_predictions.shape != self.primary_gt.shape:
            raise ValueError(
                f"Prediction shape mismatch: got {binary_predictions.shape}, expected {self.primary_gt.shape}"
            )
        if binary_predictions.dtype != bool:
            binary_predictions = binary_predictions.astype(bool)

        primary_score = self._score_block(binary_predictions, self.primary_gt, self.primary_gt_count)

        pred_secondary = binary_predictions[:, self.secondary_allowed_mask]
        secondary_score = self._score_block(pred_secondary, self.secondary_gt_allowed, self.secondary_gt_count)

        avg_labels = float(binary_predictions.sum(axis=1).mean())
        return {
            "primary": primary_score,
            "secondary": secondary_score,
            "avg_predicted_labels": avg_labels,
        }


def is_better(candidate: Dict[str, float], best: Dict[str, float] | None) -> bool:
    if best is None:
        return True
    if candidate["primary"] > best["primary"] + 1e-12:
        return True
    if abs(candidate["primary"] - best["primary"]) <= 1e-12:
        if candidate["secondary"] > best["secondary"] + 1e-12:
            return True
        if abs(candidate["secondary"] - best["secondary"]) <= 1e-12 and candidate["threshold"] < best["threshold"]:
            return True
    return False


def search_best_global_threshold(
    *,
    probabilities: np.ndarray,
    scorer: FastOfficialScorer,
    thresholds: Sequence[float],
    max_labels_per_image: int,
) -> Dict[str, float]:
    best_row: Dict[str, float] | None = None
    for threshold in thresholds:
        binary_predictions = probabilities_to_binary_predictions(
            probabilities=probabilities,
            threshold=float(threshold),
            max_labels_per_image=max_labels_per_image,
        )
        metrics = scorer.score(binary_predictions)
        candidate = {
            "threshold": float(threshold),
            "primary": float(metrics["primary"]),
            "secondary": float(metrics["secondary"]),
            "avg_predicted_labels": float(metrics["avg_predicted_labels"]),
        }
        if is_better(candidate, best_row):
            best_row = candidate
    if best_row is None:
        raise RuntimeError("Global threshold search produced no result")
    return best_row


def build_gt_matrix(gt_csv: Path, ids: Sequence[str], labels: Sequence[str]) -> np.ndarray:
    id_to_index = {image_id: idx for idx, image_id in enumerate(ids)}
    label_to_index = {label.upper(): idx for idx, label in enumerate(labels)}
    gt = np.zeros((len(ids), len(labels)), dtype=bool)

    gt_df = pd.read_csv(gt_csv, dtype={"ID": str, "CUIs": str})
    for row in gt_df.itertuples(index=False):
        image_id = str(getattr(row, "ID", "")).strip()
        row_idx = id_to_index.get(image_id)
        if row_idx is None:
            continue
        for cui in parse_cuis(getattr(row, "CUIs", "")):
            col_idx = label_to_index.get(cui.upper())
            if col_idx is not None:
                gt[row_idx, col_idx] = True
    return gt


def fit_classwise_thresholds(
    *,
    probabilities: np.ndarray,
    ground_truth: np.ndarray,
    thresholds: Sequence[float],
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if probabilities.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch in classwise fitting: probabilities={probabilities.shape}, ground_truth={ground_truth.shape}"
        )

    num_samples, num_classes = probabilities.shape
    if num_samples == 0 or num_classes == 0:
        raise ValueError("Empty probability matrix")

    threshold_grid = np.array(list(thresholds), dtype=np.float32)
    support = ground_truth.sum(axis=0, dtype=np.int32)

    best_thresholds = np.ones(num_classes, dtype=np.float32)
    best_f1 = np.zeros(num_classes, dtype=np.float32)

    for start in range(0, num_classes, chunk_size):
        end = min(start + chunk_size, num_classes)
        probs_chunk = probabilities[:, start:end]
        gt_chunk = ground_truth[:, start:end]
        support_chunk = support[start:end]
        has_positive = support_chunk > 0

        local_best_thresholds = np.ones(end - start, dtype=np.float32)
        local_best_f1 = np.full(end - start, -1.0, dtype=np.float32)

        if not np.any(has_positive):
            best_thresholds[start:end] = local_best_thresholds
            best_f1[start:end] = 0.0
            continue

        for threshold in threshold_grid:
            pred = probs_chunk >= threshold
            tp = np.logical_and(pred, gt_chunk).sum(axis=0, dtype=np.int32)
            pred_pos = pred.sum(axis=0, dtype=np.int32)
            fp = pred_pos - tp
            fn = support_chunk - tp

            denom = (2 * tp + fp + fn).astype(np.float32)
            f1 = np.divide(
                2.0 * tp.astype(np.float32),
                denom,
                out=np.zeros_like(denom, dtype=np.float32),
                where=denom > 0,
            )

            improved = has_positive & (f1 > local_best_f1 + 1e-12)
            tied = has_positive & (np.abs(f1 - local_best_f1) <= 1e-12) & (threshold < local_best_thresholds)
            update = improved | tied

            local_best_f1[update] = f1[update]
            local_best_thresholds[update] = threshold

        best_thresholds[start:end] = np.where(has_positive, local_best_thresholds, 1.0)
        best_f1[start:end] = np.where(has_positive, local_best_f1, 0.0)

    return best_thresholds, best_f1, support


def probabilities_to_binary_predictions_classwise(
    *,
    probabilities: np.ndarray,
    class_thresholds: np.ndarray,
    max_labels_per_image: int,
) -> np.ndarray:
    if max_labels_per_image <= 0:
        raise ValueError("max_labels_per_image must be > 0")
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be 2D")
    if class_thresholds.ndim != 1:
        raise ValueError("class_thresholds must be 1D")
    if probabilities.shape[1] != class_thresholds.shape[0]:
        raise ValueError("class_thresholds length must match probability columns")

    binary_predictions = probabilities >= class_thresholds[np.newaxis, :]

    no_label_rows = np.where(binary_predictions.sum(axis=1) == 0)[0]
    if len(no_label_rows) > 0:
        best_indices = probabilities[no_label_rows].argmax(axis=1)
        binary_predictions[no_label_rows, best_indices] = True

    too_many_rows = np.where(binary_predictions.sum(axis=1) > int(max_labels_per_image))[0]
    for row_idx in too_many_rows:
        top_indices = np.argpartition(probabilities[row_idx], -int(max_labels_per_image))[-int(max_labels_per_image):]
        binary_predictions[row_idx] = False
        binary_predictions[row_idx, top_indices] = True

    return binary_predictions


def compute_image_level_stats(
    *,
    binary_predictions: np.ndarray,
    ground_truth: np.ndarray,
) -> Dict[str, float]:
    tp_per_image = np.logical_and(binary_predictions, ground_truth).sum(axis=1)
    fp_per_image = np.logical_and(binary_predictions, np.logical_not(ground_truth)).sum(axis=1)
    fn_per_image = np.logical_and(np.logical_not(binary_predictions), ground_truth).sum(axis=1)
    true_labels_per_image = ground_truth.sum(axis=1)

    return {
        "avg_true_labels_per_image": float(true_labels_per_image.mean()),
        "avg_tp_per_image": float(tp_per_image.mean()),
        "avg_fp_per_image": float(fp_per_image.mean()),
        "avg_fn_per_image": float(fn_per_image.mean()),
    }


def compute_bucket_rows(
    *,
    binary_predictions: np.ndarray,
    ground_truth: np.ndarray,
    raw_probabilities: np.ndarray,
    labels: Sequence[str],
    train_frequencies: Dict[str, int],
) -> tuple[list[Dict[str, object]], Dict[str, float]]:
    tp = np.logical_and(binary_predictions, ground_truth).sum(axis=0)
    fp = np.logical_and(binary_predictions, np.logical_not(ground_truth)).sum(axis=0)
    fn = np.logical_and(np.logical_not(binary_predictions), ground_truth).sum(axis=0)
    predicted_support = binary_predictions.sum(axis=0)
    support = ground_truth.sum(axis=0)
    neg_support = ground_truth.shape[0] - support

    precision_denom = tp + fp
    recall_denom = support

    precision = np.divide(
        tp,
        precision_denom,
        out=np.zeros_like(tp, dtype=np.float32),
        where=precision_denom > 0,
    )
    recall = np.divide(
        tp,
        recall_denom,
        out=np.zeros_like(tp, dtype=np.float32),
        where=recall_denom > 0,
    )
    f1_denom = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        f1_denom,
        out=np.zeros_like(precision, dtype=np.float32),
        where=f1_denom > 0,
    )

    mean_positive_score = np.divide(
        (raw_probabilities * ground_truth.astype(np.float32)).sum(axis=0),
        support,
        out=np.zeros_like(tp, dtype=np.float32),
        where=support > 0,
    )
    mean_negative_score = np.divide(
        (raw_probabilities * np.logical_not(ground_truth).astype(np.float32)).sum(axis=0),
        neg_support,
        out=np.zeros_like(tp, dtype=np.float32),
        where=neg_support > 0,
    )
    score_margin = mean_positive_score - mean_negative_score

    bucket_labels = np.array([assign_bucket(int(train_frequencies.get(cui.upper(), 0))) for cui in labels])

    rows: list[Dict[str, object]] = []
    for bucket in BUCKET_ORDER:
        mask = (bucket_labels == bucket) & (support > 0)
        if mask.sum() == 0:
            continue
        predicted_classes = int((predicted_support[mask] > 0).sum())
        n_seen_classes = int(mask.sum())
        rows.append(
            {
                "bucket": bucket,
                "n_seen_classes": n_seen_classes,
                "macro_precision": float(np.mean(precision[mask])),
                "macro_recall": float(np.mean(recall[mask])),
                "macro_f1": float(np.mean(f1[mask])),
                "predicted_classes": predicted_classes,
                "coverage_pct": float(predicted_classes / n_seen_classes) if n_seen_classes > 0 else 0.0,
                "nonzero_f1_classes": int((f1[mask] > 0).sum()),
                "mean_positive_score": float(np.mean(mean_positive_score[mask])),
                "mean_negative_score": float(np.mean(mean_negative_score[mask])),
                "score_margin": float(np.mean(score_margin[mask])),
            }
        )

    seen_mask = support > 0
    seen_count = int(seen_mask.sum())
    seen_predicted = int((predicted_support[seen_mask] > 0).sum())
    overall = {
        "seen_classes": seen_count,
        "predicted_classes": seen_predicted,
        "coverage_pct": float(seen_predicted / seen_count) if seen_count > 0 else 0.0,
        "nonzero_f1_classes": int((f1[seen_mask] > 0).sum()),
        "mean_positive_score": float(np.mean(mean_positive_score[seen_mask])) if seen_count > 0 else 0.0,
        "mean_negative_score": float(np.mean(mean_negative_score[seen_mask])) if seen_count > 0 else 0.0,
        "score_margin": float(np.mean(score_margin[seen_mask])) if seen_count > 0 else 0.0,
    }
    return rows, overall


def build_split_payload(
    *,
    probabilities: np.ndarray,
    binary_predictions: np.ndarray,
    ground_truth: np.ndarray,
    labels: Sequence[str],
    train_frequencies: Dict[str, int],
    official_metrics: Dict[str, float],
) -> tuple[Dict[str, object], list[Dict[str, object]]]:
    image_level = compute_image_level_stats(
        binary_predictions=binary_predictions,
        ground_truth=ground_truth,
    )
    bucket_rows, overall = compute_bucket_rows(
        binary_predictions=binary_predictions,
        ground_truth=ground_truth,
        raw_probabilities=probabilities,
        labels=labels,
        train_frequencies=train_frequencies,
    )

    split_payload = {
        "official_primary": float(official_metrics["primary"]),
        "official_secondary": float(official_metrics["secondary"]),
        "avg_predicted_labels": float(official_metrics["avg_predicted_labels"]),
        "image_level": image_level,
        "overall": overall,
        "buckets": bucket_rows,
    }
    return split_payload, bucket_rows


def collect_experiment_dirs(runs_root: Path, explicit_experiments: Sequence[str] | None) -> List[Path]:
    if explicit_experiments:
        resolved = []
        for exp in explicit_experiments:
            path = resolve_path(exp)
            if not path.exists() or not path.is_dir():
                raise FileNotFoundError(f"Experiment directory not found: {path}")
            resolved.append(path)
        return sorted(resolved)

    experiments: List[Path] = []
    for group_dir in sorted(runs_root.iterdir()):
        if not group_dir.is_dir():
            continue
        for exp_dir in sorted(group_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            if all((exp_dir / file_name).exists() for file_name in REQUIRED_PROBABILITY_FILES):
                experiments.append(exp_dir)
    return experiments


def process_experiment(
    *,
    exp_dir: Path,
    annotations_dir: Path,
    train_frequencies: Dict[str, int],
    val_primary_gt: Path,
    val_secondary_gt: Path,
    test_primary_gt: Path,
    test_secondary_gt: Path,
    global_thresholds: Sequence[float],
    classwise_thresholds: Sequence[float],
    max_labels_per_image: int,
    classwise_chunk_size: int,
) -> Dict[str, object]:
    val_table = load_probability_table(exp_dir / "val_probabilities.csv")
    test_table = load_probability_table(exp_dir / "test_probabilities.csv")

    if val_table.labels != test_table.labels:
        raise ValueError(f"Label order mismatch between val/test probability files in {exp_dir}")

    val_primary_matrix = build_gt_matrix(val_primary_gt, val_table.ids, val_table.labels)
    test_primary_matrix = build_gt_matrix(test_primary_gt, test_table.ids, test_table.labels)

    val_scorer = FastOfficialScorer(
        ids=val_table.ids,
        labels=val_table.labels,
        primary_gt_csv=val_primary_gt,
        secondary_gt_csv=val_secondary_gt,
    )
    test_scorer = FastOfficialScorer(
        ids=test_table.ids,
        labels=test_table.labels,
        primary_gt_csv=test_primary_gt,
        secondary_gt_csv=test_secondary_gt,
    )

    global_best_val = search_best_global_threshold(
        probabilities=val_table.probabilities,
        scorer=val_scorer,
        thresholds=global_thresholds,
        max_labels_per_image=max_labels_per_image,
    )
    best_global_threshold = float(global_best_val["threshold"])

    global_val_binary = probabilities_to_binary_predictions(
        probabilities=val_table.probabilities,
        threshold=best_global_threshold,
        max_labels_per_image=max_labels_per_image,
    )
    global_test_binary = probabilities_to_binary_predictions(
        probabilities=test_table.probabilities,
        threshold=best_global_threshold,
        max_labels_per_image=max_labels_per_image,
    )
    global_val_metrics = val_scorer.score(global_val_binary)
    global_test_metrics = test_scorer.score(global_test_binary)

    class_threshold_vector, _, _ = fit_classwise_thresholds(
        probabilities=val_table.probabilities,
        ground_truth=val_primary_matrix,
        thresholds=classwise_thresholds,
        chunk_size=classwise_chunk_size,
    )

    classwise_val_binary = probabilities_to_binary_predictions_classwise(
        probabilities=val_table.probabilities,
        class_thresholds=class_threshold_vector,
        max_labels_per_image=max_labels_per_image,
    )
    classwise_test_binary = probabilities_to_binary_predictions_classwise(
        probabilities=test_table.probabilities,
        class_thresholds=class_threshold_vector,
        max_labels_per_image=max_labels_per_image,
    )
    classwise_val_metrics = val_scorer.score(classwise_val_binary)
    classwise_test_metrics = test_scorer.score(classwise_test_binary)

    validation_payload, validation_bucket_rows = build_split_payload(
        probabilities=val_table.probabilities,
        binary_predictions=global_val_binary,
        ground_truth=val_primary_matrix,
        labels=val_table.labels,
        train_frequencies=train_frequencies,
        official_metrics=global_val_metrics,
    )
    test_payload, test_bucket_rows = build_split_payload(
        probabilities=test_table.probabilities,
        binary_predictions=global_test_binary,
        ground_truth=test_primary_matrix,
        labels=test_table.labels,
        train_frequencies=train_frequencies,
        official_metrics=global_test_metrics,
    )

    payload = {
        "experiment": exp_dir.name,
        "thresholding": {
            "global": {
                "best_threshold": best_global_threshold,
                "validation": {
                    "primary": float(global_val_metrics["primary"]),
                    "secondary": float(global_val_metrics["secondary"]),
                    "avg_predicted_labels": float(global_val_metrics["avg_predicted_labels"]),
                },
                "test": {
                    "primary": float(global_test_metrics["primary"]),
                    "secondary": float(global_test_metrics["secondary"]),
                    "avg_predicted_labels": float(global_test_metrics["avg_predicted_labels"]),
                },
            },
            "classwise": {
                "method": "per-label-best-f1-on-validation",
                "validation": {
                    "primary": float(classwise_val_metrics["primary"]),
                    "secondary": float(classwise_val_metrics["secondary"]),
                    "avg_predicted_labels": float(classwise_val_metrics["avg_predicted_labels"]),
                },
                "test": {
                    "primary": float(classwise_test_metrics["primary"]),
                    "secondary": float(classwise_test_metrics["secondary"]),
                    "avg_predicted_labels": float(classwise_test_metrics["avg_predicted_labels"]),
                },
            },
        },
        "selected_strategy": "global",
        "selected_threshold": best_global_threshold,
        "splits": {
            "validation": validation_payload,
            "test": test_payload,
        },
    }

    return payload


def main() -> None:
    args = parse_args()

    runs_root = resolve_path(args.runs_root)
    annotations_dir = resolve_path(args.annotations_dir)
    val_primary_gt = resolve_path(args.val_primary_gt)
    val_secondary_gt = resolve_path(args.val_secondary_gt)
    test_primary_gt = resolve_path(args.test_primary_gt)
    test_secondary_gt = resolve_path(args.test_secondary_gt)
    train_frequencies = load_train_frequencies(annotations_dir / "train_concepts.csv")

    experiment_dirs = collect_experiment_dirs(runs_root, args.experiments)
    if not experiment_dirs:
        raise RuntimeError(f"No experiment directories found under: {runs_root}")

    global_thresholds = build_thresholds(
        start=float(args.global_threshold_start),
        end=float(args.global_threshold_end),
        step=float(args.global_threshold_step),
    )
    classwise_thresholds = build_thresholds(
        start=float(args.classwise_threshold_start),
        end=float(args.classwise_threshold_end),
        step=float(args.classwise_threshold_step),
    )

    print(
        f"Found {len(experiment_dirs)} experiments | "
        f"global_grid={len(global_thresholds)} thresholds | "
        f"classwise_grid={len(classwise_thresholds)} thresholds"
    )

    for exp_dir in experiment_dirs:
        payload = process_experiment(
            exp_dir=exp_dir,
            annotations_dir=annotations_dir,
            train_frequencies=train_frequencies,
            val_primary_gt=val_primary_gt,
            val_secondary_gt=val_secondary_gt,
            test_primary_gt=test_primary_gt,
            test_secondary_gt=test_secondary_gt,
            global_thresholds=global_thresholds,
            classwise_thresholds=classwise_thresholds,
            max_labels_per_image=int(args.max_labels_per_image),
            classwise_chunk_size=int(args.classwise_chunk_size),
        )

        output_path = exp_dir / args.output_json_name
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

        global_test = payload["thresholding"]["global"]["test"]
        classwise_test = payload["thresholding"]["classwise"]["test"]
        print(
            f"[{exp_dir.name}] "
            f"global(test)={global_test['primary']:.6f}/{global_test['secondary']:.6f} | "
            f"classwise(test)={classwise_test['primary']:.6f}/{classwise_test['secondary']:.6f} | "
            f"saved={output_path.name}"
        )

    print("Done.")


if __name__ == "__main__":
    main()
