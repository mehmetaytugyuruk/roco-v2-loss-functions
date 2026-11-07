from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluator import probabilities_to_binary_predictions  # noqa: E402


@dataclass(frozen=True)
class ProbabilityTable:
    ids: list[str]
    labels: list[str]
    probabilities: np.ndarray


@dataclass(frozen=True)
class ModelSpec:
    name: str
    run_dir: Path
    threshold: float


DEFAULT_COMPARISONS = (
    "A1=runs/A/A1_42:A3=runs/A/A3_42",
    "A3=runs/A/A3_42:B2=runs/B/B2_42",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired Wilcoxon signed-rank tests over per-image primary F1 "
            "scores from thresholded ROCOv2 test predictions."
        )
    )
    parser.add_argument(
        "--comparison",
        action="append",
        default=None,
        metavar="LEFT=RUN_DIR:RIGHT=RUN_DIR",
        help=(
            "Comparison to evaluate. The selected threshold is read from each "
            "run's results.json unless overridden with @threshold, e.g. "
            "A1=runs/A/A1_42@0.37:A3=runs/A/A3_42@0.74. Can be repeated."
        ),
    )
    parser.add_argument("--test-gt", default="data/annotations/test_concepts.csv")
    parser.add_argument("--probability-file", default="test_probabilities.csv")
    parser.add_argument("--results-file", default="results.json")
    parser.add_argument("--max-labels-per-image", type=int, default=100)
    parser.add_argument(
        "--alternative",
        choices=("two-sided", "less", "greater"),
        default="two-sided",
        help=(
            "Wilcoxon alternative hypothesis applied as scipy.stats.wilcoxon(left, right). "
            "Use 'less' only for a pre-specified directional test that left < right."
        ),
    )
    parser.add_argument(
        "--zero-method",
        choices=("wilcox", "pratt", "zsplit"),
        default="wilcox",
        help="How zero paired differences are handled by the Wilcoxon test.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path for writing machine-readable comparison results.",
    )
    return parser.parse_args()


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def parse_cuis(raw_value: str | float | None) -> list[str]:
    if raw_value is None:
        return []
    text = str(raw_value).strip()
    if not text or text.lower() == "nan":
        return []
    return [token.strip().upper() for token in text.split(";") if token.strip()]


def load_probability_table(path: Path) -> ProbabilityTable:
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"Probability CSV must have at least ID + one label column: {path}")
    if df.columns[0] != "ID":
        raise ValueError(f"First probability CSV column must be 'ID': {path}")

    ids = df.iloc[:, 0].astype(str).tolist()
    labels = df.columns[1:].astype(str).str.upper().tolist()
    probabilities = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    if probabilities.min() < 0.0 or probabilities.max() > 1.0:
        raise ValueError(f"Probability values must be in [0, 1]: {path}")
    return ProbabilityTable(ids=ids, labels=labels, probabilities=probabilities)


def load_selected_threshold(run_dir: Path, results_file: str) -> float:
    path = run_dir / results_file
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "selected_threshold" in payload:
        return float(payload["selected_threshold"])
    return float(payload["thresholding"]["global"]["best_threshold"])


def parse_model_spec(raw: str, results_file: str) -> ModelSpec:
    if "=" not in raw:
        raise ValueError(f"Model spec must be NAME=RUN_DIR or NAME=RUN_DIR@THRESHOLD: {raw}")
    name, value = raw.split("=", 1)
    name = name.strip()
    value = value.strip()
    if not name:
        raise ValueError(f"Model name is empty in spec: {raw}")

    threshold: float | None = None
    run_dir_value = value
    if "@" in value:
        run_dir_value, threshold_value = value.rsplit("@", 1)
        threshold = float(threshold_value)

    run_dir = resolve_path(run_dir_value)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if threshold is None:
        threshold = load_selected_threshold(run_dir, results_file)
    return ModelSpec(name=name, run_dir=run_dir, threshold=threshold)


def parse_comparison(raw: str, results_file: str) -> tuple[ModelSpec, ModelSpec]:
    parts = raw.split(":")
    if len(parts) != 2:
        raise ValueError(f"Comparison must have exactly two model specs separated by ':': {raw}")
    return parse_model_spec(parts[0], results_file), parse_model_spec(parts[1], results_file)


def build_ground_truth_matrix(gt_csv: Path, ids: Sequence[str], labels: Sequence[str]) -> np.ndarray:
    id_to_index = {image_id: idx for idx, image_id in enumerate(ids)}
    label_to_index = {label.upper(): idx for idx, label in enumerate(labels)}
    ground_truth = np.zeros((len(ids), len(labels)), dtype=bool)

    df = pd.read_csv(gt_csv, dtype={"ID": str, "CUIs": str})
    for row in df.itertuples(index=False):
        image_id = str(getattr(row, "ID", "")).strip()
        row_idx = id_to_index.get(image_id)
        if row_idx is None:
            continue
        for cui in parse_cuis(getattr(row, "CUIs", "")):
            col_idx = label_to_index.get(cui)
            if col_idx is not None:
                ground_truth[row_idx, col_idx] = True
    return ground_truth


def image_level_f1(binary_predictions: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    if binary_predictions.shape != ground_truth.shape:
        raise ValueError(
            f"Prediction/ground-truth shape mismatch: {binary_predictions.shape} != {ground_truth.shape}"
        )

    tp = np.logical_and(binary_predictions, ground_truth).sum(axis=1, dtype=np.int32)
    pred_count = binary_predictions.sum(axis=1, dtype=np.int32)
    gt_count = ground_truth.sum(axis=1, dtype=np.int32)
    denom = (pred_count + gt_count).astype(np.float32)
    f1 = np.divide(
        2.0 * tp.astype(np.float32),
        denom,
        out=np.zeros_like(denom, dtype=np.float32),
        where=denom > 0,
    )
    f1[tp == 0] = 0.0
    return f1[gt_count > 0]


def load_model_f1(
    *,
    spec: ModelSpec,
    reference_ids: Sequence[str],
    reference_labels: Sequence[str],
    ground_truth: np.ndarray,
    probability_file: str,
    max_labels_per_image: int,
) -> np.ndarray:
    table = load_probability_table(spec.run_dir / probability_file)
    if table.ids != list(reference_ids):
        raise ValueError(f"Image ID order mismatch for {spec.name}: {spec.run_dir}")
    if table.labels != [label.upper() for label in reference_labels]:
        raise ValueError(f"Label column mismatch for {spec.name}: {spec.run_dir}")

    binary_predictions = probabilities_to_binary_predictions(
        table.probabilities,
        threshold=spec.threshold,
        max_labels_per_image=max_labels_per_image,
    )
    return image_level_f1(binary_predictions, ground_truth)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    m = len(p_values)
    order = np.argsort(np.asarray(p_values, dtype=float))
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, original_index in enumerate(order):
        candidate = (m - rank) * float(p_values[original_index])
        running_max = max(running_max, candidate)
        adjusted[original_index] = min(running_max, 1.0)
    return adjusted.tolist()


def format_p_value(value: float) -> str:
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}"


def run_comparison(
    *,
    left: ModelSpec,
    right: ModelSpec,
    gt_csv: Path,
    probability_file: str,
    max_labels_per_image: int,
    alternative: str,
    zero_method: str,
) -> dict[str, object]:
    left_table = load_probability_table(left.run_dir / probability_file)
    ground_truth = build_ground_truth_matrix(gt_csv, left_table.ids, left_table.labels)
    left_f1 = load_model_f1(
        spec=left,
        reference_ids=left_table.ids,
        reference_labels=left_table.labels,
        ground_truth=ground_truth,
        probability_file=probability_file,
        max_labels_per_image=max_labels_per_image,
    )
    right_f1 = load_model_f1(
        spec=right,
        reference_ids=left_table.ids,
        reference_labels=left_table.labels,
        ground_truth=ground_truth,
        probability_file=probability_file,
        max_labels_per_image=max_labels_per_image,
    )

    if left_f1.shape != right_f1.shape:
        raise ValueError(f"Per-image F1 vector mismatch: {left.name} vs {right.name}")

    diff = right_f1 - left_f1
    test = wilcoxon(left_f1, right_f1, zero_method=zero_method, alternative=alternative, method="auto")

    return {
        "comparison": f"{left.name} vs {right.name}",
        "left_model": left.name,
        "right_model": right.name,
        "left_run_dir": str(left.run_dir.relative_to(PROJECT_ROOT)),
        "right_run_dir": str(right.run_dir.relative_to(PROJECT_ROOT)),
        "left_threshold": left.threshold,
        "right_threshold": right.threshold,
        "n_images": int(left_f1.shape[0]),
        "left_mean_f1": float(left_f1.mean()),
        "right_mean_f1": float(right_f1.mean()),
        "mean_delta_right_minus_left": float(diff.mean()),
        "median_delta_right_minus_left": float(np.median(diff)),
        "right_better_images": int(np.sum(diff > 0)),
        "left_better_images": int(np.sum(diff < 0)),
        "tied_images": int(np.sum(diff == 0)),
        "nonzero_differences": int(np.sum(diff != 0)),
        "wilcoxon_statistic": float(test.statistic),
        "p_value": float(test.pvalue),
        "alternative": alternative,
        "zero_method": zero_method,
    }


def print_markdown_table(rows: Sequence[dict[str, object]]) -> None:
    print("| Comparison | Left mean F1 | Right mean F1 | Delta | Wins / Losses / Ties | p-value | Holm p |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        wins = f"{row['right_better_images']} / {row['left_better_images']} / {row['tied_images']}"
        print(
            "| {comparison} | {left:.4f} | {right:.4f} | {delta:+.4f} | {wins} | {p} | {holm} |".format(
                comparison=row["comparison"],
                left=float(row["left_mean_f1"]),
                right=float(row["right_mean_f1"]),
                delta=float(row["mean_delta_right_minus_left"]),
                wins=wins,
                p=format_p_value(float(row["p_value"])),
                holm=format_p_value(float(row["holm_p_value"])),
            )
        )


def main() -> None:
    args = parse_args()
    comparison_values = args.comparison if args.comparison is not None else list(DEFAULT_COMPARISONS)
    comparisons = [parse_comparison(raw, args.results_file) for raw in comparison_values]
    gt_csv = resolve_path(args.test_gt)

    rows = [
        run_comparison(
            left=left,
            right=right,
            gt_csv=gt_csv,
            probability_file=args.probability_file,
            max_labels_per_image=args.max_labels_per_image,
            alternative=args.alternative,
            zero_method=args.zero_method,
        )
        for left, right in comparisons
    ]

    adjusted = holm_adjust([float(row["p_value"]) for row in rows])
    for row, adjusted_p in zip(rows, adjusted):
        row["holm_p_value"] = float(adjusted_p)

    print_markdown_table(rows)

    if args.output_json:
        output_path = resolve_path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
