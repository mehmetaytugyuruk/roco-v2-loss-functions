from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch


class CSVMetricLogger:
    def __init__(self, path: Path, fieldnames: Sequence[str]) -> None:
        self.path = path
        self.fieldnames = list(fieldnames)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()

    def log(self, row: Dict[str, object]) -> None:
        cleaned_row = {field: row.get(field, "") for field in self.fieldnames}
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writerow(cleaned_row)

def save_checkpoint(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def write_probabilities_csv(
    path: Path,
    ids: Sequence[str],
    probabilities: np.ndarray,
    index_to_label: Sequence[str],
) -> None:
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a 2D array")
    if len(ids) != probabilities.shape[0]:
        raise ValueError("ids and probabilities row count mismatch")
    if probabilities.shape[1] != len(index_to_label):
        raise ValueError(
            f"Expected {len(index_to_label)} probability columns, got {probabilities.shape[1]}"
        )
    if probabilities.min() < 0.0 or probabilities.max() > 1.0:
        raise ValueError("Probabilities must be in [0, 1]")

    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["ID"] + list(index_to_label)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row_idx, image_id in enumerate(ids):
            row_probs = [f"{float(value):.8f}" for value in probabilities[row_idx]]
            writer.writerow([image_id] + row_probs)
