from __future__ import annotations

from typing import Sequence

import numpy as np


SECONDARY_ALLOWED_CONCEPTS = {
    "C0002978", "C0040405", "C0024485", "C0032743", "C0041618", "C1306645",
    "C1140618", "C0037949", "C0030797", "C0023216", "C0037303", "C0817096",
    "C0006141", "C0000726",
}

# --- Utilities ---

def probabilities_to_binary_predictions(
    probabilities: np.ndarray,
    threshold: float,
    max_labels_per_image: int = 100,
) -> np.ndarray:
    if max_labels_per_image <= 0:
        raise ValueError("max_labels_per_image must be > 0")
    binary_predictions = probabilities >= threshold

    no_label_rows = np.where(binary_predictions.sum(axis=1) == 0)[0]
    if len(no_label_rows) > 0:
        best_indices = probabilities[no_label_rows].argmax(axis=1)
        binary_predictions[no_label_rows, best_indices] = True

    max_labels = int(max_labels_per_image)
    for row_idx in range(binary_predictions.shape[0]):
        positive_indices = np.where(binary_predictions[row_idx])[0]
        if len(positive_indices) <= max_labels:
            continue

        top_indices = np.argpartition(probabilities[row_idx], -max_labels)[-max_labels:]
        binary_predictions[row_idx] = False
        binary_predictions[row_idx, top_indices] = True

    return binary_predictions
