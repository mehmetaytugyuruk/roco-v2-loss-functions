from __future__ import annotations

import copy
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List

import yaml


REQUIRED_TOP_LEVEL_KEYS = (
    "num_classes",
    "image_size",
    "batch_size",
    "epochs",
    "lr",
    "weight_decay",
    "warmup_epochs",
    "min_lr",
    "grad_clip",
    "num_workers",
)


DEFAULTS: Dict[str, Any] = {
    "seed": 42,
    "amp": True,
    "pin_memory": True,
    "pretrained": True,
    "color_jitter_probability": 0.5,
    "max_labels_per_image": 100,
    "auto_extract_data_zip": False,
    "progress_bar": True,
    "num_workers_if_supported": 6,
    "colab_safe_num_workers": 6,
    "torch_compile": False,
    "max_train_samples": None,
    "max_valid_samples": None,
    "max_test_samples": None,
    "paths": {
        "annotations_dir": "data/annotations",
        "images_dir": "data/images",
        "data_zip": "",
    },
    "outputs": {
        "best_model": "best_model.pth",
        "last_model": "last_model.pth",
        "train_log": "train_log.csv",
        "val_probabilities": "val_probabilities.csv",
        "test_probabilities": "test_probabilities.csv",
    },
}


def _deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")

    missing = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in loaded]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    config = _deep_merge(DEFAULTS, loaded)
    return config


def apply_overrides(config: Dict[str, Any], *, output_dir: str | None, seed: int | None, num_workers: int | None) -> Dict[str, Any]:
    updated = copy.deepcopy(config)
    if output_dir is not None:
        updated["output_dir"] = output_dir
    if seed is not None:
        updated["seed"] = seed
    if num_workers is not None:
        updated["num_workers"] = num_workers
    return updated


def build_thresholds(start: float, end: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("threshold_step must be > 0")
    if end < start:
        raise ValueError("threshold_end must be >= threshold_start")

    start_dec = Decimal(str(start))
    end_dec = Decimal(str(end))
    step_dec = Decimal(str(step))

    total = (end_dec - start_dec) / step_dec
    if total != total.to_integral_value():
        raise ValueError("Threshold grid must be evenly divisible by threshold_step")

    count = int(total) + 1
    decimals = max(0, -step_dec.as_tuple().exponent)
    return [float(round(start_dec + Decimal(idx) * step_dec, decimals)) for idx in range(count)]


def resolve_output_dir(config: Dict[str, Any], project_root: Path) -> Path:
    output_dir = config.get("output_dir", ".")
    return (project_root / output_dir).resolve() if not Path(output_dir).is_absolute() else Path(output_dir)


def resolve_path(project_root: Path, path_value: str) -> Path:
    path = Path(path_value)
    return (project_root / path).resolve() if not path.is_absolute() else path


def resolve_paths(project_root: Path, path_items: Dict[str, str]) -> Dict[str, Path]:
    return {key: resolve_path(project_root, value) for key, value in path_items.items()}
