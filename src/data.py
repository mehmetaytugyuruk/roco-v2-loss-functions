from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


SPLITS = ("train", "valid", "test")


@dataclass
class DataBundle:
    metadata: Dict[str, pd.DataFrame]
    datasets: Dict[str, Dataset]
    loaders: Dict[str, DataLoader]
    label_to_index: Dict[str, int]
    index_to_label: List[str]
    num_workers: int


def parse_cuis(raw_value: str | float | None) -> List[str]:
    if raw_value is None:
        return []
    text = str(raw_value).strip()
    if not text or text.lower() == "nan":
        return []
    return [token.strip() for token in text.split(";") if token.strip()]


def build_class_vocabulary(train_concepts_csv: Path, expected_num_classes: int) -> tuple[Dict[str, int], List[str]]:
    train_df = pd.read_csv(train_concepts_csv, dtype={"ID": str, "CUIs": str})
    all_cuis = set()
    for value in train_df["CUIs"].fillna(""):
        all_cuis.update(parse_cuis(value))

    index_to_label = sorted(all_cuis)
    if len(index_to_label) != expected_num_classes:
        raise ValueError(
            f"Expected {expected_num_classes} classes from train split, got {len(index_to_label)}"
        )

    label_to_index = {cui: idx for idx, cui in enumerate(index_to_label)}
    return label_to_index, index_to_label


def _split_image_ids(image_dir: Path) -> set[str]:
    return {
        path.stem
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".jpg"
    }


def validate_split_integrity(annotations_dir: Path, images_root: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for split in SPLITS:
        csv_path = annotations_dir / f"{split}_concepts.csv"
        image_dir = images_root / split
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing annotation file: {csv_path}")
        if not image_dir.exists():
            raise FileNotFoundError(f"Missing image directory: {image_dir}")

        split_df = pd.read_csv(csv_path, dtype={"ID": str, "CUIs": str})
        ids_from_csv = set(split_df["ID"].astype(str))
        ids_from_images = _split_image_ids(image_dir)

        missing_images = sorted(ids_from_csv - ids_from_images)
        extra_images = sorted(ids_from_images - ids_from_csv)
        if missing_images:
            raise ValueError(
                f"{split}: {len(missing_images)} IDs are missing image files. Example: {missing_images[:3]}"
            )
        if extra_images:
            raise ValueError(
                f"{split}: {len(extra_images)} extra image files not found in annotations. Example: {extra_images[:3]}"
            )
        if split_df["ID"].duplicated().any():
            dupes = split_df.loc[split_df["ID"].duplicated(), "ID"].head(3).tolist()
            raise ValueError(f"{split}: duplicate annotation IDs detected. Example: {dupes}")

        counts[split] = len(split_df)

    return counts


def _build_split_metadata(
    split: str,
    concepts_csv: Path,
    images_root: Path,
    label_to_index: Dict[str, int],
    *,
    include_labels: bool,
    max_samples: int | None,
) -> pd.DataFrame:
    split_df = pd.read_csv(concepts_csv, dtype={"ID": str, "CUIs": str})

    records = []
    for row in split_df.itertuples(index=False):
        image_id = str(row.ID).strip()
        raw_cuis = parse_cuis(getattr(row, "CUIs", ""))

        if include_labels:
            unknown = [cui for cui in raw_cuis if cui not in label_to_index]
            if unknown:
                raise ValueError(
                    f"Found unknown CUI(s) in {split}: {unknown[:3]} (image_id={image_id})"
                )
            label_indices = sorted({label_to_index[cui] for cui in raw_cuis})
            cui_list = raw_cuis
        else:
            label_indices = []
            cui_list = []

        records.append(
            {
                "id": image_id,
                "split": split,
                "image_path": str(images_root / split / f"{image_id}.jpg"),
                "cui_list": cui_list,
                "label_indices": label_indices,
            }
        )

    metadata = pd.DataFrame.from_records(records, columns=["id", "split", "image_path", "cui_list", "label_indices"])
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError(f"max_samples for {split} must be > 0 when provided")
        metadata = metadata.head(max_samples).reset_index(drop=True)

    return metadata


def build_metadata_tables(
    annotations_dir: Path,
    images_root: Path,
    label_to_index: Dict[str, int],
    *,
    max_train_samples: int | None,
    max_valid_samples: int | None,
    max_test_samples: int | None,
) -> Dict[str, pd.DataFrame]:
    return {
        "train": _build_split_metadata(
            split="train",
            concepts_csv=annotations_dir / "train_concepts.csv",
            images_root=images_root,
            label_to_index=label_to_index,
            include_labels=True,
            max_samples=max_train_samples,
        ),
        "valid": _build_split_metadata(
            split="valid",
            concepts_csv=annotations_dir / "valid_concepts.csv",
            images_root=images_root,
            label_to_index=label_to_index,
            include_labels=True,
            max_samples=max_valid_samples,
        ),
        "test": _build_split_metadata(
            split="test",
            concepts_csv=annotations_dir / "test_concepts.csv",
            images_root=images_root,
            label_to_index=label_to_index,
            include_labels=False,
            max_samples=max_test_samples,
        ),
    }


class RocoMultiLabelDataset(Dataset):
    def __init__(self, metadata: pd.DataFrame, num_classes: int, transform) -> None:
        expected_columns = {"id", "split", "image_path", "cui_list", "label_indices"}
        missing = expected_columns.difference(metadata.columns)
        if missing:
            raise ValueError(f"Metadata missing columns: {sorted(missing)}")

        self.ids = metadata["id"].tolist()
        self.image_paths = metadata["image_path"].tolist()
        self.label_indices = metadata["label_indices"].tolist()
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict:
        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)

        target = torch.zeros(self.num_classes, dtype=torch.float32)
        indices = self.label_indices[index]
        if indices:
            target[indices] = 1.0

        return {
            "id": self.ids[index],
            "image": image,
            "target": target,
        }


def compute_sample_weights(
    metadata: pd.DataFrame,
    train_frequencies: Dict[str, int],
    f_ref: float = 1000.0,
    gamma: float = 0.5,
    w_max: float = 5.0,
) -> np.ndarray:
    """Compute per-image sampling weights for class-balanced oversampling.

    Weight formula (per image i):
        r_c  = clip( (f_ref / f_c)^gamma, 0, w_max )   # rarity of class c
        w_i  = clip( max_{c in labels_i}(r_c), 1, w_max )  # max rarity

    With f_ref=1000 (head threshold) and gamma=0.5:
        - Head-only image  (f_c=10000) : w ≈ 1.0  (no boost)
        - Upper-mid image  (f_c=300)   : w ≈ 1.83
        - Upper-mid image  (f_c=100)   : w ≈ 3.16
        - Lower-mid image  (f_c=10)    : w = 5.0  (capped)
        - Tail image       (f_c=2)     : w = 5.0  (capped)

    This boosts upper-mid class images by 1.8-3.2x, making the model see
    them more often per epoch without changing the loss function.

    Args:
        metadata:          DataFrame with 'cui_list' column (list of CUI strings).
        train_frequencies: Dict mapping CUI string → training-set frequency.
        f_ref:             Reference frequency for rarity computation (default 1000).
        gamma:             Power for rarity softening (default 0.5).
        w_max:             Maximum per-image weight (default 5.0).

    Returns:
        np.ndarray of shape (N,) with per-image sampling weights ≥ 1.0.
    """
    weights = np.ones(len(metadata), dtype=np.float32)
    for i, cuis in enumerate(metadata["cui_list"]):
        if not cuis:  # no labels: use weight 1
            continue
        max_rarity = 0.0
        for cui in cuis:
            f_c = float(train_frequencies.get(cui, 1))
            r = (f_ref / max(f_c, 1.0)) ** gamma
            if r > max_rarity:
                max_rarity = r
        weights[i] = float(np.clip(max_rarity, 1.0, w_max))
    return weights



def resolve_num_workers(
    base_workers: int,
    preferred_workers: int,
    override: int | None,
    colab_safe_workers: int = 2,
) -> int:
    if override is not None:
        return max(0, int(override))

    in_colab = bool(os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU"))
    if in_colab:
        return max(0, min(int(base_workers), int(colab_safe_workers)))

    cpu_count = os.cpu_count() or base_workers
    if cpu_count >= max(preferred_workers * 2, 8):
        return max(0, int(preferred_workers))
    return max(0, int(base_workers))


def build_data_bundle(
    *,
    metadata: Dict[str, pd.DataFrame],
    num_classes: int,
    train_transform,
    eval_transform,
    batch_size: int,
    eval_batch_size: int,
    pin_memory: bool,
    num_workers: int,
    seed: int,
    label_to_index: Dict[str, int],
    index_to_label: List[str],
    sample_weights: Optional[np.ndarray] = None,
) -> DataBundle:
    train_dataset = RocoMultiLabelDataset(metadata=metadata["train"], num_classes=num_classes, transform=train_transform)
    valid_dataset = RocoMultiLabelDataset(metadata=metadata["valid"], num_classes=num_classes, transform=eval_transform)
    test_dataset = RocoMultiLabelDataset(metadata=metadata["test"], num_classes=num_classes, transform=eval_transform)

    generator = torch.Generator()
    generator.manual_seed(seed)

    common_loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }

    if sample_weights is not None:
        # Class-balanced oversampling: replace shuffle with WeightedRandomSampler.
        # num_samples = len(train_dataset) keeps epoch length the same as A0.
        # replacement=True is required for weighted sampling.
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).float(),
            num_samples=len(train_dataset),
            replacement=True,
            generator=generator,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            **common_loader_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            **common_loader_kwargs,
        )


    valid_loader = DataLoader(
        valid_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        **common_loader_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        **common_loader_kwargs,
    )

    return DataBundle(
        metadata=metadata,
        datasets={"train": train_dataset, "valid": valid_dataset, "test": test_dataset},
        loaders={"train": train_loader, "valid": valid_loader, "test": test_loader},
        label_to_index=label_to_index,
        index_to_label=index_to_label,
        num_workers=num_workers,
    )
