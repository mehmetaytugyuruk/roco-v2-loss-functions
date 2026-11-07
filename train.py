from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple
import zipfile

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import (  # noqa: E402
    apply_overrides,
    load_config,
    resolve_output_dir,
    resolve_paths,
)
from data import (  # noqa: E402
    build_class_vocabulary,
    build_data_bundle,
    build_metadata_tables,
    compute_sample_weights,
    resolve_num_workers,
    validate_split_integrity,
)
from io_utils import (  # noqa: E402
    CSVMetricLogger,
    save_checkpoint,
    write_probabilities_csv,
)
from model import build_model  # noqa: E402
from loss import build_criterion  # noqa: E402
from scheduler import WarmupCosineConfig, WarmupCosineScheduler  # noqa: E402
from transforms import build_base_transform, build_train_transform  # noqa: E402
from utils import resolve_device, set_seed, summarize_counts  # noqa: E402


TRAIN_LOG_COLUMNS = ["epoch", "train_loss", "val_loss", "best_val_loss", "learning_rate"]


def build_class_frequencies(
    train_concepts_csv: Path,
    index_to_label: List[str],
) -> np.ndarray:
    """Compute per-class training frequencies aligned to index_to_label order.

    The returned array has shape (C,) where arr[i] = number of training images
    that contain class index_to_label[i].  The order is IDENTICAL to the order
    used for target vector encoding (see RocoMultiLabelDataset), so passing this
    array to class-frequency-aware losses is safe.

    Alignment guarantee:
        index_to_label  →  built by build_class_vocabulary via sorted(all_cuis)
        class_frequencies →  built here via [freq_dict.get(cui) for cui in index_to_label]
        target[i]       →  set to 1 via label_to_index[cui] which equals
                            index_to_label.index(cui) = i
    All three use the same sorted CUI list as their index key.
    """
    from data import parse_cuis  # local import to avoid circular dependency
    import pandas as pd

    train_df = pd.read_csv(train_concepts_csv, dtype={"ID": str, "CUIs": str})
    freq_dict: Dict[str, int] = {}
    for value in train_df["CUIs"].fillna(""):
        for cui in parse_cuis(value):
            freq_dict[cui] = freq_dict.get(cui, 0) + 1

    # Key step: iterate index_to_label (the canonical sorted order) —
    # NOT freq_dict.keys() — so the result is aligned to model outputs.
    frequencies = np.array(
        [freq_dict.get(cui, 0) for cui in index_to_label],
        dtype=np.float32,
    )

    # Hard assertion: shape must match num_classes exactly.
    assert len(frequencies) == len(index_to_label), (
        f"Frequency array length {len(frequencies)} != "
        f"num_classes {len(index_to_label)}. Alignment broken!"
    )
    return frequencies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROCOv2 EfficientNet-B0 multi-label trainer")
    parser.add_argument("--config", type=str, default="configs/A/A1.yaml", help="Path to config.yaml")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader workers")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and only run evaluation/prediction")
    return parser.parse_args()


def _autocast_context(use_amp: bool):
    if use_amp:
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
    return contextlib.nullcontext()


def maybe_extract_data_zip(config: Dict[str, object], annotations_dir: Path, images_dir: Path, data_zip_path: Path | None) -> None:
    if annotations_dir.exists() and images_dir.exists():
        return
    if not bool(config.get("auto_extract_data_zip", False)):
        return
    if data_zip_path is None or not data_zip_path.exists():
        return

    extract_root = annotations_dir.parents[1]
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(data_zip_path, "r") as zip_handle:
        zip_handle.extractall(extract_root)


def _create_data_bundle(
    *,
    config: Dict[str, object],
    metadata,
    label_to_index,
    index_to_label,
    train_transform,
    eval_transform,
    num_workers: int,
    pin_memory: bool,
    sample_weights=None,
):
    return build_data_bundle(
        metadata=metadata,
        num_classes=int(config["num_classes"]),
        train_transform=train_transform,
        eval_transform=eval_transform,
        batch_size=int(config["batch_size"]),
        eval_batch_size=int(config.get("eval_batch_size", config["batch_size"])),
        pin_memory=pin_memory,
        num_workers=num_workers,
        seed=int(config["seed"]),
        label_to_index=label_to_index,
        index_to_label=index_to_label,
        sample_weights=sample_weights,
    )


def train_one_epoch(
    *,
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    grad_clip: float,
    use_amp: bool,
    show_progress: bool,
    epoch: int,
    total_epochs: int,
) -> float:
    model.train()
    running_loss = 0.0
    sample_count = 0

    iterator = loader
    if show_progress:
        iterator = tqdm(
            loader,
            total=len(loader),
            desc=f"Epoch {epoch:02d}/{total_epochs:02d} [train]",
            leave=False,
            dynamic_ncols=True,
        )

    for batch in iterator:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with _autocast_context(use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        batch_size = images.size(0)
        running_loss += float(loss.detach().item()) * batch_size
        sample_count += batch_size

        if show_progress:
            avg_loss = running_loss / max(sample_count, 1)
            iterator.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{scheduler.get_last_lr():.2e}")

    return running_loss / max(sample_count, 1)


def evaluate_loss(
    *,
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    show_progress: bool,
    epoch: int,
    total_epochs: int,
) -> float:
    model.eval()
    running_loss = 0.0
    sample_count = 0

    iterator = loader
    if show_progress:
        iterator = tqdm(
            loader,
            total=len(loader),
            desc=f"Epoch {epoch:02d}/{total_epochs:02d} [valid]",
            leave=False,
            dynamic_ncols=True,
        )

    with torch.no_grad():
        for batch in iterator:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            with _autocast_context(use_amp):
                logits = model(images)
                loss = criterion(logits, targets)

            batch_size = images.size(0)
            running_loss += float(loss.detach().item()) * batch_size
            sample_count += batch_size

            if show_progress:
                avg_loss = running_loss / max(sample_count, 1)
                iterator.set_postfix(loss=f"{avg_loss:.4f}")

    return running_loss / max(sample_count, 1)


def predict_probabilities(
    *,
    model: nn.Module,
    loader,
    device: torch.device,
    use_amp: bool,
    show_progress: bool,
    split_name: str,
) -> Tuple[List[str], np.ndarray]:
    model.eval()
    ids: List[str] = []
    probabilities: List[np.ndarray] = []

    iterator = loader
    if show_progress:
        iterator = tqdm(
            loader,
            total=len(loader),
            desc=f"[{split_name}] probabilities",
            leave=False,
            dynamic_ncols=True,
        )

    with torch.no_grad():
        for batch in iterator:
            images = batch["image"].to(device, non_blocking=True)
            with _autocast_context(use_amp):
                logits = model(images)
            probs = torch.sigmoid(logits).detach().float().cpu().numpy()
            probabilities.append(probs)
            ids.extend(batch["id"])

    probabilities_np = np.concatenate(probabilities, axis=0)
    return ids, probabilities_np


def build_checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    config: Dict[str, object],
    index_to_label: Sequence[str],
) -> Dict[str, object]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": float(val_loss),
        "index_to_label": list(index_to_label),
        "config": config,
    }


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()

    config = load_config(config_path)
    config = apply_overrides(
        config,
        output_dir=args.output_dir,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    set_seed(int(config["seed"]))

    output_dir = resolve_output_dir(config, PROJECT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = resolve_paths(PROJECT_ROOT, config["paths"])
    annotations_dir = paths["annotations_dir"]
    images_dir = paths["images_dir"]
    data_zip = paths.get("data_zip") if str(config["paths"].get("data_zip", "")).strip() else None

    maybe_extract_data_zip(config, annotations_dir, images_dir, data_zip)

    split_counts = validate_split_integrity(annotations_dir=annotations_dir, images_root=images_dir)
    print(f"Data integrity check passed: {summarize_counts(split_counts)}")

    label_to_index, index_to_label = build_class_vocabulary(
        train_concepts_csv=annotations_dir / "train_concepts.csv",
        expected_num_classes=int(config["num_classes"]),
    )

    metadata = build_metadata_tables(
        annotations_dir=annotations_dir,
        images_root=images_dir,
        label_to_index=label_to_index,
        max_train_samples=config.get("max_train_samples"),
        max_valid_samples=config.get("max_valid_samples"),
        max_test_samples=config.get("max_test_samples"),
    )

    device = resolve_device(config.get("device", "auto"))
    use_amp = bool(config["amp"]) and device.type == "cuda"
    effective_pin_memory = bool(config["pin_memory"]) and device.type == "cuda"

    num_workers = resolve_num_workers(
        base_workers=int(config["num_workers"]),
        preferred_workers=int(config["num_workers_if_supported"]),
        override=args.num_workers,
        colab_safe_workers=int(config.get("colab_safe_num_workers", 2)),
    )

    train_transform = build_train_transform(
        image_size=int(config["image_size"]),
        color_jitter_probability=float(config["color_jitter_probability"]),
    )
    eval_transform = build_base_transform(image_size=int(config["image_size"]))

    # ---------------------------------------------------------------------------
    # Class-balanced oversampling (2A, 2C)
    # ---------------------------------------------------------------------------
    sampling_strategy = str(config.get("sampling_strategy", "random")).lower()
    sample_weights = None
    if sampling_strategy == "class_balanced":
        sampling_kwargs = config.get("sampling_kwargs", {})
        # Build a plain dict freq for compute_sample_weights
        from data import parse_cuis
        import pandas as _pd
        _train_df = _pd.read_csv(annotations_dir / "train_concepts.csv", dtype={"ID": str, "CUIs": str})
        _train_freq: Dict[str, int] = {}
        for _v in _train_df["CUIs"].fillna(""):
            for _c in parse_cuis(_v):
                _train_freq[_c] = _train_freq.get(_c, 0) + 1

        sample_weights = compute_sample_weights(
            metadata=metadata["train"],
            train_frequencies=_train_freq,
            f_ref=float(sampling_kwargs.get("f_ref", 1000.0)),
            gamma=float(sampling_kwargs.get("gamma", 0.5)),
            w_max=float(sampling_kwargs.get("w_max", 5.0)),
        )
        w_mean = float(sample_weights.mean())
        w_max_actual = float(sample_weights.max())
        n_boosted = int((sample_weights > 1.01).sum())
        print(
            f"Class-balanced oversampling: mean_weight={w_mean:.3f}, "
            f"max_weight={w_max_actual:.3f}, n_boosted={n_boosted}/{len(sample_weights)}"
        )
    else:
        print("Sampling strategy: random (standard shuffle)")

    data_bundle = _create_data_bundle(
        config=config,
        metadata=metadata,
        label_to_index=label_to_index,
        index_to_label=index_to_label,
        train_transform=train_transform,
        eval_transform=eval_transform,
        num_workers=num_workers,
        pin_memory=effective_pin_memory,
        sample_weights=sample_weights,
    )

    try:
        _ = next(iter(data_bundle.loaders["train"]))
    except Exception as exc:
        if data_bundle.num_workers > 0:
            print(
                f"DataLoader worker startup failed with num_workers={data_bundle.num_workers}. "
                "Falling back to num_workers=0."
            )
            data_bundle = _create_data_bundle(
                config=config,
                metadata=metadata,
                label_to_index=label_to_index,
                index_to_label=index_to_label,
                train_transform=train_transform,
                eval_transform=eval_transform,
                num_workers=0,
                pin_memory=effective_pin_memory,
                sample_weights=sample_weights,
            )
        else:
            raise RuntimeError("DataLoader startup failed with num_workers=0") from exc

    print(f"Using device={device}, amp={use_amp}, workers={data_bundle.num_workers}, pin_memory={effective_pin_memory}")

    model = build_model(
        num_classes=int(config["num_classes"]),
        pretrained=bool(config["pretrained"]),
        model_name=str(config.get("model_name", "efficientnet_b0")),
        image_size=int(config["image_size"]),
    )
    model = model.to(device)
    if bool(config.get("torch_compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)
        print("torch.compile() enabled")

    # Build class frequencies only for the active experiment families that need them.
    loss_type = config.get("loss_type", "bce")
    class_frequencies: Optional[np.ndarray] = None
    if loss_type == "head_bce_asl":
        class_frequencies = build_class_frequencies(
            train_concepts_csv=annotations_dir / "train_concepts.csv",
            index_to_label=index_to_label,
        )
        print(
            f"class_frequencies: shape={class_frequencies.shape}, "
            f"min={class_frequencies.min():.0f}, "
            f"max={class_frequencies.max():.0f}, "
            f"median={float(np.median(class_frequencies)):.0f}"
        )

    criterion = build_criterion(config, class_frequencies=class_frequencies)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )

    steps_per_epoch = len(data_bundle.loaders["train"])
    total_steps = int(config["epochs"]) * steps_per_epoch
    warmup_steps = int(config["warmup_epochs"]) * steps_per_epoch

    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        cfg=WarmupCosineConfig(
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr=float(config["min_lr"]),
            base_lr=float(config["lr"]),
        ),
    )

    # BF16 autocast doesn't need gradient scaling (no overflow risk unlike FP16)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    outputs = config["outputs"]
    best_ckpt_path = output_dir / outputs["best_model"]
    last_ckpt_path = output_dir / outputs["last_model"]
    train_log_path = output_dir / outputs["train_log"]
    val_probabilities_path = output_dir / outputs["val_probabilities"]
    test_probabilities_path = output_dir / outputs["test_probabilities"]

    logger = CSVMetricLogger(path=train_log_path, fieldnames=TRAIN_LOG_COLUMNS)
    show_progress = bool(config.get("progress_bar", True))

    best_val_loss = float("inf")

    if not args.eval_only:
        for epoch in range(1, int(config["epochs"]) + 1):
            if hasattr(criterion, "set_epoch"):
                criterion.set_epoch(epoch)
            train_loss = train_one_epoch(
                model=model,
                loader=data_bundle.loaders["train"],
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                grad_clip=float(config["grad_clip"]),
                use_amp=use_amp,
                show_progress=show_progress,
                epoch=epoch,
                total_epochs=int(config["epochs"]),
            )

            val_loss = evaluate_loss(
                model=model,
                loader=data_bundle.loaders["valid"],
                criterion=criterion,
                device=device,
                use_amp=use_amp,
                show_progress=show_progress,
                epoch=epoch,
                total_epochs=int(config["epochs"]),
            )

            if not np.isfinite(val_loss):
                raise ValueError(f"Validation loss is not finite at epoch {epoch}: {val_loss}")

            is_new_best = val_loss < best_val_loss
            if is_new_best:
                best_val_loss = val_loss

            logger.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "learning_rate": scheduler.get_last_lr(),
                }
            )

            print(
                f"Epoch {epoch:02d}/{int(config['epochs'])} "
                f"train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
                f"best_val_loss={best_val_loss:.5f}"
            )

            checkpoint_payload = build_checkpoint_payload(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                config=config,
                index_to_label=index_to_label,
            )
            save_checkpoint(last_ckpt_path, checkpoint_payload)
            if is_new_best:
                save_checkpoint(best_ckpt_path, checkpoint_payload)

    if not best_ckpt_path.exists():
        raise RuntimeError("Best checkpoint was not created")

    best_checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    val_ids, val_probabilities = predict_probabilities(
        model=model,
        loader=data_bundle.loaders["valid"],
        device=device,
        use_amp=use_amp,
        show_progress=show_progress,
        split_name="valid",
    )
    test_ids, test_probabilities = predict_probabilities(
        model=model,
        loader=data_bundle.loaders["test"],
        device=device,
        use_amp=use_amp,
        show_progress=show_progress,
        split_name="test",
    )

    if val_probabilities.shape[1] != int(config["num_classes"]):
        raise ValueError("val_probabilities class dimension mismatch")
    if test_probabilities.shape[1] != int(config["num_classes"]):
        raise ValueError("test_probabilities class dimension mismatch")

    write_probabilities_csv(val_probabilities_path, val_ids, val_probabilities, index_to_label)
    write_probabilities_csv(test_probabilities_path, test_ids, test_probabilities, index_to_label)
    print(f"Artifacts saved to: {output_dir}")
    print(
        "Threshold optimization intentionally skipped in train.py. "
        "Use offline threshold script with val/test probabilities."
    )


if __name__ == "__main__":
    main()
