"""Loss functions used by the active A/B/C experiment set.

Supported losses:
  - ``bce``: plain BCE baseline
  - ``asl``: Asymmetric Loss
  - ``head_bce_asl``: BCE on head classes, ASL on non-head classes
  - ``bce_asl_hybrid``: concurrent weighted sum of BCE and ASL
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _asymmetric_loss_elementwise(
    *,
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma_neg: float | torch.Tensor,
    gamma_pos: float,
    clip: float,
    eps: float,
) -> torch.Tensor:
    logits = logits.float()
    targets = targets.float()
    probs = torch.sigmoid(logits)

    xs_pos = probs
    xs_neg = 1.0 - probs
    if clip > 0:
        xs_neg = (xs_neg + clip).clamp(max=1.0)

    loss_pos = targets * torch.log(xs_pos.clamp(min=eps))
    loss_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=eps))
    loss = loss_pos + loss_neg

    if isinstance(gamma_neg, torch.Tensor):
        gamma_neg_tensor = gamma_neg.to(logits.device)
        if gamma_neg_tensor.ndim == 1:
            gamma_neg_tensor = gamma_neg_tensor.unsqueeze(0)
        any_focus = True
    else:
        gamma_neg_tensor = float(gamma_neg)
        any_focus = gamma_pos > 0 or gamma_neg_tensor > 0

    if any_focus:
        pt = xs_pos * targets + xs_neg * (1.0 - targets)
        one_sided_gamma = gamma_pos * targets + gamma_neg_tensor * (1.0 - targets)
        one_sided_w = torch.pow(1.0 - pt, one_sided_gamma).detach()
        loss = loss * one_sided_w

    return -loss


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification."""

    def __init__(
        self,
        gamma_neg: float | Sequence[float] = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.gamma_pos = float(gamma_pos)
        self.clip = float(clip)
        self.eps = float(eps)

        if isinstance(gamma_neg, (int, float)):
            self.gamma_neg_is_vector = False
            self.gamma_neg_scalar = float(gamma_neg)
        else:
            self.gamma_neg_is_vector = True
            self.gamma_neg_scalar = 0.0
            self.register_buffer("gamma_neg_vec", torch.tensor(np.asarray(gamma_neg, dtype=np.float32)))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return _asymmetric_loss_elementwise(
            logits=logits,
            targets=targets,
            gamma_neg=self.gamma_neg_vec if self.gamma_neg_is_vector else self.gamma_neg_scalar,
            gamma_pos=self.gamma_pos,
            clip=self.clip,
            eps=self.eps,
        ).mean()


class BCEASLHybridLoss(nn.Module):
    """Concurrent weighted sum of ASL and BCE."""

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-5,
        lambda_asl: float = 0.95,
    ):
        super().__init__()
        if not 0.0 <= lambda_asl <= 1.0:
            raise ValueError(f"lambda_asl must be in [0, 1], got {lambda_asl}")
        self.lambda_asl = float(lambda_asl)
        self.asl = AsymmetricLoss(
            gamma_neg=gamma_neg,
            gamma_pos=gamma_pos,
            clip=clip,
            eps=eps,
        )
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_asl = self.asl(logits, targets)
        loss_bce = self.bce(logits, targets)
        return self.lambda_asl * loss_asl + (1.0 - self.lambda_asl) * loss_bce


class HeadBCEAsymmetricLoss(nn.Module):
    """BCE on head classes, ASL on non-head classes."""

    def __init__(
        self,
        class_frequencies: np.ndarray,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-5,
        head_threshold: int = 1000,
    ):
        super().__init__()
        frequencies = np.asarray(class_frequencies, dtype=np.int64)
        head_mask = frequencies > int(head_threshold)
        if not np.any(head_mask):
            raise ValueError("HeadBCEAsymmetricLoss found no head classes with the current threshold")
        if np.all(head_mask):
            raise ValueError("HeadBCEAsymmetricLoss requires at least one non-head class")

        self.gamma_neg = float(gamma_neg)
        self.gamma_pos = float(gamma_pos)
        self.clip = float(clip)
        self.eps = float(eps)
        self.register_buffer("head_mask", torch.tensor(head_mask, dtype=torch.bool))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        targets = targets.float()

        elementwise_bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        elementwise_asl = _asymmetric_loss_elementwise(
            logits=logits,
            targets=targets,
            gamma_neg=self.gamma_neg,
            gamma_pos=self.gamma_pos,
            clip=self.clip,
            eps=self.eps,
        )

        head_mask = self.head_mask.to(logits.device).unsqueeze(0)
        routed_loss = torch.where(head_mask, elementwise_bce, elementwise_asl)
        return routed_loss.mean()


def build_criterion(
    config: Dict[str, Any],
    class_frequencies: Optional[np.ndarray] = None,
) -> nn.Module:
    """Build the loss function used by the active experiment set."""

    loss_type = config.get("loss_type", "bce")
    loss_kwargs = config.get("loss_kwargs", {})

    if loss_type == "bce":
        return nn.BCEWithLogitsLoss()

    if loss_type == "asl":
        return AsymmetricLoss(
            gamma_neg=float(loss_kwargs.get("gamma_neg", 4.0)),
            gamma_pos=float(loss_kwargs.get("gamma_pos", 0.0)),
            clip=float(loss_kwargs.get("clip", 0.05)),
            eps=float(loss_kwargs.get("eps", 1e-5)),
        )

    if loss_type == "head_bce_asl":
        if class_frequencies is None:
            raise ValueError("class_frequencies is required for loss_type='head_bce_asl'")
        return HeadBCEAsymmetricLoss(
            class_frequencies=class_frequencies,
            gamma_neg=float(loss_kwargs.get("gamma_neg", 4.0)),
            gamma_pos=float(loss_kwargs.get("gamma_pos", 0.0)),
            clip=float(loss_kwargs.get("clip", 0.05)),
            eps=float(loss_kwargs.get("eps", 1e-5)),
            head_threshold=int(loss_kwargs.get("head_threshold", 1000)),
        )

    if loss_type == "bce_asl_hybrid":
        return BCEASLHybridLoss(
            gamma_neg=float(loss_kwargs.get("gamma_neg", 4.0)),
            gamma_pos=float(loss_kwargs.get("gamma_pos", 0.0)),
            clip=float(loss_kwargs.get("clip", 0.05)),
            eps=float(loss_kwargs.get("eps", 1e-5)),
            lambda_asl=float(loss_kwargs.get("lambda_asl", 0.95)),
        )

    raise ValueError(
        f"Unknown loss_type='{loss_type}'. "
        "Supported: 'bce', 'asl', 'head_bce_asl', 'bce_asl_hybrid'"
    )
