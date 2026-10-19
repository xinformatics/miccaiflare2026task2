from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_ce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = target >= 0
    ce = F.cross_entropy(logits, target, ignore_index=-1) if valid.any() else logits.sum() * 0.0
    probabilities = logits.softmax(dim=1)
    safe_target = target.masked_fill(~valid, 0)
    one_hot = F.one_hot(safe_target, logits.shape[1]).movedim(-1, 1).to(probabilities.dtype)
    valid = valid[:, None]
    probabilities = probabilities * valid
    one_hot = one_hot * valid
    probabilities = probabilities[:, 1:]
    one_hot = one_hot[:, 1:]
    reduce_dims = tuple(range(2, probabilities.ndim))
    intersection = (probabilities * one_hot).sum(dim=reduce_dims)
    denominator = probabilities.sum(dim=reduce_dims) + one_hot.sum(dim=reduce_dims)
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    return ce + (1.0 - dice.mean())


def joint_loss(
    outputs: dict[str, object],
    segmentation: torch.Tensor,
    classifications: dict[str, torch.Tensor],
    class_weights: dict[str, torch.Tensor] | None = None,
    glance_segmentation: torch.Tensor | None = None,
    segmentation_weight: float = 1.0,
    coarse_weight: float = 0.25,
    classification_weight: float = 0.75,
    classification_head_weights: dict[str, float] | None = None,
    classification_label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    segmentation_logits = outputs["segmentation"]
    coarse_logits = outputs["coarse_segmentation"]
    classification_logits = outputs["classifications"]
    assert isinstance(segmentation_logits, torch.Tensor)
    assert isinstance(coarse_logits, torch.Tensor)
    assert isinstance(classification_logits, dict)

    segmentation_loss = dice_ce_loss(segmentation_logits, segmentation)
    coarse_source = segmentation if glance_segmentation is None else glance_segmentation
    coarse_target = torch.where(coarse_source < 0, -1, (coarse_source > 0).long()).float()[:, None]
    coarse_target = F.interpolate(coarse_target, size=coarse_logits.shape[-3:], mode="nearest")[:, 0].long()
    coarse_loss = dice_ce_loss(coarse_logits, coarse_target)
    classification_loss = segmentation_loss.new_zeros(())
    classification_weight_sum = 0.0
    for head, logits in classification_logits.items():
        head_weight = 1.0 if classification_head_weights is None else classification_head_weights.get(head, 1.0)
        if head_weight <= 0:
            continue
        weight = None if class_weights is None else class_weights.get(head)
        classification_loss = classification_loss + head_weight * F.cross_entropy(
            logits,
            classifications[head],
            weight=weight,
            label_smoothing=classification_label_smoothing,
        )
        classification_weight_sum += head_weight
    classification_loss = classification_loss / max(1.0, classification_weight_sum)
    total = (
        segmentation_weight * segmentation_loss
        + coarse_weight * coarse_loss
        + classification_weight * classification_loss
    )
    return total, {
        "loss": float(total.detach()),
        "seg": float(segmentation_loss.detach()),
        "coarse": float(coarse_loss.detach()),
        "cls": float(classification_loss.detach()),
    }
