from __future__ import annotations

import itertools
import math
import os
from pathlib import Path

import blosc2
import numpy as np
import torch

from unified.data import PreparedTask, _resize_and_pad
from unified.model import GlanceFocusAutoMSC
from unified.specs import TASK_SPECS, TaskSpec


def _autocast_dtype(device: torch.device) -> torch.dtype | None:
    """Use an inference dtype supported by the actual deployment GPU.

    The challenge machine uses a Turing Quadro RTX5000 (SM 7.5), which does
    not support native bfloat16. Ampere and newer use bfloat16; older CUDA
    devices use float16. CPU inference stays in float32.
    """
    if device.type != "cuda":
        return None
    override = os.environ.get("AUTOMSC_AMP_DTYPE", "auto").lower()
    if override != "auto":
        choices = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": None}
        if override not in choices:
            raise ValueError(f"Unknown AUTOMSC_AMP_DTYPE={override!r}; choose auto, float16, bfloat16, or float32")
        return choices[override]
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def _autocast(device: torch.device):
    dtype = _autocast_dtype(device)
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None)


def _starts(size: int, patch: int, overlap: float) -> list[int]:
    if size <= patch:
        return [0]
    stride = max(1, round(patch * (1.0 - overlap)))
    values = list(range(0, size - patch + 1, stride))
    if values[-1] != size - patch:
        values.append(size - patch)
    return values


def _gaussian(patch_size: tuple[int, int, int]) -> np.ndarray:
    axes = [np.linspace(-1.0, 1.0, size, dtype=np.float32) for size in patch_size]
    zz, yy, xx = np.meshgrid(*axes, indexing="ij")
    return np.exp(-0.5 * (zz * zz + yy * yy + xx * xx) / (0.5**2)).astype(np.float32)


def _aggregate_classification_logits(
    values: list[np.ndarray], foreground_scores: list[float], top_fraction: float
) -> np.ndarray:
    """Average the most foreground-rich windows, matching lesion-focused training crops."""
    if not 0 < top_fraction <= 1:
        raise ValueError(f"classification_top_fraction must be in (0, 1], got {top_fraction}")
    count = max(1, math.ceil(len(values) * top_fraction))
    selected = np.argsort(np.asarray(foreground_scores))[-count:]
    return np.stack([values[index] for index in selected]).mean(0)


def load_model(checkpoint_path: Path, base_width: int, device: torch.device) -> GlanceFocusAutoMSC:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    serialized = checkpoint.get("task_specs")
    task_specs = (
        {
            key: TaskSpec(
                value["key"],
                int(value["dataset_id"]),
                value["folder_name"],
                int(value["segmentation_classes"]),
                {name: int(count) for name, count in value["classification_heads"].items()},
            )
            for key, value in serialized.items()
        }
        if serialized
        else TASK_SPECS
    )
    embedding_rows = int(checkpoint["model"]["task_embedding.weight"].shape[0])
    model = GlanceFocusAutoMSC(
        base_width, task_specs=task_specs, task_embedding_count=embedding_rows
    )
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


@torch.inference_mode()
def predict_preprocessed(
    model: GlanceFocusAutoMSC,
    task: PreparedTask,
    identifier: str,
    device: torch.device,
    overlap: float = 0.5,
    classification_top_fraction: float = 1.0,
    foreground_threshold: float = 0.5,
    glance_size: int = 64,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    source = blosc2.open(urlpath=str(task.data_dir / f"{identifier}.b2nd"), mode="r", dparams={"nthreads": 2})
    image = np.asarray(source[:], dtype=np.float32)
    original_shape = image.shape[-3:]
    patch_size = task.patch_size
    pad_before = tuple(max(0, (patch - size) // 2) for size, patch in zip(original_shape, patch_size))
    pad_after = tuple(max(0, patch - size - before) for size, patch, before in zip(original_shape, patch_size, pad_before))
    padding = ((0, 0),) + tuple((before, after) for before, after in zip(pad_before, pad_after))
    padded = np.pad(image, padding)

    glance = _resize_and_pad(image, glance_size, "trilinear")
    glance_tensor = torch.from_numpy(glance[None]).to(device)
    modality_ids = torch.tensor(task.modality_ids, dtype=torch.long, device=device)[None]
    spatial = padded.shape[-3:]
    starts = [_starts(size, patch, overlap) for size, patch in zip(spatial, patch_size)]
    weight = _gaussian(patch_size)
    scores = np.zeros((task.spec.segmentation_classes, *spatial), dtype=np.float32)
    denominator = np.zeros(spatial, dtype=np.float32)
    classification_logits: dict[str, list[np.ndarray]] = {
        head: [] for head in task.spec.classification_heads
    }
    foreground_scores: list[float] = []

    for start in itertools.product(*starts):
        slices = tuple(slice(value, value + patch) for value, patch in zip(start, patch_size))
        focus = torch.from_numpy(np.ascontiguousarray(padded[(slice(None), *slices)])[None]).to(device)
        with _autocast(device):
            clinical = torch.from_numpy(task.clinical[identifier])[None].to(device)
            output = model(
                focus,
                modality_ids,
                task.spec.key,
                glance_images=glance_tensor,
                clinical_features=clinical,
            )
        probabilities = output["segmentation"].softmax(1)[0].float().cpu().numpy()
        foreground_scores.append(float(probabilities[1:].sum(0).mean()))
        scores[(slice(None), *slices)] += probabilities * weight[None]
        denominator[slices] += weight
        for head, logits in output["classifications"].items():
            classification_logits[head].append(logits[0].float().cpu().numpy())

    scores /= np.maximum(denominator[None], 1e-7)
    unpad = tuple(slice(before, before + size) for before, size in zip(pad_before, original_shape))
    unpadded_scores = scores[(slice(None), *unpad)]
    if task.spec.segmentation_classes == 2:
        if not 0.0 < foreground_threshold < 1.0:
            raise ValueError("foreground_threshold must be in (0, 1)")
        segmentation = (unpadded_scores[1] >= foreground_threshold).astype(np.uint8)
    else:
        segmentation = unpadded_scores.argmax(0).astype(np.uint8)
    classifications = {}
    for head, values in classification_logits.items():
        logits = _aggregate_classification_logits(values, foreground_scores, classification_top_fraction)
        logits -= logits.max()
        probabilities = np.exp(logits)
        classifications[head] = probabilities / probabilities.sum()
    return segmentation, classifications


@torch.inference_mode()
def predict_classification_array(
    model: GlanceFocusAutoMSC,
    task: PreparedTask,
    image: np.ndarray,
    device: torch.device,
    overlap: float = 0.5,
    clinical: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Classify one normalized, resampled channel-first image array."""
    image = np.asarray(image, dtype=np.float32)
    original_shape = image.shape[-3:]
    patch_size = task.patch_size
    pad_before = tuple(max(0, (patch - size) // 2) for size, patch in zip(original_shape, patch_size))
    pad_after = tuple(
        max(0, patch - size - before)
        for size, patch, before in zip(original_shape, patch_size, pad_before)
    )
    padding = ((0, 0),) + tuple((before, after) for before, after in zip(pad_before, pad_after))
    padded = np.pad(image, padding)

    glance = _resize_and_pad(image, 64, "trilinear")
    glance_tensor = torch.from_numpy(glance[None]).to(device)
    modality_ids = torch.tensor(task.modality_ids, dtype=torch.long, device=device)[None]
    starts = [_starts(size, patch, overlap) for size, patch in zip(padded.shape[-3:], patch_size)]
    classification_logits: dict[str, list[np.ndarray]] = {
        head: [] for head in task.spec.classification_heads
    }
    clinical_tensor = None if clinical is None else torch.from_numpy(np.asarray(clinical, dtype=np.float32))[None].to(device)
    for start in itertools.product(*starts):
        slices = tuple(slice(value, value + patch) for value, patch in zip(start, patch_size))
        focus = torch.from_numpy(np.ascontiguousarray(padded[(slice(None), *slices)])[None]).to(device)
        with _autocast(device):
            output = model(
                focus,
                modality_ids,
                task.spec.key,
                glance_images=glance_tensor,
                clinical_features=clinical_tensor,
            )
        for head, logits in output["classifications"].items():
            classification_logits[head].append(logits[0].float().cpu().numpy())

    classifications = {}
    for head, values in classification_logits.items():
        logits = np.stack(values).mean(0)
        logits -= logits.max()
        probabilities = np.exp(logits)
        classifications[head] = probabilities / probabilities.sum()
    return classifications


@torch.inference_mode()
def predict_classification_preprocessed(
    model: GlanceFocusAutoMSC,
    task: PreparedTask,
    identifier: str,
    device: torch.device,
    overlap: float = 0.5,
) -> dict[str, np.ndarray]:
    """Classify a case from the unified preprocessed data store."""
    source = blosc2.open(urlpath=str(task.data_dir / f"{identifier}.b2nd"), mode="r", dparams={"nthreads": 2})
    image = np.asarray(source[:], dtype=np.float32)
    return predict_classification_array(
        model,
        task,
        image,
        device,
        overlap,
        clinical=task.clinical[identifier],
    )
