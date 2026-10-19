from __future__ import annotations

import csv
import itertools
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from unified.model import GlanceFocusAutoMSC
from unified.specs import canonical_modality_id


CASE_PATTERN = re.compile(r"^(?P<identifier>.+)_(?P<channel>\d{4})\.nii(?:\.gz)?$")


def deployment_autocast(device: torch.device):
    """Select a mixed-precision type supported by the deployment GPU."""
    if device.type != "cuda":
        return torch.autocast(device_type=device.type, enabled=False)
    override = os.environ.get("AUTOMSC_AMP_DTYPE", "auto").lower()
    if override == "float32":
        dtype = None
    elif override == "float16":
        dtype = torch.float16
    elif override == "bfloat16":
        dtype = torch.bfloat16
    elif override == "auto":
        major, _ = torch.cuda.get_device_capability(device)
        dtype = torch.bfloat16 if major >= 8 else torch.float16
    else:
        raise ValueError(f"Unknown AUTOMSC_AMP_DTYPE={override!r}")
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype is not None)


@dataclass(frozen=True)
class DeploymentConfig:
    task_key: str
    modalities: tuple[str, ...]
    classification_columns: dict[str, tuple[str, ...]]
    max_voxels: int = 2_000_000
    crop_margin: int = 8

    @classmethod
    def load(cls, path: Path) -> "DeploymentConfig":
        value = json.loads(path.read_text())
        return cls(
            task_key=value["task_key"],
            modalities=tuple(value["modalities"]),
            classification_columns={key: tuple(columns) for key, columns in value["classification_columns"].items()},
            max_voxels=int(value.get("max_voxels", 2_000_000)),
            crop_margin=int(value.get("crop_margin", 8)),
        )


def group_images(input_dir: Path) -> dict[str, list[Path]]:
    grouped: dict[str, dict[int, Path]] = {}
    for path in sorted(itertools.chain(input_dir.glob("*.nii"), input_dir.glob("*.nii.gz"))):
        match = CASE_PATTERN.match(path.name)
        if match:
            grouped.setdefault(match["identifier"], {})[int(match["channel"])] = path
    return {identifier: [channels[index] for index in sorted(channels)] for identifier, channels in grouped.items()}


def robust_normalize(array: np.ndarray, modality: str) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(array)
    foreground = finite & (array != 0)
    values = array[foreground]
    if values.size < 100:
        values = array[finite]
    if not values.size:
        return np.zeros_like(array)
    if modality.strip().upper() in {"CT", "RTP"}:
        low, high = np.percentile(values, (0.5, 99.5))
        array = np.clip(array, low, high)
        values = array[foreground] if foreground.any() else array[finite]
    center = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - center)))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.std(values))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    normalized = np.clip((array - center) / scale, -8.0, 8.0)
    normalized[~finite] = 0
    return normalized.astype(np.float32, copy=False)


def foreground_crop(image: np.ndarray, margin: int) -> tuple[np.ndarray, tuple[slice, slice, slice]]:
    foreground = np.any(np.isfinite(image) & (image != 0), axis=0)
    coordinates = np.argwhere(foreground)
    shape = np.asarray(image.shape[-3:])
    if not len(coordinates):
        slices = tuple(slice(0, int(size)) for size in shape)
    else:
        lower = np.maximum(coordinates.min(0) - margin, 0)
        upper = np.minimum(coordinates.max(0) + margin + 1, shape)
        slices = tuple(slice(int(start), int(stop)) for start, stop in zip(lower, upper))
    return np.ascontiguousarray(image[(slice(None), *slices)]), slices


def target_shape(shape: tuple[int, int, int], max_voxels: int, divisor: int = 16) -> tuple[int, int, int]:
    scale = min(1.0, (max_voxels / math.prod(shape)) ** (1.0 / 3.0))
    values = [max(divisor, int(round(size * scale / divisor)) * divisor) for size in shape]
    while math.prod(values) > max_voxels:
        index = int(np.argmax(values))
        values[index] = max(divisor, values[index] - divisor)
    return tuple(values)


def resize(array: np.ndarray, shape: tuple[int, int, int], mode: str) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(array))[None].float()
    kwargs = {} if mode == "nearest" else {"align_corners": False}
    return F.interpolate(tensor, size=shape, mode=mode, **kwargs)[0].numpy()


def load_case(paths: list[Path], config: DeploymentConfig) -> tuple[np.ndarray, sitk.Image, tuple[slice, slice, slice]]:
    if len(paths) != len(config.modalities):
        raise ValueError(f"Expected {len(config.modalities)} channels, found {len(paths)}: {paths}")
    reference = sitk.ReadImage(str(paths[0]))
    arrays = []
    for path, modality in zip(paths, config.modalities):
        image = sitk.ReadImage(str(path))
        geometry_matches = (
            image.GetSize() == reference.GetSize()
            and np.allclose(image.GetSpacing(), reference.GetSpacing(), rtol=0.0, atol=1e-5)
            and np.allclose(image.GetOrigin(), reference.GetOrigin(), rtol=0.0, atol=1e-4)
            and np.allclose(image.GetDirection(), reference.GetDirection(), rtol=0.0, atol=1e-6)
        )
        if not geometry_matches:
            raise ValueError(f"Geometry mismatch for {path}")
        arrays.append(robust_normalize(sitk.GetArrayFromImage(image), modality))
    cropped, crop = foreground_crop(np.stack(arrays), config.crop_margin)
    return cropped, reference, crop


def write_segmentation(
    small_segmentation: np.ndarray,
    crop: tuple[slice, slice, slice],
    reference: sitk.Image,
    output_path: Path,
) -> None:
    crop_shape = tuple(part.stop - part.start for part in crop)
    restored_crop = resize(small_segmentation[None].astype(np.float32), crop_shape, "nearest")[0].astype(np.uint8)
    full = np.zeros(tuple(reversed(reference.GetSize())), dtype=np.uint8)
    full[crop] = restored_crop
    output = sitk.GetImageFromArray(full)
    output.CopyInformation(reference)
    sitk.WriteImage(output, str(output_path), useCompression=True)


@torch.inference_mode()
def predict_case(
    model: GlanceFocusAutoMSC,
    image: np.ndarray,
    config: DeploymentConfig,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, np.ndarray], tuple[int, int, int]]:
    shape = target_shape(tuple(image.shape[-3:]), config.max_voxels)
    focus = torch.from_numpy(resize(image, shape, "trilinear")[None]).to(device)
    glance = torch.from_numpy(resize(image, (64, 64, 64), "trilinear")[None]).to(device)
    modality_ids = torch.tensor(
        [[canonical_modality_id(name) for name in config.modalities]], dtype=torch.long, device=device
    )
    with deployment_autocast(device):
        output = model(focus, modality_ids, config.task_key, glance_images=glance)
    segmentation = output["segmentation"].argmax(1)[0].byte().cpu().numpy()
    classifications = {
        name: logits[0].float().softmax(0).cpu().numpy()
        for name, logits in output["classifications"].items()
    }
    return segmentation, classifications, shape


def run(
    input_dir: Path,
    output_dir: Path,
    checkpoint_path: Path,
    config_path: Path,
    base_width: int = 32,
) -> dict[str, object]:
    config = DeploymentConfig.load(config_path)
    grouped = group_images(input_dir)
    if not grouped:
        raise RuntimeError(f"No nnU-Net-formatted NIfTI images found in {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require_cuda = os.environ.get("AUTOMSC_REQUIRE_CUDA", "1").strip().lower() not in {"0", "false", "no"}
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required (set AUTOMSC_REQUIRE_CUDA=0 only for diagnostic CPU runs)")
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    model = GlanceFocusAutoMSC(base_width)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    columns = ["identifier"] + [column for values in config.classification_columns.values() for column in values]
    rows: list[dict[str, object]] = []
    timings: list[dict[str, object]] = []
    started = time.perf_counter()
    for identifier, paths in sorted(grouped.items()):
        case_started = time.perf_counter()
        image, reference, crop = load_case(paths, config)
        segmentation, classifications, inference_shape = predict_case(model, image, config, device)
        write_segmentation(segmentation, crop, reference, output_dir / f"{identifier}.nii.gz")
        row: dict[str, object] = {"identifier": identifier}
        for head, output_columns in config.classification_columns.items():
            probabilities = classifications[head]
            if len(probabilities) != len(output_columns):
                raise ValueError(f"{head}: model returned {len(probabilities)} classes for {len(output_columns)} columns")
            row.update(zip(output_columns, map(float, probabilities)))
        rows.append(row)
        timings.append(
            {
                "identifier": identifier,
                "seconds": time.perf_counter() - case_started,
                "input_shape": list(image.shape[-3:]),
                "inference_shape": list(inference_shape),
            }
        )
    with (output_dir / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "cases": len(rows),
        "total_seconds": time.perf_counter() - started,
        "device": str(device),
        "peak_gpu_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0,
        "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else 0,
        "timings": timings,
    }
    runtime_path = os.environ.get("AUTOMSC_RUNTIME_JSON")
    if runtime_path:
        Path(runtime_path).write_text(json.dumps(result, indent=2) + "\n")
    return result
