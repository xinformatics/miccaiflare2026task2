#!/usr/bin/env python3
"""Generate public-test masks and classification probabilities from one checkpoint."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from unified.data import _read_clinical, _resize_and_pad, discover_tasks
from unified.infer import _aggregate_classification_logits, _autocast, _gaussian, _starts, load_model


CASE_PATTERN = re.compile(r"^(?P<identifier>.+)_(?P<channel>\d{4})\.nii(?:\.gz)?$")


def group_cases(folder: Path) -> dict[str, list[Path]]:
    grouped: dict[str, dict[int, Path]] = {}
    for path in sorted((*folder.glob("*.nii"), *folder.glob("*.nii.gz"))):
        match = CASE_PATTERN.match(path.name)
        if match:
            grouped.setdefault(match["identifier"], {})[int(match["channel"])] = path
    output = {}
    for identifier, channels in grouped.items():
        expected = list(range(len(channels)))
        if sorted(channels) != expected:
            raise ValueError(f"{identifier}: channels={sorted(channels)}, expected={expected}")
        output[identifier] = [channels[index] for index in expected]
    return output


@torch.inference_mode()
def predict_array(
    model,
    task,
    image: np.ndarray,
    clinical: np.ndarray,
    device: torch.device,
    overlap: float,
    classification_top_fraction: float,
    foreground_threshold: float,
    classification_logit_biases: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    original_shape = image.shape[-3:]
    patch_size = task.patch_size
    pad_before = tuple(max(0, (patch - size) // 2) for size, patch in zip(original_shape, patch_size))
    pad_after = tuple(max(0, patch - size - before) for size, patch, before in zip(original_shape, patch_size, pad_before))
    padded = np.pad(image, ((0, 0), *tuple((a, b) for a, b in zip(pad_before, pad_after))))
    glance_size = int(getattr(model, "glance_size", 64))
    glance = torch.from_numpy(_resize_and_pad(image, glance_size, "trilinear")[None]).to(device)
    modality_ids = torch.tensor(task.modality_ids, dtype=torch.long, device=device)[None]
    clinical_tensor = torch.from_numpy(np.asarray(clinical, dtype=np.float32))[None].to(device)
    spatial = padded.shape[-3:]
    starts = [_starts(size, patch, overlap) for size, patch in zip(spatial, patch_size)]
    weight = _gaussian(patch_size)
    scores = np.zeros((task.spec.segmentation_classes, *spatial), dtype=np.float32)
    denominator = np.zeros(spatial, dtype=np.float32)
    classification_logits = {head: [] for head in task.spec.classification_heads}
    foreground_scores: list[float] = []
    for start in itertools.product(*starts):
        slices = tuple(slice(value, value + patch) for value, patch in zip(start, patch_size))
        focus = torch.from_numpy(np.ascontiguousarray(padded[(slice(None), *slices)])[None]).to(device)
        with _autocast(device):
            output = model(
                focus,
                modality_ids,
                task.spec.key,
                glance_images=glance,
                clinical_features=clinical_tensor,
            )
        probabilities = output["segmentation"].softmax(1)[0].float().cpu().numpy()
        foreground_scores.append(float(probabilities[1:].sum(0).mean()))
        scores[(slice(None), *slices)] += probabilities * weight[None]
        denominator[slices] += weight
        for head, logits in output["classifications"].items():
            classification_logits[head].append(logits[0].float().cpu().numpy())
    scores /= np.maximum(denominator[None], 1e-7)
    unpad = tuple(slice(before, before + size) for before, size in zip(pad_before, original_shape))
    logits = np.log(np.maximum(scores[(slice(None), *unpad)], 1e-7))
    if task.spec.segmentation_classes == 2:
        if not 0.0 < foreground_threshold < 1.0:
            raise ValueError("foreground_threshold must lie strictly between zero and one")
        # nnU-Net exports argmax(logits). Shifting the foreground logit makes
        # that argmax equivalent to thresholding its probability at the
        # validation-selected operating point.
        logits[1] += np.log((1.0 - foreground_threshold) / foreground_threshold)
    classifications = {}
    for head, values in classification_logits.items():
        value = _aggregate_classification_logits(values, foreground_scores, classification_top_fraction)
        if head in classification_logit_biases:
            bias = classification_logit_biases[head]
            if bias.shape != value.shape:
                raise ValueError(f"Calibration for {head} has shape {bias.shape}, expected {value.shape}")
            value = value + bias
        value -= value.max()
        probability = np.exp(value)
        classifications[head] = probability / probability.sum()
    return logits, classifications


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="Auto-discovered task key printed by unified.specs")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--classification-top-fraction", type=float, default=1.0)
    parser.add_argument("--foreground-threshold", type=float, default=0.5)
    parser.add_argument(
        "--calibration",
        type=Path,
        help="Optional JSON with a classification_logit_biases mapping learned on the development fold.",
    )
    parser.add_argument(
        "--inference-patch-voxels",
        type=int,
        default=1_000_000,
        help="Global inference patch budget chosen to remain below 12 GB on all development datasets.",
    )
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text()) if args.calibration else {}
    classification_logit_biases = {
        head: np.asarray(values, dtype=np.float32)
        for head, values in calibration.get("classification_logit_biases", {}).items()
    }

    raw_root = Path(os.environ.get("nnUNet_raw", "data/nnUNet_raw"))
    preprocessed_root = Path(os.environ.get("nnUNet_preprocessed", "data/nnUNet_preprocessed"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    tasks = discover_tasks(raw_root, preprocessed_root, args.inference_patch_voxels)
    task = tasks[args.task]
    raw_dir = task.raw_dir
    cases = group_cases(raw_dir / "imagesTs")
    if not cases:
        raise RuntimeError(f"No test cases found in {raw_dir / 'imagesTs'}")
    if any(len(paths) != len(task.modality_ids) for paths in cases.values()):
        raise ValueError("A test case has the wrong number of modalities")

    clinical = _read_clinical(
        raw_dir,
        list(cases),
        fit_identifiers=task.identifiers,
    )
    plans = json.loads((preprocessed_root / task.spec.folder_name / "nnUNetPlans.json").read_text())
    dataset_json = json.loads((raw_dir / "dataset.json").read_text())
    plans_manager = PlansManager(plans)
    configuration = plans_manager.get_configuration("3d_fullres")
    preprocessor = configuration.preprocessor_class(verbose=False)
    label_manager = plans_manager.get_label_manager(dataset_json)
    reader_writer = plans_manager.image_reader_writer_class()
    device = torch.device("cuda")
    model = load_model(args.checkpoint, int(config.get("base_width", 32)), device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, (identifier, paths) in enumerate(sorted(cases.items()), 1):
        image, _, properties = preprocessor.run_case(
            [str(path) for path in paths], None, plans_manager, configuration, dataset_json
        )
        logits, classifications = predict_array(
            model,
            task,
            image.astype(np.float32, copy=False),
            clinical[identifier],
            device,
            args.overlap,
            args.classification_top_fraction,
            args.foreground_threshold,
            classification_logit_biases,
        )
        segmentation = convert_predicted_logits_to_segmentation_with_correct_shape(
            logits,
            plans_manager,
            configuration,
            label_manager,
            properties,
            return_probabilities=False,
        )
        reader_writer.write_seg(segmentation, str(args.output_dir / f"{identifier}.nii.gz"), properties)
        row = {"case_id": identifier}
        if len(task.spec.classification_heads) == 1:
            head, class_count = next(iter(task.spec.classification_heads.items()))
            probability = classifications[head]
            if class_count == 2:
                row["label"] = float(probability[1])
            else:
                row.update({f"label_{label}": float(probability[label]) for label in range(class_count)})
        else:
            for head, class_count in task.spec.classification_heads.items():
                probability = classifications[head]
                row.update(
                    {f"{head}_{label}": float(probability[label]) for label in range(class_count)}
                )
        rows.append(row)
        print(f"[{index}/{len(cases)}] {identifier}: classification complete", flush=True)

    columns = list(rows[0])
    with (args.output_dir / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} masks and {args.output_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
