#!/usr/bin/env python3
"""Strictly validate the two-dataset AutoMSC testing submission."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk


DATASETS = {
    "Dataset008_EGCT": {
        "prediction_dir": "Dataset008_EGCT_prediction",
        "columns": ["case_id", "label_0", "label_1", "label_2"],
        "labels": {0, 1},
    },
    "Dataset009_PHLF": {
        "prediction_dir": "Dataset009_PHLF_prediction",
        "columns": ["case_id", "label"],
        "labels": {0, 1, 2, 3, 4},
    },
}
CHANNEL_SUFFIX = re.compile(r"_\d{4}$")


def case_id(path: Path) -> str:
    name = path.name.removesuffix(".nii.gz").removesuffix(".nii")
    return CHANNEL_SUFFIX.sub("", name)


def close_tuple(left, right, tolerance: float = 1e-5) -> bool:
    return np.allclose(np.asarray(left), np.asarray(right), rtol=0.0, atol=tolerance)


def validate_dataset(submission: Path, testdata: Path, dataset: str, spec: dict) -> list[str]:
    errors: list[str] = []
    image_dir = testdata / dataset / "imagesTs"
    output_dir = submission / "predictions" / spec["prediction_dir"]
    inputs = {case_id(path): path for path in sorted(image_dir.glob("*.nii*"))}
    # Prediction filenames are the complete case ID. Only source images carry
    # the trailing nnU-Net channel suffix (for example ``_0000``).
    masks = {
        path.name.removesuffix(".nii.gz").removesuffix(".nii"): path
        for path in sorted(output_dir.glob("*.nii*"))
    } if output_dir.is_dir() else {}
    if set(masks) != set(inputs):
        errors.append(
            f"{dataset}: mask IDs differ; missing={sorted(set(inputs)-set(masks))}, "
            f"extra={sorted(set(masks)-set(inputs))}"
        )

    csv_path = output_dir / "predictions.csv"
    rows: list[dict[str, str]] = []
    if not csv_path.is_file():
        errors.append(f"{dataset}: missing {csv_path}")
    else:
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != spec["columns"]:
                errors.append(f"{dataset}: CSV columns {reader.fieldnames}, expected {spec['columns']}")
            rows = list(reader)
        ids = [row.get("case_id", "") for row in rows]
        if len(ids) != len(set(ids)):
            errors.append(f"{dataset}: duplicate CSV case IDs")
        if set(ids) != set(inputs):
            errors.append(
                f"{dataset}: CSV IDs differ; missing={sorted(set(inputs)-set(ids))}, "
                f"extra={sorted(set(ids)-set(inputs))}"
            )
        probability_columns = spec["columns"][1:]
        for row in rows:
            try:
                values = np.asarray([float(row[column]) for column in probability_columns])
            except (KeyError, ValueError) as error:
                errors.append(f"{dataset}/{row.get('case_id', '?')}: invalid probability ({error})")
                continue
            if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
                errors.append(f"{dataset}/{row.get('case_id', '?')}: probabilities outside [0,1]")
            if dataset == "Dataset008_EGCT" and not np.isclose(values.sum(), 1.0, atol=1e-5):
                errors.append(f"{dataset}/{row.get('case_id', '?')}: probabilities sum to {values.sum():.8f}")

    for identifier in sorted(set(inputs) & set(masks)):
        source = sitk.ReadImage(str(inputs[identifier]))
        mask = sitk.ReadImage(str(masks[identifier]))
        if source.GetSize() != mask.GetSize():
            errors.append(f"{dataset}/{identifier}: size {mask.GetSize()} != {source.GetSize()}")
        for name, left, right in (
            ("spacing", mask.GetSpacing(), source.GetSpacing()),
            ("origin", mask.GetOrigin(), source.GetOrigin()),
            ("direction", mask.GetDirection(), source.GetDirection()),
        ):
            if not close_tuple(left, right):
                errors.append(f"{dataset}/{identifier}: {name} differs")
        observed = set(np.unique(sitk.GetArrayViewFromImage(mask)).astype(int).tolist())
        if not observed <= spec["labels"]:
            errors.append(f"{dataset}/{identifier}: invalid mask labels {sorted(observed - spec['labels'])}")
    print(f"{dataset}: {len(inputs)} expected cases, {len(masks)} masks, {len(rows)} CSV rows")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path, help="Unzipped submission root containing predictions/ and codebase/")
    parser.add_argument("--testdata", type=Path, default=Path("testdata/extracted"))
    args = parser.parse_args()
    errors: list[str] = []
    for dataset, spec in DATASETS.items():
        errors.extend(validate_dataset(args.submission, args.testdata, dataset, spec))
    if not (args.submission / "codebase" / "README.md").is_file():
        errors.append("Missing codebase/README.md")
    if errors:
        print("\nFAILED", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        raise SystemExit(1)
    print("PASS: exact cases, CSV schemas/probabilities, mask labels, and image geometry verified")


if __name__ == "__main__":
    main()
