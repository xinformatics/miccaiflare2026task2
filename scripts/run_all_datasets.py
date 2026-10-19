#!/usr/bin/env python3
"""Automated dataset.json-driven preprocessing, training, and prediction."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from unified.specs import discover_task_specs


def run(command: list[str], env: dict[str, str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/nnUNet_raw"))
    parser.add_argument("--preprocessed-root", type=Path, default=Path("data/nnUNet_preprocessed"))
    parser.add_argument("--results-root", type=Path, default=Path("data/nnUNet_results"))
    parser.add_argument("--config", type=Path, default=Path("configs/public_test_train.yaml"))
    parser.add_argument("--refit-config", type=Path, default=Path("configs/public_test_refit.yaml"))
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/automated"))
    parser.add_argument("--predictions", type=Path, default=Path("predictions"))
    parser.add_argument("--datasets", nargs="*", type=int, help="Dataset IDs; default discovers every Dataset* folder")
    parser.add_argument("--steps", type=int, help="Global override used identically for every dataset")
    parser.add_argument("--skip-preprocessing", action="store_true")
    args = parser.parse_args()

    args.raw_root = args.raw_root.resolve()
    args.preprocessed_root = args.preprocessed_root.resolve()
    args.results_root = args.results_root.resolve()
    env = os.environ.copy()
    env.update(
        nnUNet_raw=str(args.raw_root),
        nnUNet_preprocessed=str(args.preprocessed_root),
        nnUNet_results=str(args.results_root),
    )
    specs = discover_task_specs(args.raw_root)
    selected = {
        key: spec for key, spec in specs.items() if args.datasets is None or spec.dataset_id in set(args.datasets)
    }
    if not selected:
        raise SystemExit("No matching Dataset* folders containing dataset.json")

    if not args.skip_preprocessing:
        run(
            [
                "nnUNetv2_plan_and_preprocess",
                "-d",
                *[str(spec.dataset_id) for spec in selected.values()],
                "-c",
                "3d_fullres",
                "--verify_dataset_integrity",
            ],
            env,
        )
    for key, spec in selected.items():
        preprocessed = args.preprocessed_root / spec.folder_name
        run(
            [
                sys.executable,
                "-m",
                "scripts.generate_automsc_splits",
                str(args.raw_root / spec.folder_name),
                str(preprocessed),
                "--folds",
                "5",
                "--seed",
                "20260901",
            ],
            env,
        )
        audit_output = args.artifacts / f"{spec.folder_name}_fold0"
        output = args.artifacts / f"{spec.folder_name}_all_refit"
        train = [
            sys.executable,
            "-m",
            "unified.train",
            "--config",
            str(args.config),
            "--tasks",
            key,
            "--validation-fold",
            "0",
            "--validation-task",
            key,
            "--output-dir",
            str(audit_output),
        ]
        if args.initialize_from:
            train.extend(("--initialize-from", str(args.initialize_from)))
        if args.steps is not None:
            train.extend(("--steps", str(args.steps)))
        run(train, env)
        refit = [
            sys.executable,
            "-m",
            "unified.train",
            "--config",
            str(args.refit_config),
            "--tasks",
            key,
            "--initialize-from",
            str(audit_output / "checkpoint_final.pth"),
            "--output-dir",
            str(output),
        ]
        run(refit, env)
        prediction_dir = args.predictions / f"{spec.folder_name}_prediction"
        run(
            [
                sys.executable,
                "-m",
                "scripts.predict_public_test",
                "--task",
                key,
                "--checkpoint",
                str(output / "checkpoint_final.pth"),
                "--classification-top-fraction",
                "0.25",
                "--inference-patch-voxels",
                "1000000",
                "--output-dir",
                str(prediction_dir),
            ],
            env,
        )


if __name__ == "__main__":
    main()
