#!/usr/bin/env python3
"""Exercise one real train/inference step for every discovered AutoMSC task."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import yaml

from unified.data import TaskPatchLoader, discover_tasks
from unified.losses import joint_loss
from unified.model import GlanceFocusAutoMSC
from unified.specs import discover_task_specs
from unified.train import class_weights


def device_memory_mib() -> int:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    for line in output.splitlines():
        pid, memory = (value.strip() for value in line.split(","))
        if int(pid) == os.getpid():
            return int(memory)
    return 0


def execute_phase(args, config: dict) -> dict[str, object]:
    raw_root = Path(os.environ.get("nnUNet_raw", "data/nnUNet_raw"))
    preprocessed_root = Path(os.environ.get("nnUNet_preprocessed", "data/nnUNet_preprocessed"))
    voxel_budget = config["target_patch_voxels"] if args.phase == "train" else 1_000_000
    tasks = discover_tasks(raw_root, preprocessed_root, voxel_budget)
    task = tasks[args.task]
    device = torch.device("cuda")
    model = GlanceFocusAutoMSC(
        config["base_width"], task_specs=discover_task_specs(raw_root)
    ).to(device)
    loader = TaskPatchLoader(
        {args.task: task},
        workers=min(2, config["loader_workers"]),
        seed=config["seed"],
        balanced_fraction=config.get("balanced_sample_fraction", 1.0),
        glance_size=config["glance_size"],
        glance_cache=Path(config["glance_cache"]),
    )
    channel_voxels = math.prod(task.patch_size) * len(task.modality_ids)
    batch_size = (
        min(config["max_batch_size"], max(1, config["target_batch_voxels"] // channel_voxels))
        if args.phase == "train"
        else 1
    )
    batch = loader.batch(args.task, batch_size, augment=args.phase == "train")
    images = batch["images"].to(device)
    segmentation = batch["segmentation"].to(device)
    modality_ids = batch["modality_ids"].to(device)
    glance = batch["glance_images"].to(device)
    clinical = batch["clinical"].to(device)
    classifications = {name: value.to(device) for name, value in batch["classifications"].items()}
    torch.cuda.reset_peak_memory_stats()
    if args.phase == "train":
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, fused=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(images, modality_ids, args.task, glance_images=glance, clinical_features=clinical)
            loss, _ = joint_loss(
                output,
                segmentation,
                classifications,
                class_weights(task, device),
                glance_segmentation=batch["glance_segmentation"].to(device),
                coarse_weight=config["coarse_loss_weight"],
                classification_weight=config["classification_loss_weight"],
                classification_label_smoothing=config["classification_label_smoothing"],
            )
        loss.backward()
        optimizer.step()
        limit_mib = 24 * 1024
    else:
        model.eval()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(images, modality_ids, args.task, glance_images=glance, clinical_features=clinical)
        limit_mib = 12 * 1024
    torch.cuda.synchronize()
    record = {
        "task": args.task,
        "phase": args.phase,
        "cases": len(task.identifiers),
        "channels": len(task.modality_ids),
        "patch_size": list(task.patch_size),
        "batch_size": batch_size,
        "segmentation_classes": task.spec.segmentation_classes,
        "classification_heads": task.spec.classification_heads,
        "segmentation_shape": list(output["segmentation"].shape),
        "classification_shapes": {name: list(value.shape) for name, value in output["classifications"].items()},
        "torch_peak_mib": round(torch.cuda.max_memory_allocated() / 2**20),
        "device_mib": device_memory_mib(),
        "limit_mib": limit_mib,
    }
    if record["device_mib"] >= limit_mib:
        raise RuntimeError(f"{args.task}/{args.phase}: {record['device_mib']} MiB >= {limit_mib} MiB")
    args.phase_output.write_text(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/public_test_train.yaml"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/all_dataset_smoke.json"))
    parser.add_argument("--phase", choices=["all", "train", "infer"], default="all")
    parser.add_argument("--task")
    parser.add_argument("--phase-output", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.phase != "all":
        if not args.task or args.phase_output is None:
            parser.error("worker phases require --task and --phase-output")
        execute_phase(args, config)
        return

    raw_root = Path(os.environ.get("nnUNet_raw", "data/nnUNet_raw"))
    preprocessed_root = Path(os.environ.get("nnUNet_preprocessed", "data/nnUNet_preprocessed"))
    tasks = discover_tasks(raw_root, preprocessed_root, config["target_patch_voxels"])
    records = []
    with tempfile.TemporaryDirectory(prefix="automsc_smoke_") as temporary:
        for task in tasks:
            for phase in ("train", "infer"):
                output = Path(temporary) / f"{task}_{phase}.json"
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "scripts.smoke_all_tasks",
                        "--config",
                        str(args.config),
                        "--phase",
                        phase,
                        "--task",
                        task,
                        "--phase-output",
                        str(output),
                    ],
                    check=True,
                )
                records.append(json.loads(output.read_text()))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"status": "pass", "records": records}, indent=2) + "\n")
    print(f"PASS: {len(tasks)} tasks completed training and inference checks; {args.report}")


if __name__ == "__main__":
    main()
