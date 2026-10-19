from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from unified.data import TaskPatchLoader, _read_clinical, discover_tasks
from unified.losses import joint_loss
from unified.model import GlanceFocusAutoMSC
from unified.specs import discover_task_specs


def class_weights(task, device: torch.device) -> dict[str, torch.Tensor]:
    output = {}
    for head, class_count in task.spec.classification_heads.items():
        counts = Counter(labels[head] for labels in task.labels.values())
        values = np.asarray(
            [len(task.labels) / (class_count * max(1, counts.get(index, 0))) for index in range(class_count)]
        )
        values = np.minimum(values, 5.0)
        values /= values.mean()
        output[head] = torch.tensor(values, dtype=torch.float32, device=device)
    return output


def save_checkpoint(path: Path, model, optimizer, scheduler, step: int, config: dict) -> None:
    temporary = path.with_suffix(".tmp")
    unwrapped = getattr(model, "_orig_mod", model)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config,
            "task_specs": {key: asdict(spec) for key, spec in unwrapped.task_specs.items()},
        },
        temporary,
    )
    os.replace(temporary, path)


def load_transfer_weights(model: torch.nn.Module, checkpoint_state: dict[str, torch.Tensor]) -> dict[str, object]:
    """Load all shape-compatible weights and preserve old task-embedding rows."""
    model_state = model.state_dict()
    loaded: list[str] = []
    skipped: list[str] = []
    for key, value in checkpoint_state.items():
        if key not in model_state:
            skipped.append(key)
            continue
        if model_state[key].shape == value.shape:
            model_state[key] = value
            loaded.append(key)
            continue
        skipped.append(key)
    model.load_state_dict(model_state)
    return {"loaded": loaded, "skipped": skipped}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/unified_train.yaml"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help="Load model weights but start a fresh optimizer, schedule, and step count.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--target-batch-voxels", type=int)
    parser.add_argument("--target-patch-voxels", type=int)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--validation-fold", type=int)
    parser.add_argument("--validation-task", default="glioma_idh_grade")
    args = parser.parse_args()
    if args.resume and args.initialize_from:
        parser.error("--resume and --initialize-from are mutually exclusive")
    config = yaml.safe_load(args.config.read_text())
    if args.steps is not None:
        config["total_steps"] = args.steps
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    if args.target_batch_voxels is not None:
        config["target_batch_voxels"] = args.target_batch_voxels
    if args.target_patch_voxels is not None:
        config["target_patch_voxels"] = args.target_patch_voxels
    if args.warmup_steps is not None:
        config["warmup_steps"] = args.warmup_steps
    if args.checkpoint_every is not None:
        config["checkpoint_every"] = args.checkpoint_every
    if args.log_every is not None:
        config["log_every"] = args.log_every

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    raw_root = Path(os.environ.get("nnUNet_raw", "data/nnUNet_raw"))
    preprocessed_root = Path(os.environ.get("nnUNet_preprocessed", "data/nnUNet_preprocessed"))
    tasks = discover_tasks(raw_root, preprocessed_root, config["target_patch_voxels"])
    if args.tasks:
        tasks = {key: tasks[key] for key in args.tasks}
    validation_identifiers: list[str] = []
    if args.validation_fold is not None:
        validation_task = tasks[args.validation_task]
        splits_path = preprocessed_root / validation_task.spec.folder_name / "splits_final.json"
        splits = json.loads(splits_path.read_text())
        validation_identifiers = splits[args.validation_fold]["val"]
        validation_set = set(validation_identifiers)
        training_identifiers = [value for value in validation_task.identifiers if value not in validation_set]
        tasks[args.validation_task] = replace(
            validation_task,
            identifiers=training_identifiers,
            labels={value: validation_task.labels[value] for value in training_identifiers},
            clinical=_read_clinical(
                validation_task.raw_dir,
                training_identifiers,
                fit_identifiers=training_identifiers,
            ),
        )
    if not tasks:
        raise SystemExit("No preprocessed tasks found")

    device = torch.device("cuda")
    model = GlanceFocusAutoMSC(
        config["base_width"], task_specs=discover_task_specs(raw_root)
    ).to(device)
    trainable_prefixes = tuple(config.get("trainable_parameter_prefixes", ()))
    if trainable_prefixes:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(trainable_prefixes))
        trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        total_count = sum(parameter.numel() for parameter in model.parameters())
        print(
            f"Restricted optimization to prefixes={trainable_prefixes}: "
            f"{trainable_count:,}/{total_count:,} trainable parameters"
        )
    if config.get("compile"):
        model = torch.compile(model, dynamic=True)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        fused=True,
    )
    warmup = config["warmup_steps"]
    total_steps = config["total_steps"]

    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    start_step = 0
    initialization_checkpoint = args.resume or args.initialize_from
    if initialization_checkpoint:
        checkpoint = torch.load(initialization_checkpoint, map_location="cpu", weights_only=False)
        if args.initialize_from:
            transfer_report = load_transfer_weights(model, checkpoint["model"])
            print(
                f"Transferred {len(transfer_report['loaded'])} tensors; "
                f"skipped {len(transfer_report['skipped'])} incompatible tensors"
            )
        else:
            model.load_state_dict(checkpoint["model"])
    if args.resume:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = checkpoint["step"] + 1

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest = {
        key: {
            "folder": task.spec.folder_name,
            "cases": len(task.identifiers),
            "identifiers": task.identifiers,
            "modalities": list(task.modality_ids),
            "patch_size": list(task.patch_size),
        }
        for key, task in tasks.items()
    }
    (output_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if validation_identifiers:
        (output_dir / "validation_identifiers.json").write_text(
            json.dumps(validation_identifiers, indent=2) + "\n"
        )
    metrics_path = output_dir / "metrics.jsonl"
    loader = TaskPatchLoader(
        tasks,
        workers=config["loader_workers"],
        seed=config["seed"],
        balanced_fraction=config.get("balanced_sample_fraction", 0.33),
        glance_size=config["glance_size"],
        glance_cache=Path(config["glance_cache"]),
    )
    weights = {key: class_weights(task, device) for key, task in tasks.items()}
    task_rng = random.Random(config["seed"])
    accumulation = config["gradient_accumulation"]
    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()

    print(f"Training {sum(len(task.identifiers) for task in tasks.values())} cases across {len(tasks)} tasks")
    def task_batch_size(task) -> int:
        channel_voxels = math.prod(task.patch_size) * len(task.modality_ids)
        return min(config["max_batch_size"], max(1, config["target_batch_voxels"] // channel_voxels))

    for key, task in tasks.items():
        batch_size = task_batch_size(task)
        print(f"  {key}: {len(task.identifiers)} cases, {len(task.modality_ids)} modalities, "
              f"patch={task.patch_size}, batch={batch_size}")

    task_keys = list(tasks)
    sampling_weights = []
    for key in task_keys:
        task = tasks[key]
        batch_size = task_batch_size(task)
        batches_per_pass = math.ceil(len(task.identifiers) / batch_size)
        weight = batches_per_pass ** config.get("task_sampling_power", 0.5)
        if key == "glioma_idh_grade":
            weight *= config.get("target_task_multiplier", 1.0)
        sampling_weights.append(weight)
    print("Task sampling: " + ", ".join(f"{key}={weight / sum(sampling_weights):.3f}" for key, weight in zip(task_keys, sampling_weights)))

    for step in range(start_step, total_steps):
        task_key = task_rng.choices(task_keys, weights=sampling_weights, k=1)[0]
        task = tasks[task_key]
        batch_size = task_batch_size(task)
        batch = loader.batch(task_key, batch_size)
        images = batch["images"].to(device, non_blocking=True)
        segmentation = batch["segmentation"].to(device, non_blocking=True)
        modality_ids = batch["modality_ids"].to(device, non_blocking=True)
        glance_images = batch["glance_images"].to(device, non_blocking=True)
        glance_segmentation = batch["glance_segmentation"].to(device, non_blocking=True)
        classifications = {name: value.to(device, non_blocking=True) for name, value in batch["classifications"].items()}
        clinical = batch["clinical"].to(device, non_blocking=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                images,
                modality_ids,
                task_key,
                glance_images=glance_images,
                clinical_features=clinical,
            )
            loss, metrics = joint_loss(
                outputs,
                segmentation,
                classifications,
                weights[task_key],
                glance_segmentation=glance_segmentation,
                segmentation_weight=config.get("segmentation_loss_weight", 1.0),
                coarse_weight=config.get("coarse_loss_weight", 0.25),
                classification_weight=(
                    config.get("target_classification_loss_weight", 1.5)
                    if task_key == "glioma_idh_grade"
                    else config.get("classification_loss_weight", 0.75)
                ),
                classification_head_weights=(
                    config.get("target_classification_head_weights")
                    if task_key == "glioma_idh_grade"
                    else None
                ),
                classification_label_smoothing=(
                    config.get("target_classification_label_smoothing", 0.0)
                    if task_key == "glioma_idh_grade"
                    else config.get("classification_label_smoothing", 0.0)
                ),
            )
            loss = loss / accumulation
        loss.backward()
        if (step + 1) % accumulation == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if step % config["log_every"] == 0:
            elapsed = time.time() - start_time
            memory = torch.cuda.max_memory_allocated() / 2**30
            record = {
                "step": step,
                "task": task_key,
                "batch": batch_size,
                **metrics,
                "lr": scheduler.get_last_lr()[0],
                "gpu_gib": round(memory, 2),
                "steps_per_s": round((step - start_step + 1) / elapsed, 4),
            }
            line = json.dumps(record)
            print(line, flush=True)
            with metrics_path.open("a") as handle:
                handle.write(line + "\n")
        if step and step % config["checkpoint_every"] == 0:
            save_checkpoint(output_dir / f"checkpoint_{step:07d}.pth", model, optimizer, scheduler, step, config)
            save_checkpoint(output_dir / "checkpoint_latest.pth", model, optimizer, scheduler, step, config)

    save_checkpoint(output_dir / "checkpoint_final.pth", model, optimizer, scheduler, total_steps - 1, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
