from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path


MODALITY_ALIASES = {
    "t1": "T1",
    "t1n": "T1",
    "t1c": "T1CE",
    "t1ce": "T1CE",
    "t1post": "T1CE",
    "t2": "T2",
    "t2w": "T2",
    "flair": "FLAIR",
    "t2f": "FLAIR",
    "ct": "CT",
    "rtp": "RTP",
    "adc": "ADC",
    "adc_4000_or_adc_vendor": "ADC",
    "hbv": "HBV",
    "pet": "PET",
}

MODALITIES = ("T1", "T1CE", "T2", "FLAIR", "CT", "RTP", "ADC", "HBV", "PET", "UNKNOWN")
MODALITY_TO_ID = {name: index for index, name in enumerate(MODALITIES)}


@dataclass(frozen=True)
class TaskSpec:
    key: str
    dataset_id: int
    folder_name: str
    segmentation_classes: int
    classification_heads: dict[str, int]


TASK_SPECS = {
    "brainmets": TaskSpec("brainmets", 1, "Dataset001_BrainMets", 4, {"origin": 2}),
    "mu_glioma": TaskSpec("mu_glioma", 2, "Dataset002_MU-Glioma_Post", 5, {"diagnosis": 3}),
    "ucsd_ptgb": TaskSpec("ucsd_ptgb", 3, "Dataset003_UCSD-PTGB", 2, {"idh": 2}),
    "picai": TaskSpec("picai", 4, "Dataset004_PICAI", 3, {"isup": 6}),
    "luna25": TaskSpec("luna25", 5, "Dataset005_LUNA25", 2, {"malignancy": 2}),
    "glioma_idh_grade": TaskSpec(
        "glioma_idh_grade", 7, "Dataset007_GliomaIDHType", 2, {"idh": 2, "grade": 3}
    ),
    "petwb_lung": TaskSpec("petwb_lung", 61, "Dataset061_PETWB_Lung", 2, {"cancer": 2}),
    "petwb_liver": TaskSpec("petwb_liver", 62, "Dataset062_PETWB_Liver", 2, {"cancer": 2}),
    "egct": TaskSpec("egct", 8, "Dataset008_EGCT", 2, {"tumor_type": 3}),
    "phlf": TaskSpec("phlf", 9, "Dataset009_PHLF", 5, {"phlf": 2}),
}


def canonical_modality_id(name: str) -> int:
    canonical = MODALITY_ALIASES.get(name.strip().lower(), "UNKNOWN")
    return MODALITY_TO_ID[canonical]


def safe_key(value: str) -> str:
    """Return a stable ModuleDict/CLI-safe key without assuming an ID format."""
    value = re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()
    return value or "task"


def task_spec_from_dataset(raw_dir: Path) -> TaskSpec:
    """Infer all output dimensions from dataset.json for an unseen dataset."""
    dataset = json.loads((raw_dir / "dataset.json").read_text())
    match = re.match(r"Dataset(?P<id>\d+)_?", raw_dir.name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Dataset folder must start with Dataset<ID>: {raw_dir.name}")
    dataset_id = int(match.group("id"))
    known = next((spec for spec in TASK_SPECS.values() if spec.folder_name == raw_dir.name), None)
    key = known.key if known is not None else safe_key(raw_dir.name)
    label_values = []
    for value in dataset["labels"].values():
        label_values.extend(value if isinstance(value, list) else [value])
    segmentation_classes = max(int(value) for value in label_values) + 1
    raw_heads = dataset.get("classification_labels", {})
    if not raw_heads:
        raise ValueError(f"{raw_dir.name}: dataset.json has no classification_labels")
    heads = {safe_key(name): len(classes) for name, classes in raw_heads.items()}
    return TaskSpec(key, dataset_id, raw_dir.name, segmentation_classes, heads)


def discover_task_specs(raw_root: Path) -> dict[str, TaskSpec]:
    """Discover known and hidden tasks deterministically from raw dataset folders."""
    discovered: dict[str, TaskSpec] = {}
    directories = sorted(
        (path for path in raw_root.glob("Dataset*") if (path / "dataset.json").is_file()),
        key=lambda path: path.name,
    )
    inferred = {path.name: task_spec_from_dataset(path) for path in directories}
    for key, known in TASK_SPECS.items():
        if known.folder_name in inferred:
            discovered[key] = inferred.pop(known.folder_name)
    for folder_name in sorted(inferred):
        spec = inferred[folder_name]
        key = spec.key
        suffix = 2
        while key in discovered:
            key = f"{spec.key}_{suffix}"
            suffix += 1
        discovered[key] = TaskSpec(
            key, spec.dataset_id, spec.folder_name, spec.segmentation_classes, spec.classification_heads
        )
    return discovered
