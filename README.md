# MICCAI FLARE 2026 AutoMSC Task 2

Publication code and model documentation for joint 3D segmentation and
case-level clinical classification in the FLARE 2026 AutoMSC challenge.

The repository contains the modality-aware, dual-resolution Glance--Focus
implementation used for EGCT and PHLF. The low-resolution **Glance** pathway
encodes the complete examination, while the high-resolution **Focus**
encoder--decoder processes sliding-window patches. The two pathways run in
parallel and their global, bottleneck, and lesion-weighted features are fused
for task-specific classification. Focus produces the final segmentation.

## Repository contents

```text
configs/       Training, refit, and validation-selected calibration settings
docs/          Architecture and validation figures
models/        Model card and integrity information for Zenodo-hosted weights
results/       Fold-0 comparison records, including 1-mm NSD
scripts/       Preprocessing, training orchestration, inference, and validation
unified/       Glance--Focus model, data pipeline, losses, and training code
```

No patient images, labels, clinical tables, predictions, or challenge test
data are included.

## Installation

The environment is locked with `uv`:

```bash
git clone https://github.com/xinformatics/miccaiflare2026task2.git
cd miccaiflare2026task2
uv sync --frozen
```

Set the nnU-Net data locations:

```bash
export nnUNet_raw="$PWD/data/nnUNet_raw"
export nnUNet_preprocessed="$PWD/data/nnUNet_preprocessed"
export nnUNet_results="$PWD/data/nnUNet_results"
```

Place the organizer-provided datasets below `$nnUNet_raw`; do not commit them.

## Released checkpoints

The trained weights are archived separately on Zenodo so that this Git
repository remains lightweight. **The permanent Zenodo DOI will be inserted
here when the prepared record is published.** Download and extract the model
archive into the repository root; the checkpoints will then appear under
`models/`.

| File | Purpose |
|---|---|
| `models/gf_automsc_foundation.pth` | Shared multi-dataset initialization checkpoint |
| `models/egct_final.pth` | Final single EGCT segmentation/tumor-type model |
| `models/phlf_final.pth` | Final single PHLF segmentation/status model |

The checkpoint hashes, provenance, and limitations are documented in
[`models/MODEL_CARD.md`](models/MODEL_CARD.md). Each downstream task uses one
checkpoint; the released models are not members of a same-target ensemble.

## Reproduce EGCT and PHLF inference

Plan and preprocess the datasets and create the fixed splits:

```bash
uv run nnUNetv2_plan_and_preprocess -d 8 9 -c 3d_fullres --verify_dataset_integrity
uv run python -m scripts.generate_automsc_splits \
  "$nnUNet_raw/Dataset008_EGCT" \
  "$nnUNet_preprocessed/Dataset008_EGCT" --folds 5 --seed 20260901
uv run python -m scripts.generate_automsc_splits \
  "$nnUNet_raw/Dataset009_PHLF" \
  "$nnUNet_preprocessed/Dataset009_PHLF" --folds 5 --seed 20260901
```

Run the released final models:

```bash
uv run python -m scripts.predict_public_test \
  --task egct \
  --checkpoint models/egct_final.pth \
  --output-dir predictions/Dataset008_EGCT_prediction \
  --classification-top-fraction 0.25 \
  --foreground-threshold 0.70 \
  --calibration configs/egct_calibration.json

uv run python -m scripts.predict_public_test \
  --task phlf \
  --checkpoint models/phlf_final.pth \
  --output-dir predictions/Dataset009_PHLF_prediction \
  --classification-top-fraction 0.25 \
  --calibration configs/phlf_calibration.json
```

The complete training and all-data refit commands are given in
[`docs/REPRODUCTION.md`](docs/REPRODUCTION.md).

## Local fold-0 results

These are local held-out results, not hidden-test scores. NSD uses a 1-mm
tolerance. PHLF segmentation averages liver, tumor, spleen, and muscle.

| Dataset | Model | DSC | NSD | BA | AUROC |
|---|---|---:|---:|---:|---:|
| EGCT | Organizer-style 3D MTL nnU-Net | 0.4058 | 0.0230 | 0.3333 | 0.4591 |
| EGCT | Glance--Focus | **0.4772** | **0.2679** | **0.5483** | **0.7047** |
| PHLF | Organizer-style 3D MTL nnU-Net | 0.3705 | 0.1552 | 0.5167 | **0.7500** |
| PHLF | Glance--Focus | **0.7154** | **0.3715** | **0.6833** | 0.6417 |

The PHLF Glance--Focus BA includes the saved fold-0 decision-boundary
adjustment. Its unadjusted BA is 0.5000; AUROC is unchanged.

## Glioma validation specialist

The paper also reports a public-validation glioma specialist based on the
organizer-provided AutoMSC multitask nnU-Net checkpoints. Dataset072 fold 3
provides segmentation and IDH, while Dataset007 fold 0 provides grade. Those
organizer-owned weights are not redistributed here and are not claimed as our
trained models. Obtain the baseline code and validation weights from the
[official AutoMSC-Baseline repository](https://github.com/medfm-flare/AutoMSC-Baseline).

The reported 122-case public-validation values are DSC 0.9020, NSD 0.8119,
IDH BA/AUROC 0.8882/0.9386, grade BA/AUROC 0.6519/0.8979, and overall 0.8484.
They should not be interpreted as performance of the released Glance--Focus
checkpoints.

## Citation

Please cite the accompanying FLARE 2026 AutoMSC technical report. Until its
proceedings record is available, cite this repository as:

```bibtex
@software{xinformatics2026automsc,
  author  = {{xinformatics}},
  title   = {MICCAI FLARE 2026 AutoMSC Task 2: Glance--Focus},
  year    = {2026},
  url     = {https://github.com/xinformatics/miccaiflare2026task2}
}
```

## License and acknowledgements

Code is released under the Apache License 2.0. The AutoMSC task, datasets, and
organizer baseline remain subject to their respective licenses and challenge
terms. This repository does not grant rights to redistribute the challenge
data or organizer checkpoints.

