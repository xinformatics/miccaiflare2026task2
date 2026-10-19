# Reproduction guide

## Architecture

The implementation uses a modality-set stem followed by two complementary
resolution pathways. Glance encodes a resized whole-volume view; Focus is a 3D
residual encoder--decoder applied to high-resolution patches. Classification
combines Glance global features, Focus bottleneck features, lesion-weighted
decoder features, a task embedding, and optional clinical features. Each
dataset owns its segmentation and clinical classification heads.

Glance does not select Focus patches in this implementation. Inference uses a
fixed sliding-window traversal and aggregates classification logits from the
foreground-rich top quarter of patches.

## Deterministic development training

After nnU-Net planning and split creation, initialize one fold-0 model per
dataset from the released foundation checkpoint:

```bash
uv run python -m unified.train \
  --config configs/public_test_train.yaml \
  --tasks egct \
  --initialize-from models/gf_automsc_foundation.pth \
  --validation-fold 0 --validation-task egct \
  --output-dir artifacts/egct_fold0

uv run python -m unified.train \
  --config configs/public_test_train.yaml \
  --tasks phlf \
  --initialize-from models/gf_automsc_foundation.pth \
  --validation-fold 0 --validation-task phlf \
  --output-dir artifacts/phlf_fold0
```

## All-data refit

```bash
uv run python -m unified.train \
  --config configs/public_test_refit.yaml \
  --tasks egct \
  --initialize-from artifacts/egct_fold0/checkpoint_final.pth \
  --output-dir artifacts/egct_all

uv run python -m unified.train \
  --config configs/public_test_refit.yaml \
  --tasks phlf \
  --initialize-from artifacts/phlf_fold0/checkpoint_final.pth \
  --output-dir artifacts/phlf_all

uv run python -m unified.train \
  --config configs/public_test_egct_decoder_joint_refit_all.yaml \
  --tasks egct \
  --initialize-from artifacts/egct_all/checkpoint_final.pth \
  --output-dir artifacts/egct_final

uv run python -m unified.train \
  --config configs/public_test_joint_refine.yaml \
  --tasks phlf \
  --initialize-from artifacts/phlf_all/checkpoint_final.pth \
  --output-dir artifacts/phlf_final
```

EGCT uses a fixed 3,000-step decoder-joint adaptation: the encoder and
classification-specific parameters are frozen while the decoder and
segmentation head remain constrained by both losses. PHLF uses the fixed
3,000-step joint refinement.

## Validation-selected adjustments

`configs/egct_calibration.json` and `configs/phlf_calibration.json` contain
intercept-only adjustments learned from fold-0 predictions. They change class
decision boundaries but cannot change AUROC ranking. The EGCT foreground
threshold is 0.70. No folds, seeds, or checkpoints are averaged at inference.

## Resource settings

- Training budget: 2,000,000 channel-voxels per sample.
- Batch budget: 4,194,304 channel-voxels, maximum batch size four.
- Inference budget: 1,000,000 channel-voxels, batch size one.
- Training peak for the full target model: approximately 15.9 GB.
- EGCT decoder-joint adaptation peak: approximately 10.14 GB.
- Standalone inference peak: below 12 GB in development measurements.

Run `python -m scripts.smoke_all_tasks` after preprocessing to verify tensor
shapes and memory on every discovered dataset.

