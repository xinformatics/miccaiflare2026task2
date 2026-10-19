# Model card

## Released files

| Model | Bytes | SHA-256 |
|---|---:|---|
| `gf_automsc_foundation.pth` | 248,641,777 | `3714f77320a6fceeb977623f359f774b1039cb8ce6344e3c8d0487bca0a08f4e` |
| `egct_final.pth` | 149,743,036 | `5c74bd6eb714e813905995b2338002a9eb7abda21086a2a20216b6dc0712e5f0` |
| `phlf_final.pth` | 250,146,999 | `6adec18adb4ca61e65f3d50cc48e2d613d2e2c9cd73c14354422956a7406fdff` |

The files are distributed in the
[companion Zenodo archive](https://doi.org/10.5281/zenodo.23483494) rather than
in Git. The archive SHA-256 is
`e3ed9b48c8b198941419d98461185e020c4f1aaefa0fde6db8a4c34d263917a7`.
After extracting it into the repository root, verify the checkpoints with:

```bash
sha256sum models/*.pth
```

## Intended use

- `gf_automsc_foundation.pth` initializes supervised adaptation to a labeled
  AutoMSC-style dataset with declared modalities and task-specific heads.
- `egct_final.pth` is the single final model for Dataset008 EGCT.
- `phlf_final.pth` is the single final model for Dataset009 PHLF.

The checkpoints perform research segmentation and clinical classification.
They are not medical devices and must not be used for clinical decisions.

## Inputs and outputs

Inputs follow the organizer's nnU-Net NIfTI layout and modality order declared
in `dataset.json`. The code validates contiguous channels and restores masks to
native geometry. Classification outputs use the task-specific class schema.
The models do not infer an unknown clinical label space without compatible
supervision and output heads.

## Provenance

The foundation model and downstream EGCT/PHLF weights were trained using the
released AutoMSC development resources. The organizer-provided glioma baseline
checkpoints are not included. The repository contains no patient data or
clinical metadata.

## Limitations

Results were selected from a single local fold for EGCT and PHLF and have not
been established as clinical performance. Distribution shifts in scanners,
protocols, anatomy, missing modalities, label definitions, or preprocessing
may substantially reduce accuracy. The final checkpoints expect their declared
modalities; missing-modality robustness was not validated.

