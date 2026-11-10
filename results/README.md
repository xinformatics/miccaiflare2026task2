# Results

The JSON files preserve the case-level fold-0 evaluation records used for the
publication comparison. Files ending in `with_nsd.json` include normalized
surface Dice at a 1-mm tolerance.

The organizer-style MTL nnU-Net evaluations used `--no-mirroring`. All four
files now use mean case-wise foreground DSC. Earlier pooled-voxel nnU-Net DSC
values are not comparable and have been replaced. The PHLF Glance--Focus file
contains raw 0.5-threshold balanced accuracy; the same-fold post-hoc boundary
in `configs/phlf_calibration.json` raises BA to 0.6833 and leaves AUROC
unchanged, but is not used in the fair comparison.

The `*_bootstrap_ci95.json` files contain 5,000-replicate class-stratified
bootstrap intervals. The `*_timing_rtx_pro_6000.json` files record ten-case
benchmarks whose scope is preprocessed-array reading plus model inference;
they exclude raw-image preprocessing, model loading, and output serialization.

These are local fold-0 results and must not be described as organizer
hidden-test scores.

