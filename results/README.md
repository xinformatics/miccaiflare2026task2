# Results

The JSON files preserve the case-level fold-0 evaluation records used for the
publication comparison. Files ending in `with_nsd.json` include normalized
surface Dice at a 1-mm tolerance.

The organizer-style MTL nnU-Net evaluations used `--no-mirroring`, matching the
previously stored DSC and classification probabilities. The PHLF
Glance--Focus result file contains the raw 0.5-threshold balanced accuracy;
the paper reports 0.6833 after applying the fixed boundary adjustment in
`configs/phlf_calibration.json`. AUROC is unchanged.

These are local fold-0 results and must not be described as organizer
hidden-test scores.

