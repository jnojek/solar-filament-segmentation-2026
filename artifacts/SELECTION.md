# Selection record

The 419-image refit with small-proposal verification scored **0.29 public PQ** as Kaggle submission **58324985**. The release manifest binds the exact scored CSV.

A frozen verifier fitted from the 336-image model adds 150–699 pixel components to predictions from the 419-image refit. The verifier was transferred without fitting or tuning on the comparison groups. It preserved all original refit masks and added 48 matches across the previously evaluated confirmation and Udaipur groups. Pooled PQ was 0.33984, up from 0.32975 for the refit. See [matched transfer evidence](../docs/refit-verifier-transfer.md) for the per-fold scores and near-limb tradeoff.

## Historical baseline

The initial release selected the native U-Net and its connected components before evaluating the confirmation and Udaipur groups. The local contrast challenger removed unsupported detections and added 37 complete misses on development, exceeding the predefined cap of seven. The [baseline validation](validation.json) records that comparison.
