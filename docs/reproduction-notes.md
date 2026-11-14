# Reproducing the detector and native mask export

Use Python 3.12.11 and the Windows CUDA 12.8 package lock. Obtain the competition data through Kaggle and extract it to `data/MAGFiLO_1.0_Kaggle_2026/`. Open `reproduce.ipynb` from the checkout root and execute every cell. The notebook verifies the release file identities, regenerates detector proposals and the selected masks, then independently checks every expected image and the final CSV hash.

```powershell
uv venv --python 3.12.11 .venv
uv pip sync --python .venv/Scripts/python.exe --extra-index-url https://download.pytorch.org/whl/cu128 requirements-windows.lock
```

The exact GPU replay uses the shipped weights and no training labels. The two GPU stages run sequentially. Other software or hardware may change floating-point decisions near a threshold. See [the matched design](detector-pivot.md), [the earlier U-Net fit](refit-verifier-transfer.md) and the detector training receipt for lineage.

To retrain the detector, copy the byte-identical training source and configuration from `scripts/yolo11s_scratch.py` and `artifacts/detector-config.json` to `.local/yolo11s_scratch.py` and `.local/yolo11s-scratch-config.json`, respectively. Copy `artifacts/refit-split.json` to `runs/higher-data-refit-plan/split.json`. Set `PYTHONPATH=src` and run the script with the stages `prepare`, `profile` and `train` in order. The exact replay uses the archived trained checkpoint.

For the native continuation, copy the included native training script and native config to `.local/yolo11s_native_tiles.py` and `.local/yolo11s-native-tiles-config.json` and run its `prepare`, `profile` and `train` stages after rebuilding the full-disk scratch parent.
