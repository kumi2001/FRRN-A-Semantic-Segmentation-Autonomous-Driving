# FRRN-style two-stream segmentation: reproducible modern reconstruction

**Status: new, independent PyTorch reproduction.** This folder was developed in November 2026 to provide an executable training/inference/metrics workflow where the original May 2026 TensorFlow project files are not available. **It is not recovered historical source code, does not reproduce the original May weights/results and is not a one-to-one FRRN-A implementation.** It demonstrates the FRRN two-stream idea in an accessible, scaled-down implementation.

## Provenance and important limitations

- Original **CSC 590 May 2026** project used TensorFlow compatibility mode, Cityscapes/Darmstadt-style **85 training and 15 validation image-mask pairs**, 100 epochs, 320×640 image crops and a ~17.7 million parameter **FRRN-A** model, in Paperspace Gradient. The original checkpoint had **four output channels**, according to May 11 troubleshooting notes.
- This **new** independent experiment uses a compact FRRN-**style** architecture in `reproduction/model.py`, not the original TensorFlow model. Its saved PyTorch checkpoints are **not compatible** with the original TensorFlow model/checkpoints. Any synthetic results are only smoke tests, *not* Cityscapes scores.
- The originally generated checkpoint, the original 85/15 dataset subset, `docker/models/FRRN.py` and the original `predict_frrna_custom.py` could not be recovered from the materials currently accessible. Do not claim original results are recreated.
- No raw Cityscapes imagery or label masks are redistributed. [Cityscapes terms](https://www.cityscapes-dataset.com/license/) require a separate authorized download and forbid dataset redistribution. Cityscapes original IDs (e.g. 7, 8, 11, 24, 26...) **must be mapped into contiguous classes** for training. The historical 4-class mapping is unknown and cannot safely be inferred.

## Quick start (works without restricted data)

Python 3.10+ recommended. In a fresh virtual environment:

```bash
python -m pip install -r requirements.txt
bash run_smoke.sh
```

This generates deterministic toy RGB/mask pairs, runs **one CPU training epoch**, writes a PyTorch checkpoint and training log, performs inference to a raw PNG label-ID mask, and runs unit/integration tests. Output goes under `artifacts/` (not included in version control).

## Run on independently authorized data

Make folders `my_data/{train,val}/{images,masks}` where images are RGB PNG/JPG and masks are **single-channel PNG integer labels** in `0..C-1` or `255` for ignore. Each image `example.png` should have a corresponding `example.png` mask. Use *your own authorized dataset*. The toy example uses classes 0=background, 1=road, 2=vehicle-like rectangle, 3=pedestrian-like rectangle.

```bash
python -m reproduction.train \
  --data my_data --output artifacts/my_experiment --epochs 100 \
  --height 320 --width 640 --classes 4 --model-width 16 --batch-size 1 --device cuda

python -m reproduction.predict \
  --checkpoint artifacts/my_experiment/model.pt \
  --image my_data/val/images/example.png \
  --output artifacts/my_experiment/predicted_class_ids.png
```

The metrics in `history.json` are computed from actual raw mask class IDs, unlike a screenshot-color proxy. They are **metrics of this new model on your supplied data** and should never be presented as the official Cityscapes benchmark or the historical project results.

## Files

- `reproduction/model.py` — minimal two-stream residual model inspired by FRRN.
- `reproduction/data.py` — strict paired-data loader and deterministic toy generator.
- `reproduction/train.py` — repeatable training/evaluation and metadata-preserving checkpoint.
- `reproduction/metrics.py` — confusion matrix, pixel accuracy, per-class IoU and mIoU from raw class-ID masks.
- `reproduction/predict.py` — checkpoint-based prediction to a raw ID mask.
- `tests/test_smoke.py` — end-to-end and metric validation tests.

## References

- Pohlen et al. (CVPR 2017), *Full-Resolution Residual Networks for Semantic Segmentation in Street Scenes* (FRRN architecture concept).
- Cordts et al. (CVPR 2016), *The Cityscapes Dataset for Semantic Urban Scene Understanding*.
- Kumar Abhinav, **CSC 590 report**, *Teaching Autonomous Vehicles to See the Road: Pixel-Level Scene Understanding with FRRN-A* (May 17, 2026, CSUDH; faculty guidance by Dr. Bin Tang).
