#!/usr/bin/env bash
set -euo pipefail
python -m reproduction.data --output artifacts/toy_data --train-count 8 --val-count 2 --height 32 --width 32
python -m reproduction.train --data artifacts/toy_data --output artifacts/smoke --epochs 1 --height 32 --width 32 --classes 4 --model-width 8 --batch-size 2
python -m reproduction.predict --checkpoint artifacts/smoke/model.pt --image artifacts/toy_data/val/images/0000.png --output artifacts/smoke/prediction_raw_ids.png
python -m pytest -q tests
