# Research integrity and reproducibility matrix

**Last updated: 2026-11-23.** Completion and preservation are different questions.

| Artifact/claim | 2026 course work | Presently available on GitHub | Interpretation |
| --- | --- | --- | --- |
| 41-page CSC 590 report | Submitted, May 17 | Indexed and described; original PDF not uploaded in repository | Documented historical results |
| 8-slide CSC 590 presentation | Delivered, May 13 | Indexed and described; original PDF not uploaded in repository | Historical presentation |
| [Original demo](https://drive.google.com/file/d/1W5sv3qTNaZtnd3INmvQyWFEA0zLoet2B/view) | Recorded | External link (access may vary) | Evidence, not executable source |
| Original training/evaluation code | Ran inside paid Paperspace GPU workspace | **Unavailable**, including `docker/main.py` and `docker/models/FRRN.py` | Original exact rerun unsupported |
| Original trained FRRN-A checkpoint | Generated | **Unavailable** | Can't restore 2026 weights from screenshots |
| Original 85/15 image subset | Used | **Unavailable**; Cityscapes dataset cannot be redistributed without rights | Exact historical split unavailable |
| Historical loss log and output masks | Generated in 2026 | Raw files unavailable; figures survive in report/presentation | Do not invent raw data |
| Legacy root helper scripts | Partially preserved | Present in root; incomplete and tied to Paperspace | **Historical reference only**, not runnable standalone |
| Official full Cityscapes mIoU | Not computed | Not reported | No official benchmark claim |
| Grade A | Author lists on portfolio source resume | Stated in README; public transcript not included | Academic assessment |
| Highest class score | Author-reported | No independent class ranking record attached | Clearly mark as author-reported |
| CSC 599 LiDAR research | Earlier thesis-track course research, Spring 2025 | Timeline described; original full thesis source not included | Academic background, not sensor-fusion source |
| **New** PyTorch FRRN-style implementation | Written November 2026 | **Yes**, in `reconstruction/` | Runnable modern educational reproduction—not original FRRN-A |
| New synthetic-data training/inference/tests | Tested locally | **Yes** | GitHub Actions workflow added for independent checkout testing; CI result must be checked separately |

## Run the modern reconstructed workflow

```bash
git clone https://github.com/kumi2001/FRRN-A-Semantic-Segmentation-Autonomous-Driving.git
cd FRRN-A-Semantic-Segmentation-Autonomous-Driving/reconstruction
python -m pip install -r requirements.txt
bash run_smoke.sh
```

It generates synthetic input data, trains a separate compact PyTorch segmentation model, writes a new checkpoint, predicts a class-ID PNG and checks functionality.

**Do not equate any synthetic smoke-test loss, IoU, output PNG or checkpoint with the May 2026 original data or trained weights.**

## Provenance and restrictions

The May 2026 work adapted the established FRRN architecture (Pohlen et al., CVPR 2017). The project did not invent FRRN. The Cityscapes dataset and imagery are not redistributed here. The modern model is a different, scaled-down FRRN-inspired model.

The 2026 work remains academically completed and graded. Its *exact executable preservation* is incomplete because the original paid Paperspace Gradient workspace is no longer accessible.
