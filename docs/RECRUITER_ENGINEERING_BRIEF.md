# FRRN-A — Engineering Snapshot for Recruiters

**Kumar Abhinav** · M.S. Computer Science, CSUDH · Completed 2026 graduate project (CSC 590) · **Grade A**

## Problem and engineering challenge

Semantic segmentation labels every pixel of a road-scene image. The project aimed to adapt an older published FRRN-A implementation to a newer GPU environment and execute a complete limited-scale training and inference experiment. It continued an earlier CSC 599 investigation of autonomous perception and LiDAR-based sensing, but **the trained FRRN-A model used camera-image input rather than LiDAR fusion**.

## Completed work and verifiable outcomes

| Area | What was completed |
| --- | --- |
| Compatibility and debugging | Resolved legacy TensorFlow API/dependency issues using `tf.compat.v1` and `tf_slim`, plus environment and GPU memory constraints. |
| Experiment execution | Trained the **17,747,620-parameter FRRN-A** model for **100 epochs** using a **paid Paperspace Gradient GPU**. |
| Data | Worked with **85 training / 15 validation** image-mask pairs from a Cityscapes/Darmstadt-style urban scene subset using **320 × 640** crops and **batch size 1**. |
| Outputs | Produced training checkpoints, a loss trace, and visual comparisons of inputs, ground truth, and generated segmentation masks within the original environment. |
| Academic deliverables | **41-page project report** (May 17, 2026), **8-slide presentation** (May 13, 2026), and recorded demonstration. |

**What the result supports:** the course-scale model was trained and produced qualitative predictions, with visible strengths on larger road-like regions and weaknesses on thin boundaries and distant objects.

**What it does not support:** a claim of official Cityscapes mIoU, per-class accuracy, road deployment, novel FRRN architecture, or a production-ready autonomous-driving system. No official full-validation segmentation accuracy was calculated in the submitted report. Do not substitute a parameter count or training epoch count for accuracy.

## What a recruiter can inspect today

- [Original 41-page report — complete searchable text](CSC590_FINAL_REPORT_SEARCHABLE_TRANSCRIPT.txt) (figure images omitted from text export)
- [Original 8-slide presentation — complete searchable text](CSC590_FINAL_PRESENTATION_SEARCHABLE_TRANSCRIPT.txt)
- [Academic progression and evidence index](PROJECT_HISTORY_AND_EVIDENCE.md), including figure references to report pp. 23–24 and 34–35
- [Archived artifacts/reproducibility matrix](REPRODUCIBILITY_STATUS.md)
- [Independent reproducible PyTorch FRRN-inspired model](../reconstruction/README.md), with synthetic training and inference and GitHub Actions smoke tests

**Original-model preservation:** the historical paid Paperspace workspace is no longer accessible. This public repository therefore does not contain its exact TensorFlow model source, original checkpoint weights, or restricted dataset images. The new runnable PyTorch model is **not represented as those recovered historical artifacts**.

**Technical focus:** Python · TensorFlow compatibility · deep learning · semantic segmentation · cloud/GPU debugging · reproducibility · testable research software

[Portfolio](https://kumarabhinavtech.com) · [Repository main page](../README.md)
