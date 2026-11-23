# Teaching Autonomous Vehicles to See the Road
### Completed graduate research: CSC 599 (Spring 2025) → CSC 590 (Spring 2026)

**Kumar Abhinav** · M.S. Computer Science, California State University, Dominguez Hills  
**CSC 590 instructor:** Dr. Bin Tang

> **The May 2026 project was completed, submitted, presented and graded—not abandoned.** I executed the final FRRN-A semantic segmentation experiment on a **paid Paperspace Gradient GPU**, trained for 100 epochs, generated checkpoints and prediction masks, and submitted a **41-page final report** and **8-slide presentation**. I earned an **A in CSC 590**, and report receiving the **highest score in the class**. The paid Paperspace workspace is no longer accessible; the missing historical code/weights are an **archival gap**, not evidence that the project was unfinished.

[**Academic timeline and evidence**](docs/PROJECT_HISTORY_AND_EVIDENCE.md) · [**Reproducibility status**](docs/REPRODUCIBILITY_STATUS.md) · [**Runnable independent reconstruction**](reconstruction/README.md) · [**Original demo recording**](https://drive.google.com/file/d/1W5sv3qTNaZtnd3INmvQyWFEA0zLoet2B/view)

## CSC 599 → CSC 590: the complete research story

These courses form one **academic research progression** in autonomous-vehicle perception:

| Stage | When | Focus | Outcome |
| --- | --- | --- | --- |
| **CSC 599 — LiDAR / Autonomous Perception & Mapping** | Spring **2025** | Earlier thesis-track investigation of LiDAR point-cloud sensing, perception, mapping, and autonomy-related decisions within a three-faculty research framework. | Established the broader autonomous-perception research direction. |
| **CSC 590 — FRRN-A Semantic Segmentation** | Spring **2026** | Master's **project-track** continuation of this topic, now focusing on implemented **camera-image pixel-level semantic segmentation**. | Completed trained FRRN-A workflow, outputs, report, presentation and course assessment. |

**Research continuity does not mean direct sensor fusion.** The final report does not claim that the FRRN-A network processed live LiDAR data or was deployed in an autonomous vehicle. CSC 599 supplied earlier research foundations; CSC 590 completed a related but distinct, implemented camera-perception experiment.

~~~mermaid
flowchart LR
  A["CSC 599 / Spring 2025<br/>LiDAR, sensing, mapping<br/>research foundation"] --> B["CSC 590 / Spring 2026<br/>FRRN-A image segmentation<br/>trained on paid Paperspace GPU"]
  B --> C["100 epochs; checkpoints; masks<br/>41-page report; 8 slides<br/>grade A"]
  B -. "new, distinct code" .-> D["Reconstruction / Nov 2026<br/>separate runnable PyTorch<br/>FRRN-inspired demonstration"]
~~~

## Original, completed FRRN-A experiment (May 2026)

| Component | Historical experiment, as documented |
| --- | --- |
| Model | **FRRN-A**, Full-Resolution Residual Network-A; **17,747,620 parameters** |
| Task | Per-pixel classification of urban road images |
| Dataset | **85 training + 15 validation** image-mask pairs from a Cityscapes/Darmstadt-style subset |
| Training inputs | **320 × 640** crops; batch size **1** |
| Training | **100 epochs** on a **paid Paperspace Gradient GPU** |
| Engineering | Legacy TensorFlow compatibility fixes, **tf.compat.v1 / tf_slim**, GPU dependency and environment troubleshooting |
| Outputs actually produced | Trained checkpoint files, loss logs, predicted segmentation masks, visual input/ground-truth/prediction comparisons |
| Evidence | Final report **May 17, 2026** (41 pages); presentation **May 13, 2026** (8 slides); recorded demonstration |

The report's **page 23** contains the training-loss graph. **Page 24** contains actual **input / ground-truth / prediction** comparisons; **pages 34–35** include additional masks. The presentation's **slide 6** summarizes these actual historical results. Details are indexed in [Academic timeline and evidence](docs/PROJECT_HISTORY_AND_EVIDENCE.md).

**Evaluation boundaries:** The loss trend generally decreased with noisy spikes on a small dataset/batch size 1. Predictions captured some larger scene regions, vehicles and pedestrians, while thin boundaries and distant/small objects remained difficult. The course project **did not publish official Cityscapes mIoU, per-class IoU, or full pixel accuracy**. The limited three-image foreground/background visual proxy described in the report is **not** an official segmentation score. This is an academic perception pipeline, **not a production-ready vehicle system**.

## What is and is not runnable here

| Location | Contents | Reproducibility |
| --- | --- | --- |
| [**reconstruction/**](reconstruction/README.md) | **New, independent, smaller PyTorch FRRN-style** model, synthetic-data generator, training/inference, checkpoint and metric computation, tests | **Runnable** without Paperspace; three smoke tests passed locally, with separate CI workflow for fresh checkouts. |
| Historical root scripts and legacy prediction file | Partial copy of support scripts from the 2026 TensorFlow workflow | **Archive only; not standalone runnable.** They reference missing Paperspace directories and deprecated TensorFlow imports. |
| Historical FRRN-A source, original dataset subset and checkpoint | Used in completed graded CSC 590 experiment | **Not recovered in this GitHub export.** Exact original rerun is not currently possible. |

### Run the new reconstruction, not the historical TensorFlow scripts

Requires Python 3.10+ and CPU:

~~~bash
git clone https://github.com/kumi2001/FRRN-A-Semantic-Segmentation-Autonomous-Driving.git
cd FRRN-A-Semantic-Segmentation-Autonomous-Driving/reconstruction
python -m pip install -r requirements.txt
bash run_smoke.sh
~~~

The smoke test creates synthetic four-class images/masks, trains the new network for one epoch, saves a new checkpoint, generates a raw class-ID mask and executes tests. **Its synthetic results are not the original trained 17.7-million-parameter FRRN-A result, and the weights are not interchangeable.**

**Why original files are missing:** The original work ran inside a *paid cloud GPU workspace that I no longer have access to*. My course report and presentation provide historical evidence, but they are not source-code archives and cannot reconstitute the original weights. Original restricted Cityscapes imagery is not redistributed. See the full [reproducibility matrix](docs/REPRODUCIBILITY_STATUS.md).

## Academic materials and attribution

- **May 17, 2026 final report:** Kumar Abhinav, *Teaching Autonomous Vehicles to See the Road: Pixel-Level Scene Understanding with FRRN-A*, CSC 590, CSUDH (41 pages); detailed page index in [evidence map](docs/PROJECT_HISTORY_AND_EVIDENCE.md).
- **May 13, 2026 presentation:** *Pixel-Level Urban Scene Understanding for Autonomous Driving Using FRRN-A Semantic Segmentation* (8 slides).
- **Original project demonstration:** [Google Drive recording](https://drive.google.com/file/d/1W5sv3qTNaZtnd3INmvQyWFEA0zLoet2B/view); external viewing permissions may vary.
- **Academic assessment:** **CSC 590 grade A**. Highest class score is **author-reported**; ordinary transcripts establish grades, not necessarily class rank.
- **Research attribution:** FRRN was introduced by **Pohlen et al.** (*Full-Resolution Residual Networks for Semantic Segmentation in Street Scenes*, CVPR 2017). Cityscapes was introduced by **Cordts et al.** (*The Cityscapes Dataset for Semantic Urban Scene Understanding*, CVPR 2016). My course implementation adapted an existing architecture and legacy code; it did not invent FRRN. [Cityscapes dataset terms](https://www.cityscapes-dataset.com/license/) govern its images and labels.

**Summary:** **CSC 599** established the LiDAR/autonomous-perception research foundation; **CSC 590** continued and completed this broader direction through an implemented, graded **FRRN-A** image-segmentation experiment on a **paid GPU**. The academic project is complete. Its original cloud artifacts are partially inaccessible. The separate reconstruction is runnable, clearly labeled and **not passed off as the original model**.
