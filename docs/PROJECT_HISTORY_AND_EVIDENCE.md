# CSC 599 → CSC 590: research history and evidence

## Academic sequence

**Spring 2025 — CSC 599 (LiDAR/autonomous perception & mapping).** According to the author and his current portfolio resume, this earlier master's thesis-track work explored LiDAR point-cloud sensing, object detection, mapping, and navigation-related decisions, within a three-faculty research framework.

**Spring 2026 — CSC 590 (FRRN-A semantic segmentation).** This was the continuation/full completion of the broader autonomous-perception research direction as a master's *project-track* implementation focused on camera-image semantic segmentation. The FRRN-A experiment ran on a **paid Paperspace Gradient GPU workspace** under the instruction of **Dr. Bin Tang**.

**Important:** A continuing research topic does not mean the FRRN-A network itself used live LiDAR inputs, fused sensors, or operated an autonomous vehicle. It is a camera-image segmentation model.

## Completed May 2026 academic deliverables

| Evidence | Exact content |
| --- | --- |
| **41-page CSC 590 final report**, May 17, 2026 | *Teaching Autonomous Vehicles to See the Road: Pixel-Level Scene Understanding with FRRN-A*. Includes setup, training architecture, diagrams, experimental observations, error analysis and limitations. |
| **8-slide CSC 590 presentation**, May 13, 2026 | *Pixel-Level Urban Scene Understanding for Autonomous Driving Using FRRN-A Semantic Segmentation*. Shows research motivation, FRRN-A concept, dataset/training setup and original results. |
| [**Original project walkthrough recording**](https://drive.google.com/file/d/1W5sv3qTNaZtnd3INmvQyWFEA0zLoet2B/view) | Original environment/demonstration. The external Drive link may require permission to open; it is not the original source archive. |
| **CSC 590 course assessment** | Author's portfolio resume lists **grade A**. The author additionally reports **highest score in class**; ordinary transcripts would substantiate the grade but *not necessarily* class rank. |

## How to inspect the report

- **pp. 1–2:** title page, course instructor, abstract, completed scope
- **pp. 15–16:** Cityscapes/Darmstadt-style urban driving subset and dataset preparation
- **pp. 17–21:** implementation, FRRN-A model, GPU training and legacy-code adaptations
- **pp. 22–23:** forms of evaluation, training loss across 100 epochs, noisy spikes and declining overall trend
- **p. 24:** actual paired input, ground-truth segmentation and predicted mask comparisons
- **pp. 25–27:** result analysis and limitations
- **pp. 28–29:** conclusions and future work
- **pp. 31–33:** demonstration URL, Paperspace project directory layout and historical commands
- **pp. 34–35:** additional historical segmentation-mask evidence
- **pp. 36–38:** engineering constraints and limited visual proxy versus full mIoU
- **pp. 39–41:** recording outline and reproducibility audit

## How to inspect the presentation

- **Slide 4:** FRRN-A, full-resolution and pooled-stream explanation
- **Slide 5:** 320 × 640 crops, batch size 1, 100 epochs, ~17.7M parameters and checkpoints/masks
- **Slide 6:** **historical** loss chart and input / ground truth / FRRN-A prediction
- **Slide 7:** interpretation and future work

## Original engineering milestones (reported in the final report)

1. Adapt legacy TensorFlow code to work with an updated GPU runtime using `tf.compat.v1` and `tf_slim`.
2. Address missing dependencies, CUDA/TensorFlow compatibility and memory issues.
3. Train a 17,747,620-parameter model for **100 epochs** on **85 training + 15 validation** pairs.
4. Save weights and output prediction masks from the paid Paperspace session.
5. Examine qualitative results and identify limitations with small objects, boundaries and absent formal mIoU.

The *historical* checkpoint files and full source code were **not** recovered from the inaccessible paid cloud workspace. The original work was completed; the surviving public repository is only a *partial archive*. See [reproducibility matrix](REPRODUCIBILITY_STATUS.md).
