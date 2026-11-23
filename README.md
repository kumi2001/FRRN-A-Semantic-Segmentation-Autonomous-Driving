# Teaching Autonomous Vehicles to See the Road
## Pixel-Level Urban Scene Understanding with FRRN-A

**Kumar Abhinav | M.S. Computer Science, California State University, Dominguez Hills | CSC 590, May 2026**

Faculty guidance: **Dr. Bin Tang**

### Project at a glance

This graduate project adapted and trained a legacy **Full-Resolution Residual Network (FRRN-A)** for semantic segmentation of urban driving images. Unlike object detection, which outputs bounding boxes, semantic segmentation assigns a scene category to every pixel.

The main engineering contribution was getting an older TensorFlow segmentation pipeline to run in a newer Paperspace GPU environment, completing model training, saving checkpoints and examining prediction masks.

| Item | Experiment described in the final report |
| --- | --- |
| Model | FRRN-A, **17,747,620 trainable parameters** |
| Data | Cityscapes/Darmstadt-style **course-project subset** |
| Data split | **85** image-mask pairs for training; **15** for validation |
| Crop size | **320 × 640** |
| Batch size | **1** (GPU memory constraint) |
| Training | **100 epochs**, Paperspace Gradient |
| Outputs | Checkpoint files and qualitative segmentation masks |

### What I worked on

- Prepared paired urban-scene images and segmentation masks.
- Adapted TensorFlow 1.x-style code to newer library environments, including `tf.compat.v1`, `tf_slim` in place of `tensorflow.contrib.slim`, and batch-normalization compatibility adjustments.
- Troubleshot GPU environment and dependency issues and trained the model for 100 epochs.
- Inspected training-loss behavior, saved checkpoints, and compared selected colorized predictions with ground-truth masks.
- Wrote a **41-page technical report** documenting design decisions, implementation, evidence and limitations, and prepared a presentation and recorded walkthrough.

### Findings and limitations

The trained pipeline produced masks representing major road-like/background regions and some vehicles and pedestrians. The loss generally decreased over training, with fluctuations likely affected by the small data subset and batch size of one. Small/distant objects and thin boundaries remained challenging.

**Important:** This is a course-scale implementation and qualitative evaluation, **not** a production autonomous-driving system or an official Cityscapes benchmark submission. The project did **not** complete full validation-set mean intersection-over-union (mIoU), per-class IoU, or pixel-accuracy scoring. A three-example foreground/background *visualization proxy* in the report is **not** a semantic-segmentation accuracy score.

The original FRRN architecture was introduced by Pohlen et al., *Full-Resolution Residual Networks for Semantic Segmentation in Street Scenes* (CVPR 2017). This project adapted existing research code; it did not invent the FRRN architecture.

### Public repository status — please read before running

**This GitHub repository is an incomplete public snapshot of the original Paperspace project environment.** At the time of this README update, the repository includes a small set of shell scripts, a legacy video prediction script, requirements and this documentation. It **does not yet include** the full `docker/` implementation, `models/FRRN.py`, the custom prediction script, training logs, checkpoint files or generated mask images described in the final report.

As a result:

- `run_project.sh` refers to `/notebooks/docker/main.py`, which is **not in this repository**.
- The original prediction command refers to `docker/predict_frrna_custom.py`, which is **not in this repository**.
- The checked-in `predict_video.py` is **legacy code** and imports deprecated TensorFlow components and additional files that are not present here.
- `restore_env.sh` is environment-specific and may require privileges, GPU drivers and software versions not available on a different machine.

**Do not expect a fresh clone to train or predict successfully.** The original Paperspace environment, missing source files, appropriate dataset access and checkpoints are needed for reproduction. I intend to restore publishable source and sample output artifacts when they are available and their reuse rights have been checked.

### Project artifacts

- [Five-minute project demo recording (Google Drive; access may be required)](https://drive.google.com/file/d/1W5sv3qTNaZtnd3INmvQyWFEA0zLoet2B/view)
- [My portfolio](https://kumarabhinavtech.com)
- Final report: *Teaching Autonomous Vehicles to See the Road: Pixel-Level Scene Understanding with FRRN-A* (CSC 590, May 17, 2026). The report includes the model's training-loss plot, sample input/ground-truth/prediction comparisons, and explicit evaluation limitations.

### Planned next steps

1. Recover and review the original source tree, scripts and a small permissible test sample.
2. Reconcile the documented environment and repository execution commands.
3. Add genuine experiment plots and prediction examples.
4. Add raw label-map outputs and implement official full-validation-set metrics where the source data permits.
5. Validate fresh-environment setup instructions.

---

*Academic project completed in May 2026. Technical claims above describe the completed project; the repository status describes what is currently published.*
