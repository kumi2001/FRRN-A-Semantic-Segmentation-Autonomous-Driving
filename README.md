# Teaching Autonomous Vehicles to See the Road: Pixel-Level Scene Understanding with FRRN-A

## Overview
This repository contains the code and results for my Master's final project on semantic segmentation for autonomous driving scene understanding.

The project uses FRRN-A, a Full-Resolution Residual Network, to classify urban driving scenes at the pixel level.

## Dataset
The project uses Cityscapes/Darmstadt urban driving images with pixel-level ground-truth segmentation masks.

Input image shape: H x W x 3, where H is height, W is width, and 3 represents RGB color channels.

Training crop size: 320 x 640.

## Model
The model used is FRRN-A, or Full-Resolution Residual Network-A.

FRRN-A preserves full-resolution spatial detail while also learning deep scene context. This is important for road boundaries, vehicles, pedestrians, and sidewalks.

## Implementation Summary
Main implementation work included:
- Setting up the Cityscapes/Darmstadt data
- Adapting legacy TensorFlow code
- Replacing deprecated TensorFlow contrib components with tf_slim
- Fixing batch normalization compatibility issues
- Training FRRN-A for 100 epochs
- Saving checkpoints
- Generating prediction masks

## Training Setup
Model: FRRN-A  
Dataset: Cityscapes/Darmstadt  
Input crop size: 320 x 640  
Batch size: 1  
Training epochs: 100  
Trainable parameters: approximately 17.7 million  
Framework: TensorFlow compatibility mode  

## Results
The model successfully trained for 100 epochs and generated segmentation prediction masks.

The training loss contains spikes because the dataset subset is small and batch size is 1. With batch size 1, difficult images can temporarily increase the loss. However, the model recovered after spikes and the overall trend decreased.

Qualitative prediction outputs show that the model learned major scene regions such as road-like areas, vehicles, pedestrians, and background.

## How to Run
Restore environment:
cd /notebooks
./restore_env.sh

Train model:
cd /notebooks
./run_project.sh

Generate predictions:
cd /notebooks/docker
python3 predict_frrna_custom.py

## Author
Kumar Abhinav  
Master's Final Project  
Computer Science  
California State University, Dominguez Hills
