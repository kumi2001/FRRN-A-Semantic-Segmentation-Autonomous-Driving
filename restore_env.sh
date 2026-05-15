#!/bin/bash
echo "🚀 Setting up segmentation environment (using system Python)..."

# Update base system
sudo apt update -y
sudo apt install -y python3 python3-venv python3-dev python3-pip \
                    libgl1-mesa-glx libglib2.0-0

# Upgrade pip
python3 -m pip install --upgrade pip

# Install required AI libraries (TensorFlow 2.15 for Python 3.11)
python3 -m pip install tensorflow==2.15 tf-slim opencv-python-headless \
                         matplotlib scipy scikit-learn

echo "✅ Environment Ready!"
python3 --version
python3 - <<'EOF'
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Available GPUs:", tf.config.list_physical_devices('GPU'))
EOF


