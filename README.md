# glaucoma-onset-prediction-gan-cnn
Multi-modal deep learning pipeline combining CNNs and GAN-driven generative augmentation on retinal fundus imagery for early glaucoma prediction in low-resource settings.
# Neurophysiology-Image Fusion with Generative Augmentation for Early Glaucoma Onset Prediction in Low-Resource Settings

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Glaucoma ranks among the leading global causes of irreversible blindness, progressing asymptomatically until permanent optic nerve damage has occurred. In low-resource healthcare settings, early diagnosis is hindered by a lack of specialized eye specialists and costly imaging equipment (e.g., Optical Coherence Tomography).

This repository contains an automated deep learning framework designed to screen for early-stage glaucoma using accessible **retinal fundus photography**. To tackle severe clinical class imbalances and limited sample availability, the system pairs **Generative Adversarial Networks (GANs)** for synthetic data augmentation with **Convolutional Neural Networks (CNNs)** trained on key neurophysiological ocular biomarkers, complemented by Grad-CAM visual explainability.

---

## Pipeline Architecture

```text
       +-------------------------------+
       | Retinal Fundus Image Input    |
       +---------------+---------------+
                       |
                       v
       +-------------------------------+
       | Preprocessing & Enhancement   |
       | (OpenCV, Adaptive CLAHE, etc.)|
       +---------------+---------------+
                       |
        +--------------+--------------+
        |                             |
        v                             v
+---------------+             +---------------+
| Real Enhanced |             | GAN Generator |
|    Dataset    |             |  (Synthetic)  |
+-------+-------+             +-------+-------+
        |                             |
        +--------------+--------------+
                       |
                       v
       +-------------------------------+
       | Balanced Retinal Fundus Pool  |
       +---------------+---------------+
                       |
                       v
       +-------------------------------+
       | Deep CNN Feature Extractor    |
       | - Optic Disc & Cup Analysis   |
       | - RNFL Defect Pattern Mining  |
       +---------------+---------------+
                       |
        +--------------+--------------+
        |                             |
        v                             v
+---------------+             +---------------+
| Binary Risk   |             | Grad-CAM      |
| Diagnostic    |             | Visual Saliency
+---------------+             +---------------+






NEUROPHYSIOLOGY-IMAGE-FUSION-GLAUCOMA/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── README.md
│
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_baseline_cnn.ipynb
│   ├── 03_gan_augmentation.ipynb
│   ├── 04_cnn_gan_training.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_gradcam.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── cnn_model.py
│   ├── gan_model.py
│   ├── training.py
│   ├── evaluation.py
│   └── gradcam.py
│
├── models/
│   └── README.md
│
├── results/
│   ├── figures/
│   ├── metrics/
│   └── gradcam/
│
├── documentation/
│   ├── methodology.md
│   └── results.md
│
└── presentation/
    └── project_presentation.pdf



git clone [https://github.com/yazhini-josy-victor/glaucoma-onset-prediction-gan-cnn.git](https://github.com/yazhini-josy-victor/glaucoma-onset-prediction-gan-cnn.git)
cd glaucoma-onset-prediction-gan-cnn

pip install -r requirements.txt

# 1. Preprocess raw fundus images
python src/preprocessing.py

# 2. Train GAN augmentation pipeline
python src/gan_model.py

# 3. Train the CNN classifier on augmented data
python src/training.py

# 4. Evaluate performance and generate metrics
python src/evaluation.py

# 5. Generate Grad-CAM explainability heatmaps
python src/gradcam.py

