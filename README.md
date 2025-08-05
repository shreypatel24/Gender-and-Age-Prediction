# Gender-and-Age-Prediction

A deep learning project for simultaneous **age estimation** and **gender classification** from face images using the [UTKFace dataset](https://susanqq.github.io/UTKFace/). This solution leverages **transfer learning** (VGG16) and the **Keras Functional API** to build a multi-output model.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Setup](#setup)
- [Usage](#usage)
- [Results](#results)
- [Improvements & Next Steps](#improvements--next-steps)

---

## Overview

This repository implements a multi-task deep learning model that predicts both age and gender from face images. The model uses a pre-trained VGG16 backbone and custom Dense branches for each task.

- **Gender Prediction:** Classifies as Male (0) or Female (1)
- **Age Prediction:** Estimates age as a regression task

## Features

- Uses **Keras Functional API** for flexible model design
- **Transfer Learning** with frozen VGG16 convolutional base
- **Multi-output architecture**: simultaneous regression (age) and classification (gender)
- **Data augmentation** for improved generalization
- **Training/validation split** and performance visualization

## Architecture

- **Backbone:** VGG16 (pre-trained on ImageNet, convolutional layers frozen)
- **Heads:** Two parallel Dense branches for age (regression) and gender (binary classification)
- **Losses:** MAE (Mean Absolute Error) for age, Binary Crossentropy for gender

```
Input Image (200x200) → VGG16 (Conv Base) → Flatten → 
    ├─ Dense → Dense → Dense → Age Output (linear)
    └─ Dense → Dense → Dense → Gender Output (sigmoid)
```

## Dataset

- [UTKFace Dataset](https://susanqq.github.io/UTKFace/)
- 23,000+ face images with labels: age, gender, ethnicity
- In this project: **5,000 images for training**, **1,208 for testing** (due to GPU constraints)

## Setup

1. **Install dependencies**
   ```bash
   pip install numpy pandas matplotlib seaborn tensorflow keras
   ```

2. **Download UTKFace dataset**
   - Place the dataset under `/kaggle/input/utkface-new/UTKFace` or update the path in the script.

3. **Run the notebook or script**
   - All major steps (EDA, data prep, augmentation, model building, training, evaluation) are included.

## Usage

- **Data Preparation:** Extracts age, gender, and file paths; splits into train/test sets.
- **EDA:** Visualizes age and gender distributions.
- **Augmentation:** Implements random rotations, shifts, shear, zoom, and flips.
- **Custom Generators:** Handles multi-output batches for TensorFlow/Keras.
- **Model Building:** Functional API with frozen VGG16 and two Dense branches.
- **Training:** Optimizer (`adam`), losses (`mae`, `binary_crossentropy`), metrics (`mae`, `accuracy`).
- **Evaluation:** Plots loss curves and metrics for both outputs.

## Results

- **Age Prediction (MAE):** 8.14 (validation)
- **Gender Classification Accuracy:** 84.78% (validation)
- **Epochs:** 20

Training and validation curves indicate effective multi-task learning and robust generalization, despite the limited training size.

## Improvements & Next Steps

- Increase training data size (full dataset instead of 5k samples)
- Experiment with other pre-trained backbones (e.g., ResNet, EfficientNet)
- Apply regularization (Dropout, L2) to reduce potential overfitting
- Tune hyperparameters and train for more epochs
- Further visualize and analyze misclassifications

---

## References

- [UTKFace Dataset](https://susanqq.github.io/UTKFace/)
- [Keras Documentation](https://keras.io/api/)
- [VGG16 Paper](https://arxiv.org/abs/1409.1556)

## Author

- [shreypatel24](https://github.com/shreypatel24)
