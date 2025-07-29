# Skin Lesion Classification with EfficientNet-B3

This project classifies skin lesions using a fine-tuned EfficientNet-B3 model. The pipeline includes custom augmentations, balanced sampling, and post-hoc explainability through Grad-CAM and SHAP.

## Features

- EfficientNet-B3 pretrained on ImageNet and fine-tuned for binary classification
- Advanced augmentations using albumentations
- Handling class imbalance with WeightedRandomSampler
- Explainability with:
  - **Grad-CAM** for visual localization
  - **SHAP** for feature-level explanation

## Model Architecture

- EfficientNet-B3 base model
- Only top layers and classifier are fine-tuned

## Evaluation Metrics

- Accuracy, Precision, Recall, F1-score
- Confusion Matrix
- Classification Report

## Tech Stack

- Python
- PyTorch
- Torchvision
- Albumentations
- Grad-CAM
- SHAP

## Visual Explainability

- Grad-CAM overlays highlight class-specific regions on images
- SHAP plots offer feature attribution insights
