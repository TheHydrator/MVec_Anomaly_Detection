# Zero-Shot Industrial Anomaly Detection via Deep Feature Modeling

**Industry 4.0 Quality Control System**  
Detecting unseen manufacturing defects using only normal (defect-free) samples.

## Overview

Global manufacturing loses **$1.3 trillion annually** due to quality failures. Traditional supervised defect detection fails in real-world scenarios because defects are extremely rare (<1% of production) and new defect types constantly appear.

This project implements **zero-shot anomaly detection** by learning the "distribution of perfection" from normal samples only. We extract rich semantic features using a pretrained ResNet-18, reduce dimensionality with PCA, and apply multiple one-class anomaly detection algorithms.

**Key Highlights:**
- Trained **exclusively on defect-free images**
- Handles **novel/unseen defects** at test time
- Comprehensive comparison of 4 anomaly detection methods
- Strong performance on the challenging MVTec AD benchmark

## Features

- **Deep Feature Extraction**: ResNet-18 (ImageNet pretrained) → 512-dim embeddings
- **Dimensionality Reduction**: PCA (95% variance retained) + standardization
- **Multiple Models**:
  - Gaussian Mixture Model + Mahalanobis Distance
  - One-Class SVM (RBF kernel)
  - Isolation Forest
  - DBSCAN (converted to anomaly scorer)
- **Robust Evaluation**: AUC-ROC and Average Precision (handles class imbalance)
- **Production Ready**: Saved pipelines (`.pkl`) for each model

**Challenging Categories**: `Grid` and `Screw` (subtle defects + complex textures/patterns), and comparative analysis shows that DBscan outperforms other models in this case.

Full results available in [`Project Report.docx`](Project%20Report.docx).

## Dataset

**MVTec Anomaly Detection (MVTec AD)**  
- 15 industrial product categories (5 textures + 10 objects)
- Training: ~5,354 defect-free images
- Testing: ~1,725 images (normal + defective)
- Categories include: bottle, cable, capsule, carpet, grid, hazelnut, leather, metal nut, pill, screw, tile, toothbrush, transistor, wood, zipper

## Methodology

### 1. Feature Extraction
- Resize images to 224×224
- Pretrained **ResNet-18** (average pooling layer → 512 features)
- Grayscale images converted to RGB

### 2. Preprocessing
- Standard scaling
- PCA (category-specific components, 95% variance)

### 3. Anomaly Detection Models
See `Project Report.docx` for detailed methodology, hyperparameters, and mathematical formulations (Mahalanobis distance, etc.).

 
