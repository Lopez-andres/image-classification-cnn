# 🏛️ Wonders of the World Image Classification (CNN)

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Convolutional Neural Network for multi-class image classification (12 Wonders of the World)

**Universidad del Valle** - 2025-II

---

## 📋 Description

Convolutional Neural Network designed to classify images of the **12 Wonders of the World**.  

The project includes:
- Dataset download, extraction, and cleaning  
- Conversion to RGB and removal of corrupted images  
- Exploratory Data Analysis (EDA)  
- Data augmentation pipeline  
- CNN model training and optimization using callbacks  
- Performance evaluation with confusion matrix and classification report  

**Achieved Validation Accuracy:** **≈ 81%**

---

## 🚀 Quick Setup
```bash
# 1. Clone repository
git clone https://github.com/Lopez-andres/image-classification-cnn.git
cd image-classification-cnn

# 2. Create virtual environment (Anaconda)
conda create -n wonders_project python=3.10
conda activate wonders_project

# 3. Install dependencies
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn pillow gdown

# 4. Run Jupyter Notebook
jupyter notebook

# 5. Open project_2.ipynb and run all cells
```

---

## 🏗️ Model Architecture
```
Input (128 × 128 × 3 images)
 → Conv2D(32) + MaxPool
 → Conv2D(64) + MaxPool
 → Conv2D(128) + MaxPool
 → Conv2D(256) + MaxPool
 → Flatten
 → Dense(256, ReLU) + Dropout(0.5)
 → Dense(12, Softmax)
```

**Applied Techniques:**
- Data Augmentation (flip, brightness, contrast, saturation, hue)
- EarlyStopping
- ReduceLROnPlateau
- ModelCheckpoint
- One-hot encoding and image normalization

---

## 📊 Results

### Classification Metrics (per class)
- F1-scores range between 0.75 and 0.87
- **Best performing classes:**
  - Roman Colosseum (0.86)
  - Machu Picchu (0.84)
  - Pyramids of Giza (0.86)

### Confusion Matrix
Generated using seaborn heatmap for all 12 classes.

### Accuracy
- **Training Accuracy:** > 98%
- **Validation Accuracy:** ~81%

---

## 📁 Project Structure
```
Wonders-Classification/
├── LICENSE
├── README.md
├── project_2.ipynb             # Main notebook
└── wonders_dataset/            # Auto-downloaded dataset (created on first run)
```

---

## 🎨 Accessibility

Color palette used in plots is friendly for colorblind users:
- 🔵 Blue: Training curves
- 🟠 Orange: Validation curves
- 🟣 Purple/Yellow: Heatmaps

---

## 📧 Contact

**Andres Mauricio Peña:** andres.mauricio.pena@correounivalle.edu.co

---

<div align="center">

⭐ If this project helped you, consider giving it a star ⭐

**Universidad del Valle - 2025**

</div>
