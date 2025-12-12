# Deep Learning Image Classification (Capstone Project)

## 📌 Overview
This project implements a deep learning–based image classification system using
Convolutional Neural Networks (CNNs) in PyTorch. The goal is to classify images
into multiple categories using a publicly available dataset.

## 🎯 Objectives
- Build and train a CNN for image classification
- Apply data preprocessing and augmentation techniques
- Tune hyperparameters for optimal performance
- Evaluate the model using accuracy, precision, recall, and F1-score

## 📊 Dataset
- **Dataset:** CIFAR-10 (Torchvision)
- **Classes:** 10
- **Images:** 60,000 RGB images (32×32)

## 🧠 Model Architecture
- Convolutional layers with Batch Normalization and ReLU
- Max Pooling layers
- Dropout for regularization
- Fully connected classifier

## ⚙️ Training Details
- Optimizer: AdamW
- Loss Function: CrossEntropyLoss
- Batch Size: 128
- Learning Rate: 0.001
- Early Stopping applied

## 📈 Results
| Metric | Value |
|------|------|
| Accuracy | ~80% |
| Precision | ~0.80 |
| Recall | ~0.80 |
| F1-score | ~0.80 |

## 📷 Visualizations
- Training vs Validation Loss
- Training vs Validation Accuracy
- Confusion Matrix
- Misclassified Images

## 🚀 How to Run
```bash
git clone https://github.com/yourusername/deep-learning-image-classification-capstone.git
cd deep-learning-image-classification-capstone
pip install -r requirements.txt
python src/train.py
