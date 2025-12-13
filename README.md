# Deep Learning Image Classification (Capstone Project)

## 📌 Project Overview
This capstone project focuses on building, training, and evaluating a deep learning–based
image classification system using **PyTorch**. A Convolutional Neural Network (CNN) is
implemented to classify images from a publicly available dataset. The project demonstrates
the complete deep learning workflow, including data preprocessing, model training,
hyperparameter tuning, evaluation, and result interpretation.

This project was developed as part of the **Artificial Intelligence & Machine Learning**
program at **Fanshawe College**.

---

## 🎯 Project Objectives
- Understand and apply core deep learning concepts
- Implement and train CNNs using PyTorch
- Apply deep learning techniques to a real-world computer vision task
- Evaluate model performance using standard classification metrics
- Analyze and interpret experimental results
- Demonstrate proficiency in PyTorch
- Design and execute an end-to-end deep learning project

---

## 📊 Dataset
- **Dataset:** CIFAR-10 (Torchvision)
- **Number of Classes:** 10
- **Image Type:** RGB images (32×32)
- **Total Images:** 60,000
- **Split:** Training, Validation, and Test sets

---

## 🧠 Model Architecture
The image classification model is based on a Convolutional Neural Network (CNN) and includes:
- Multiple convolutional layers with **ReLU activation**
- **Batch Normalization** for stable training
- **Max Pooling** layers for spatial downsampling
- **Dropout** layers to reduce overfitting
- Fully connected layers for final classification

---

## ⚙️ Training Details
- **Framework:** PyTorch
- **Loss Function:** CrossEntropyLoss
- **Optimizer:** AdamW
- **Learning Rate:** 0.001
- **Batch Size:** 128
- **Regularization:** Dropout & weight decay
- **Early Stopping:** Applied to prevent overfitting

---

## 📈 Test Results (Best Model)

The final model was evaluated on a held-out test dataset to measure its generalization
performance on unseen data.

=== TEST RESULTS (Best Model) ===
Loss: 0.4214
Accuracy: 0.8550
Precision (macro): 0.8545
Recall (macro): 0.8553
F1-score (macro): 0.8532


### 🔍 Interpretation
- The model achieves **85.5% accuracy**, indicating strong classification performance.
- Balanced **precision, recall, and F1-score** show that the model performs consistently
  across all classes.
- Low test loss suggests good generalization with minimal overfitting.

---

## 📊 Evaluation & Visualizations
- Training vs Validation Loss Curve
- Training vs Validation Accuracy Curve
- Confusion Matrix
- Visualization of misclassified samples

These visualizations help analyze model behavior and identify areas for improvement.

---

## 🚀 How to Run the Project

```bash
git clone https://github.com/yourusername/deep-learning-image-classification-capstone.git
cd deep-learning-image-classification-capstone
pip install -r requirements.txt
python src/train.py
