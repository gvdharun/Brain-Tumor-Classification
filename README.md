# Brain Tumor Classification Project 🧠🔬

---
## 🚀 Project Overview

This repository contains an end-to-end solution for **brain tumor classification** from MRI images using deep learning. The models classify tumor types into four categories:  
- **glioma**  
- **meningioma**  
- **no tumor**  
- **pituitary tumor**

The project includes custom CNNs and transfer learning using pretrained models (ResNet50, EfficientNetB0), with data preprocessing, augmentation, training, evaluation, and deployment via Streamlit.

---

## 📝 Features

- ✅ Data loading & preprocessing (resizing, normalization)  
- ✅ Data augmentation (rotations, flips, zoom, brightness, shifts)  
- ✅ Handling class imbalance via weighted loss  
- ✅ Custom CNN architecture with dropout & batch normalization  
- ✅ Transfer learning using ResNet50 and EfficientNetB0 pretrained on ImageNet  
- ✅ Training with early stopping and best model checkpoint saving  
- ✅ Extensive evaluation: accuracy, loss plots, classification report, confusion matrix  
- ✅ Visualization of sample predictions with confidence  
- ✅ Interactive Streamlit app for user image upload and real-time prediction

---

## 📊 Models & Performance

| Model            | Params    | Test Accuracy | Macro F1-score | Notes                                  |
|------------------|-----------|---------------|----------------|----------------------------------------|
| Custom CNN       | ~13.2M    | 78%           | 0.76           | Solid baseline, some class imbalance issues |
| EfficientNetB0   | ~4.4M     | 66%           | 0.60           | Lightweight but lower recall on some classes |
| **ResNet50**     | ~24.1M    | **83%**       | **0.82**       | Best balance of accuracy & robustness |

---

## ⚙️ Setup Instructions

### 1. Clone the repository
git clone ``

### 2. Install dependencies
pip install `tensorflow matplotlib scikit-learn seaborn streamlit pillow numpy`

### 3. Prepare dataset
- Organize MRI images into `train`, `valid`, and `test` folders with subfolders per tumor class.
- Ensure images are in JPG/PNG/BMP format.

### 4. Run training scripts
- Train custom CNN or transfer learning models with included scripts.

### 5. Launch Streamlit web app
streamlit run `app.py`
Upload MRI images and get tumor predictions with confidence scores instantly.

---

## 📚 Usage Examples

### Data preprocessing & augmentation
- Image resizing to **224x224**  
- Normalization to pixel range **0–1**  
- Augmentations: rotation, flipping, zoom, brightness adjustments, shifts

### Model training with class weighting
- Addressed class imbalance through weighted loss using sklearn utility  
- Used EarlyStopping and ModelCheckpoint callbacks for efficient training

### Evaluation and visualization
- Accuracy and loss curves  
- Confusion matrix heatmap  
- Classification report (precision, recall, f1-score)  
- Sample prediction images with actual vs predicted labels and confidence  

### Streamlit app
- Upload brain MRI images interactively  
- View predicted tumor class and confidence percentage  
- Bar chart visualization of confidence across classes  

---

## 📂 Project Structure
├── data/ # MRI images organized by split and class
├── models/ # Trained model files (.h5)
├── notebook/ # training, evaluation notebooks
├── app.py # Streamlit application for deployment
└── README.md # Project overview and instructions

---

## 🎯 Conclusion

## Summary of Brain Tumor Classification Project 🧠

- ✅ **Effective Models Developed:** Custom CNN, ResNet50, and EfficientNetB0 pretrained architectures were implemented and evaluated for classifying brain MRI images into four tumor types.
- 🌟 **Best Performing Model:**  
  **ResNet50 with transfer learning and fine-tuning** stood out as the most accurate and reliable model, achieving **83% test accuracy** and strong balanced class performance (macro F1-score ~0.82).
- ⚖️ **Balanced Performance:**  
  The chosen model demonstrated **high recall for critical tumor classes** (glioma and pituitary), essential for clinical sensitivity, while maintaining good precision.
- 🕒 **Efficient Training Strategy:**  
  Starting with frozen pretrained layers and then fine-tuning top layers allowed efficient training, improved generalization, and robustness.
- 🔍 **Comprehensive Evaluation:**  
  Performance was validated using accuracy/loss curves, confusion matrices, classification reports, and visualization of exemplar predictions with confidence scores.
- 🚀 **Practical Deployment:**  
  An interactive Streamlit web application enables user-friendly MRI image upload and real-time tumor classification with confidence visualization.
