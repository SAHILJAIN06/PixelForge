# PixelForge
Synthetic Data Generation using GANs for Privacy-Preserving and Data Augmentation (C-NMC Leukemia Dataset)
# PixelForge 🧬
Synthetic Data Generation using GANs for Privacy-Preserving and Data Augmentation  

## 📌 Overview
PixelForge is a deep learning project that leverages Convolutional Neural Networks (CNNs) and Generative Adversarial Networks (GANs) for medical image classification and augmentation.  
The project is tested on the **C-NMC Leukemia Dataset (ALL vs Normal cells)**.

### 🔹 Stage 1
- Built a baseline CNN classifier.
- Achieved **84% accuracy** and **0.90 ROC-AUC**.
- High recall for Leukemia cases (95%) – critical for medical diagnosis.

### 🔹 Stage 2
- Use GANs to generate synthetic cell images.
- Retrain CNN with augmented dataset.
- Evaluate performance improvement.

---

## 🚀 How to Run
### 1. Install dependencies
```bash
pip install -r requirements.txt

📊 Results (Stage 1)

Accuracy: 84%
ROC-AUC: 0.90
Classification Report + Confusion Matrix included in results/.
