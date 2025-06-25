# 🔢 Handwritten Digits Classification with SVM

This project demonstrates how to classify handwritten digits (0–9) using **Support Vector Machines (SVM)** from `scikit-learn`. The model is trained on the classic **Digits Dataset**, which contains 8x8 grayscale images of handwritten numbers.  

It includes a full pipeline: from **data loading**, **preprocessing**, and **model training** to **evaluation** and **visualizing predictions** — all clearly structured and well-commented for learning and reuse.

---

## 📦 Features

- ✅ Loads the built-in `sklearn.datasets.load_digits` dataset (no download required)
- ✅ Flattens 8x8 image matrices into 1D feature vectors
- ✅ Splits data into training and testing sets (70/30)
- ✅ Trains a **Support Vector Machine (SVC)** classifier with adjustable `gamma`
- ✅ Evaluates the model using:
  - Accuracy score
  - Confusion matrix (with labels)
  - Classification report (precision, recall, F1-score)
- ✅ Displays prediction and actual label for a sample digit with its image

---

## 📊 Dataset Overview

The [Digits Dataset](https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html) contains:

- **1797 images** of handwritten digits (0–9)
- Each image is **8×8 pixels** (grayscale)
- Total of **64 features per sample**
- Labels are integers from **0 to 9**

---

## 🧠 Technologies Used

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- scikit-learn (SVM, datasets, train/test split, metrics)

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/handwritten-digit-svm.git
cd handwritten-digit-svm
