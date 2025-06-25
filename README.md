# 🌸 Iris Dataset - EDA & K-Nearest Neighbors Classification

This project involves an end-to-end data science workflow using the classic **Iris flower dataset**. It includes data cleaning, exploratory data analysis (EDA), data visualization, outlier handling, feature encoding, and applying **K-Nearest Neighbors (KNN)** classification.

---

## 📁 Project Structure
---

## 📊 Key Features

### ✅ Data Preprocessing
- Renamed columns to `Sepal_Length`, `Sepal_Width`, etc.
- Checked and confirmed no missing values
- Handled outliers using the IQR method

### 📈 Visualizations (Seaborn & Matplotlib)
- Boxplots
- Violin plots
- Scatter plots
- Displots
- Saved figures as PNG for non-interactive execution

### 🧠 Machine Learning with `scikit-learn`
- Encoded target labels with `LabelEncoder`
- Scaled features using `StandardScaler`
- Trained a **KNN classifier**
- Evaluated performance using:
  - Accuracy score
  - Classification report
  - Confusion matrix

### 🧪 Final KNN Accuracy: ~93%

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/tripathi-22/iris-knn-classifier.git
   cd iris-knn-classifier

## Install dependencies
pip install -r requirements.txt
