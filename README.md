<div align="center">

# 🚀 Ensemble Learning — Student Performance Intelligence

### Classification + Regression using Modern Ensemble Machine Learning

<p align="center">
  <img src="assets/workflow.png" alt="Ensemble Learning Workflow" width="100%">
</p>

<p align="center">
  <b>Bagging</b> •
  <b>AdaBoost</b> •
  <b>Gradient Boosting</b> •
  <b>LightGBM</b> •
  <b>XGBoost</b> •
  <b>Voting</b> •
  <b>Stacking</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white">
  <img src="https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy&logoColor=white">
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikit-learn&logoColor=white">
  <img src="https://img.shields.io/badge/LightGBM-Boosting-9ACD32">
  <img src="https://img.shields.io/badge/XGBoost-Boosting-EC4E20">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white">
</p>

</div>

---

# 📌 Project Overview

This project is a complete **Ensemble Learning Machine Learning system** designed to solve two different predictive tasks from student learning and activity data.

### 🎯 Two Machine Learning Tasks

| Task              | Target              | Objective                                      |
| ----------------- | ------------------- | ---------------------------------------------- |
| 🎯 Classification | `completion_status` | Predict whether a student completes the course |
| 📈 Regression     | `final_score`       | Predict the student's final score              |

The project compares multiple ensemble learning strategies and evaluates their performance using appropriate classification and regression metrics.

---

# 🏆 Final Model Performance

## 🎯 Best Classification Model

### 🥇 Stacking Classifier

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **74.42%** |
| Precision | **0.6867** |
| Recall    | **0.5846** |
| F1 Score  | **0.6316** |
| ROC-AUC   | **0.7940** |

> **Stacking achieved the strongest overall classification performance on the test set.**

---

## 📈 Best Regression Model

### 🥇 Stacking Regressor

| Metric |      Score |
| ------ | ---------: |
| MAE    | **7.8049** |
| RMSE   | **9.7166** |
| R²     | **0.4947** |

> **Stacking achieved the lowest prediction error and highest R² among the evaluated regression models.**

---

# 🎯 Project Objectives

The project focuses on answering the following questions:

* Can ensemble models outperform a single Decision Tree?
* Which boosting algorithm performs best?
* How does Bagging reduce model variance?
* How does AdaBoost improve weak learners?
* How does Gradient Boosting reduce prediction errors?
* How do LightGBM and XGBoost perform?
* Does Hard Voting outperform Soft Voting?
* Can Stacking improve predictive performance?
* Which model is best for classification?
* Which model is best for regression?

---

# 🗂️ Dataset

The project uses:

```text
dataset.5.csv
```

### Dataset Snapshot

| Property              |               Value |
| --------------------- | ------------------: |
| Total Rows            |           **5,200** |
| Total Columns         |              **19** |
| Classification Target | `completion_status` |
| Regression Target     |       `final_score` |
| Train/Test Split      |       **80% / 20%** |
| Training Samples      |           **4,160** |
| Testing Samples       |           **1,040** |
| Processed Features    |              **30** |

---

# 🔍 Important Features

### 👤 Student Profile

```text
age
country_region
device_type
education_background
```

### 📚 Course Information

```text
course_level
course_category
course_start_date
week_of_year
```

### 🖥️ Learning Activity

```text
sessions
time_spent_hours
videos_watched
quiz_attempts
assignments_submitted
forum_posts
```

### 📊 Performance Features

```text
avg_quiz_score
attendance_rate
```

### 🎯 Target Variables

```text
completion_status
final_score
```

---

# 📊 Data Understanding

The classification target contains two classes:

| Completion Status | Count |  Share |
| ----------------- | ----: | -----: |
| 0                 | 3,248 | 62.45% |
| 1                 | 1,952 | 37.55% |

### Final Score Statistics

| Statistic          |  Value |
| ------------------ | -----: |
| Mean               |  74.82 |
| Standard Deviation |  13.53 |
| Minimum            |  35.20 |
| Median             |  74.10 |
| Maximum            | 100.00 |

---

# 🧹 Data Preprocessing

The project uses a reusable:

```text
ColumnTransformer + Pipeline
```

approach.

### 🔢 Numerical Features

```text
Missing Values
      ↓
Median Imputation
      ↓
StandardScaler
```

### 🔤 Categorical Features

```text
Missing Values
      ↓
Most Frequent Imputation
      ↓
OneHotEncoder
      ↓
Unknown Categories Ignored
```

After preprocessing:

```text
Training samples : 4,160
Testing samples  : 1,040
Processed features: 30
```

---

# 🧠 Machine Learning Architecture

<p align="center">
  <img src="assets/workflow.png" alt="Machine Learning Architecture" width="100%">
</p>

The project explores both **parallel ensemble learning** and **sequential ensemble learning** techniques.

---

# 1️⃣ Bagging

**Bagging — Bootstrap Aggregating**

Bagging trains multiple models independently on bootstrap samples and combines their predictions.

### Implemented

```text
Bagging Classifier
Bagging Regressor
```

### Main Benefit

> Reduces variance and improves model stability.

---

# 2️⃣ AdaBoost

AdaBoost builds weak learners sequentially.

Later learners focus more on observations that were difficult for previous learners.

### Implemented

```text
AdaBoost Classifier
AdaBoost Regressor
```

### Main Benefit

> Improves weak learners by focusing on difficult observations.

---

# 3️⃣ Gradient Boosting

Gradient Boosting builds models sequentially to reduce the errors made by previous models.

### Implemented

```text
GradientBoostingClassifier
GradientBoostingRegressor
```

The project also explores:

```text
Learning Rate → 0.01, 0.05, 0.10

Estimators → 50, 100, 200
```

---

# 4️⃣ LightGBM

**Light Gradient Boosting Machine**

LightGBM is an efficient gradient boosting framework designed for fast and scalable machine learning.

### Implemented

```text
LGBMClassifier
LGBMRegressor
```

---

# 5️⃣ XGBoost

**Extreme Gradient Boosting**

XGBoost is a powerful and regularized gradient boosting algorithm.

### Implemented

```text
XGBClassifier
XGBRegressor
```

---

# 6️⃣ Voting Ensemble

Voting combines predictions from multiple classifiers.

### 🗳️ Hard Voting

Uses the majority class prediction.

```text
Model 1 → Class 1
Model 2 → Class 0
Model 3 → Class 1

Final → Class 1
```

### 🗳️ Soft Voting

Uses predicted probabilities from different classifiers.

```text
Probability
      ↓
Average
      ↓
Final Prediction
```

---

# 7️⃣ Stacking

Stacking combines multiple base models and uses a **meta-learner** to learn how their predictions should be combined.

### Classification

```text
Base Models
    ↓
Predictions
    ↓
Meta Learner
    ↓
Final Classification
```

### Regression

```text
Base Models
    ↓
Predictions
    ↓
Meta Learner
    ↓
Final Score Prediction
```

---

# 📊 Classification Model Comparison

<p align="center">
  <img src="assets/classification_model_comparison.png" alt="Classification Model Comparison" width="100%">
</p>

| Model                |   Accuracy |  Precision |     Recall |         F1 |    ROC-AUC |
| -------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| Single Decision Tree |     0.6298 |     0.5062 |     0.5256 |     0.5157 |     0.6090 |
| Bagging              |     0.7221 |     0.6472 |     0.5692 |     0.6057 |     0.7739 |
| AdaBoost             |     0.7365 |     0.6883 |     0.5436 |     0.6074 |     0.7850 |
| Gradient Boosting    |     0.7385 |     0.6832 |     0.5641 |     0.6180 |     0.7896 |
| LightGBM             |     0.7288 |     0.6656 |     0.5564 |     0.6061 |     0.7805 |
| XGBoost              |     0.7356 |     0.6727 |     0.5744 |     0.6196 |     0.7875 |
| Hard Voting          |     0.7413 |     0.6921 |     0.5590 |     0.6184 |          — |
| Soft Voting          |     0.7346 |     0.6770 |     0.5590 |     0.6124 |     0.7891 |
| 🏆 **Stacking**      | **0.7442** | **0.6867** | **0.5846** | **0.6316** | **0.7940** |

---

# 🥇 Classification Winner

## Stacking Classifier

```text
Accuracy  → 74.42%
F1 Score  → 0.6316
ROC-AUC   → 0.7940
```

Stacking produced the strongest F1 and ROC-AUC among the evaluated classification models.

---

# 📈 Regression Performance

<p align="center">
  <img src="assets/regression_rmse_comparison.png" alt="Regression RMSE Comparison" width="100%">
</p>

| Model             |        MAE |       RMSE |         R² |
| ----------------- | ---------: | ---------: | ---------: |
| AdaBoost          |     8.5452 |    10.5174 |     0.4080 |
| Gradient Boosting |     7.9645 |     9.9190 |     0.4734 |
| LightGBM          |     7.9309 |     9.8756 |     0.4780 |
| XGBoost           |     7.9205 |     9.8749 |     0.4781 |
| 🏆 **Stacking**   | **7.8049** | **9.7166** | **0.4947** |

---

# 📊 Regression R² Comparison

<p align="center">
  <img src="assets/regression_r2_comparison.png" alt="Regression R2 Comparison" width="100%">
</p>

### 🥇 Regression Winner

```text
Model → Stacking Regressor

MAE  → 7.8049
RMSE → 9.7166
R²   → 0.4947
```

---

# ⚔️ Decision Tree vs Bagging

One of the key experiments was comparing a single Decision Tree with Bagging.

| Model         | Classification Accuracy | Regression RMSE | Regression R² |
| ------------- | ----------------------: | --------------: | ------------: |
| Decision Tree |                  62.98% |         14.3772 |       -0.1063 |
| Bagging       |              **72.21%** |     **10.0406** |    **0.4604** |

### 💡 Key Insight

Bagging significantly improved performance over the single Decision Tree.

```text
Classification Accuracy

62.98%
   ↓
72.21%

+9.23 percentage points
```

Regression also improved considerably:

```text
RMSE
14.3772
   ↓
10.0406
```

This demonstrates how combining multiple learners can improve stability and generalization.

---

# 🗳️ Hard Voting vs Soft Voting

| Method         |   Accuracy |  Precision | Recall |         F1 |
| -------------- | ---------: | ---------: | -----: | ---------: |
| 🥇 Hard Voting | **0.7413** | **0.6921** | 0.5590 | **0.6184** |
| Soft Voting    |     0.7346 |     0.6770 | 0.5590 |     0.6124 |

### Result

**Hard Voting performed slightly better than Soft Voting on this test set.**

---

# 📏 Evaluation Metrics

## Classification Metrics

### Accuracy

Measures the proportion of correct predictions.

```text
Accuracy =
Correct Predictions / Total Predictions
```

### Precision

Measures how many predicted positive cases were actually positive.

```text
Precision =
TP / (TP + FP)
```

### Recall

Measures how many actual positive cases were correctly identified.

```text
Recall =
TP / (TP + FN)
```

### F1 Score

Balances Precision and Recall.

```text
F1 =
2 × Precision × Recall
----------------------
Precision + Recall
```

### ROC-AUC

Measures the model's ability to distinguish between classes.

---

## Regression Metrics

### MAE

Average absolute prediction error.

```text
MAE =
Mean(|Actual - Predicted|)
```

### RMSE

Penalizes larger prediction errors more strongly.

```text
RMSE =
√ Mean((Actual - Predicted)²)
```

### R²

Measures the proportion of variance explained by the model.

```text
R² =
1 - SS_res / SS_tot
```

---

# 🖼️ Project Visualizations

### 🔹 Ensemble Workflow

<img src="assets/workflow.png" alt="Workflow" width="100%">

### 🔹 Classification Comparison

<img src="assets/classification_model_comparison.png" alt="Classification Comparison" width="100%">

### 🔹 Regression RMSE

<img src="assets/regression_rmse_comparison.png" alt="Regression RMSE" width="100%">

### 🔹 Regression R²

<img src="assets/regression_r2_comparison.png" alt="Regression R2" width="100%">

---

# 📸 Notebook Results

<p align="center">
  <img src="assets/notebook_output_1.png" alt="Notebook Output 1" width="95%">
</p>

<p align="center">
  <img src="assets/notebook_output_2.png" alt="Notebook Output 2" width="95%">
</p>

---

# 🔄 End-to-End Workflow

```text
                RAW DATASET
                     │
                     ▼
             Data Understanding
                     │
                     ▼
          Missing Value Analysis
                     │
                     ▼
           Feature / Target Split
                     │
                     ▼
             Train-Test Split
                     │
                     ▼
       Imputation + Encoding + Scaling
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
   CLASSIFICATION           REGRESSION
          │                     │
          ▼                     ▼
   Bagging / AdaBoost     Bagging / AdaBoost
   Gradient Boosting      Gradient Boosting
   LightGBM / XGBoost     LightGBM / XGBoost
   Voting / Stacking      Stacking
          │                     │
          ▼                     ▼
 Accuracy / Precision       MAE / RMSE
 Recall / F1 / ROC-AUC          R²
          │                     │
          ▼                     ▼
 Stacking Classifier      Stacking Regressor
```

---

# 🧰 Tech Stack

| Technology          | Purpose               |
| ------------------- | --------------------- |
| 🐍 Python 3.13      | Programming           |
| 🐼 Pandas           | Data manipulation     |
| 🔢 NumPy            | Numerical computation |
| 📊 Matplotlib       | Data visualization    |
| 🤖 Scikit-learn     | ML & preprocessing    |
| ⚡ LightGBM          | Gradient boosting     |
| 🚀 XGBoost          | Gradient boosting     |
| 📓 Jupyter Notebook | Development           |

---

# 📁 Project Structure

```text
Ensemble-Learning/
│
├── 📓 Project_5_Ensemble_Learning.ipynb
├── 📊 dataset.5.csv
├── 📖 README.md
│
└── 📁 assets/
    ├── workflow.png
    ├── classification_model_comparison.png
    ├── regression_rmse_comparison.png
    ├── regression_r2_comparison.png
    ├── notebook_output_1.png
    └── notebook_output_2.png
```

> ⚠️ Keep the `assets` folder in the same location as `README.md` so GitHub can correctly display the images.

---

# ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/pmanan2031-web/ENSEMBLE-LEARNING.git
```

```bash
cd ENSEMBLE-LEARNING
```

### 2️⃣ Install Required Libraries

```bash
pip install pandas numpy matplotlib scikit-learn lightgbm xgboost jupyter
```

### 3️⃣ Start Jupyter Notebook

```bash
jupyter notebook
```

### 4️⃣ Open the Notebook

```text
Project_5_Ensemble_Learning.ipynb
```

Make sure the dataset is available:

```text
dataset.5.csv
```

---

# ▶️ How to Use

```text
1. Load dataset
      ↓
2. Explore data
      ↓
3. Handle missing values
      ↓
4. Encode categorical features
      ↓
5. Scale numerical features
      ↓
6. Train ensemble models
      ↓
7. Evaluate models
      ↓
8. Compare performance
      ↓
9. Select best models
```

---

# 💡 Key Learnings

### 🧠 Ensemble Learning

Combining multiple models can improve predictive performance and stability.

### 📦 Bagging

Bagging is particularly useful for reducing variance and stabilizing tree-based models.

### 🚀 Boosting

Boosting builds models sequentially, allowing later models to focus on previous errors.

### 🗳️ Voting

Voting combines predictions from multiple classifiers using majority voting or probability averaging.

### 🧩 Stacking

Stacking uses a meta-model to learn how different base models should be combined.

---

# 🔥 Key Project Insights

### Insight 1 — Ensemble > Single Model

Bagging improved the Decision Tree classification accuracy from:

```text
62.98% → 72.21%
```

### Insight 2 — Stacking Was the Overall Winner

Stacking achieved the best result for both tasks:

```text
Classification
F1 = 0.6316
ROC-AUC = 0.7940

Regression
R² = 0.4947
RMSE = 9.7166
MAE = 7.8049
```

### Insight 3 — Boosting Models Are Competitive

Gradient Boosting, LightGBM and XGBoost all produced strong results across the experiments.

### Insight 4 — Voting Behavior

Hard Voting slightly outperformed Soft Voting on this dataset.

---

# 📋 Project Checklist

* [x] Dataset loading
* [x] Data understanding
* [x] Missing value analysis
* [x] Train/Test split
* [x] Numerical preprocessing
* [x] Categorical encoding
* [x] Feature scaling
* [x] Decision Tree baseline
* [x] Bagging Classifier
* [x] Bagging Regressor
* [x] Bagging vs base model
* [x] AdaBoost Classifier
* [x] AdaBoost Regressor
* [x] Gradient Boosting Classifier
* [x] Gradient Boosting Regressor
* [x] Learning rate analysis
* [x] Estimator analysis
* [x] LightGBM Classifier
* [x] LightGBM Regressor
* [x] XGBoost Classifier
* [x] XGBoost Regressor
* [x] Boosting comparison
* [x] Hard Voting
* [x] Soft Voting
* [x] Stacking Classifier
* [x] Stacking Regressor
* [x] Classification evaluation
* [x] Regression evaluation
* [x] Model comparison
* [x] Best model selection
* [x] Final analysis

---

# 🚀 Future Improvements

Possible future extensions:

```text
🔹 Hyperparameter Optimization
🔹 Cross-Validation
🔹 Feature Importance Analysis
🔹 SHAP Explainability
🔹 Streamlit Prediction App
🔹 Model Saving with Joblib
🔹 REST API Deployment
🔹 Cloud Deployment
```

---

# 🏁 Final Conclusion

This project provides a practical and complete comparison of **ensemble learning techniques for classification and regression**.

The experiments demonstrate that ensemble approaches can significantly improve upon a single Decision Tree baseline.

### 🏆 Final Recommended Models

| Task              | Best Model              |          Result |
| ----------------- | ----------------------- | --------------: |
| 🎯 Classification | **Stacking Classifier** | F1 = **0.6316** |
| 📈 Regression     | **Stacking Regressor**  | R² = **0.4947** |

The project demonstrates the practical value of **Bagging, Boosting, Voting and Stacking** for building stronger machine learning solutions.

---

# 👨‍💻 Author

<div align="center">

### **Manan**

🐍 Python • 🤖 Machine Learning • 📊 Data Science

---

### ⭐ If you found this project useful, consider giving the repository a star!

**Made with ❤️ by Manan**

</div>
