🎓 Ensemble Learning — Student Completion & Final Score Prediction

<p align="center">
  <img src="assets/workflow.png" alt="End-to-End Ensemble Learning Workflow" width="100%">
</p>

<p align="center">
  <b>End-to-End Ensemble Machine Learning Project</b><br>
  Classification + Regression • Bagging • AdaBoost • Gradient Boosting • LightGBM • XGBoost • Voting • Stacking
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue?logo=python">
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas">
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikit-learn">
  <img src="https://img.shields.io/badge/LightGBM-Boosting-9ACD32">
  <img src="https://img.shields.io/badge/XGBoost-Boosting-EC4E20">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter">
</p>

📌 Project Overview

This project demonstrates ensemble learning techniques for two machine learning tasks using student learning/activity data:

Classification: Predict whether a student completes the course.

Regression: Predict the student's final score.

The notebook compares multiple ensemble strategies and identifies the strongest models using task-specific evaluation metrics.

Best Classification Model: Stacking — F1 = 0.6316, Accuracy = 0.7442, ROC-AUC = 0.7940
Best Regression Model: Stacking — R² = 0.4947, RMSE = 9.7166, MAE = 7.8049

🎯 Business / Learning Objective

The project is designed to answer questions such as:

Which students are likely to complete a course?

What final score can be expected from a student's learning activity?

Does an ensemble model outperform a single Decision Tree?

Which boosting algorithm performs best on this dataset?

Does combining different models through Voting or Stacking improve performance?

🗂️ Dataset

The notebook loads:

dataset.5.csv

Dataset Snapshot

Property

Value

Rows

5,200

Columns

19

Classification Target

completion_status

Regression Target

final_score

Missing values

Present in 3 numeric columns

Train/Test Split

80% / 20%

Main Features

Student profile

age

country_region

device_type

education_background

Course information

course_level

course_category

course_start_date

week_of_year

Learning activity

sessions

time_spent_hours

videos_watched

quiz_attempts

assignments_submitted

forum_posts

Performance

avg_quiz_score

attendance_rate

Targets

completion_status

final_score

🔍 Data Understanding

The dataset contains 5,200 records and 19 columns.

The classification target distribution is:

Completion Status

Count

Approx. Share

0

3,248

62.45%

1

1,952

37.55%

The final_score statistics are:

Statistic

Value

Mean

74.82

Std. Dev.

13.53

Minimum

35.20

Median

74.10

Maximum

100.00

🧹 Data Preprocessing

The project uses a reusable ColumnTransformer + Pipeline preprocessing approach.

Numerical Features

Missing values → Median Imputation

Scaling → StandardScaler

Categorical Features

Missing values → Most Frequent Imputation

Encoding → OneHotEncoder

Unknown categories → safely ignored

After preprocessing:

Training samples: 4,160
Testing samples : 1,040
Processed features: 30

🧠 Machine Learning Architecture

<p align="center">
  <img src="assets/workflow.png" alt="Machine Learning Workflow" width="100%">
</p>

The project covers both parallel ensemble learning and sequential ensemble learning.

1️⃣ Bagging

Bagging trains multiple models on bootstrap samples and combines their predictions.

Implemented:

Bagging Classifier

Bagging Regressor

Main benefit: reduces variance and improves model stability.

2️⃣ AdaBoost

AdaBoost builds weak learners sequentially. Later learners focus more on difficult observations.

Implemented:

AdaBoost Classifier

AdaBoost Regressor

3️⃣ Gradient Boosting

Gradient Boosting sequentially builds trees to reduce previous prediction errors.

Implemented:

Gradient Boosting Classifier

Gradient Boosting Regressor

The project also studies the effect of:

Learning Rate → 0.01, 0.05, 0.10

Estimators → 50, 100, 200

4️⃣ LightGBM

LightGBM is an efficient gradient boosting framework designed for speed and scalability.

Implemented:

LGBMClassifier

LGBMRegressor

5️⃣ XGBoost

XGBoost is a powerful, regularized gradient boosting algorithm.

Implemented:

XGBClassifier

XGBRegressor

6️⃣ Voting

Multiple classifiers are combined through:

Hard Voting → majority class

Soft Voting → predicted probabilities

7️⃣ Stacking

Different base models generate predictions and a meta-learner learns how to combine them.

Implemented:

Stacking Classifier

Stacking Regressor

📊 Model Performance

🏆 Classification Results

<p align="center">
  <img src="assets/classification_model_comparison.png" alt="Classification Model Comparison" width="100%">
</p>

Model

Accuracy

Precision

Recall

F1

ROC-AUC

Single Decision Tree

0.6298

0.5062

0.5256

0.5157

0.6090

Bagging

0.7221

0.6472

0.5692

0.6057

0.7739

AdaBoost

0.7365

0.6883

0.5436

0.6074

0.7850

Gradient Boosting

0.7385

0.6832

0.5641

0.6180

0.7896

LightGBM

0.7288

0.6656

0.5564

0.6061

0.7805

XGBoost

0.7356

0.6727

0.5744

0.6196

0.7875

Hard Voting

0.7413

0.6921

0.5590

0.6184

—

Soft Voting

0.7346

0.6770

0.5590

0.6124

0.7891

Stacking

0.7442

0.6867

0.5846

0.6316

0.7940

🥇 Classification Winner

Stacking Classifier

Accuracy: 74.42%

F1 Score: 0.6316

ROC-AUC: 0.7940

This model achieved the highest F1 score and ROC-AUC among the evaluated models.

📈 Regression Results

<p align="center">
  <img src="assets/regression_rmse_comparison.png" alt="Regression RMSE Comparison" width="90%">
</p>

<p align="center">
  <img src="assets/regression_r2_comparison.png" alt="Regression R2 Comparison" width="90%">
</p>

Model

MAE

RMSE

R²

AdaBoost

8.5452

10.5174

0.4080

Gradient Boosting

7.9645

9.9190

0.4734

LightGBM

7.9309

9.8756

0.4780

XGBoost

7.9205

9.8749

0.4781

Stacking

7.8049

9.7166

0.4947

🥇 Regression Winner

Stacking Regressor

MAE: 7.8049

RMSE: 9.7166

R²: 0.4947

The Stacking Regressor provides the lowest error and highest R² among the evaluated regression models.

⚔️ Bagging vs Single Decision Tree

Model

Classification Accuracy

Regression RMSE

Regression R²

Single Decision Tree

0.6298

14.3772

-0.1063

Bagging

0.7221

10.0406

0.4604

Key Insight

Bagging substantially improves over the single Decision Tree:

Classification accuracy increases from 62.98% → 72.21%

Regression RMSE decreases from 14.3772 → 10.0406

Regression R² improves from -0.1063 → 0.4604

This demonstrates the practical benefit of combining multiple base learners.

🗳️ Hard Voting vs Soft Voting

Method

Accuracy

Precision

Recall

F1

Hard Voting

0.7413

0.6921

0.5590

0.6184

Soft Voting

0.7346

0.6770

0.5590

0.6124

Hard Voting performed slightly better than Soft Voting on this test set.

🧪 Evaluation Metrics

Classification

The project evaluates:

Accuracy — overall correct predictions

Precision — correctness of positive predictions

Recall — ability to identify positive cases

F1 Score — balance between precision and recall

ROC-AUC — ranking/discrimination capability

Regression

The project evaluates:

MAE — average absolute prediction error

RMSE — error measure that penalizes larger errors more strongly

R² — proportion of variance explained by the model

🧰 Tech Stack

Technology

Purpose

Python 3.13

Programming

Pandas

Data manipulation

NumPy

Numerical computation

Matplotlib

Visualization

Scikit-learn

Preprocessing & ML

LightGBM

Gradient boosting

XGBoost

Gradient boosting

Jupyter Notebook

Development environment

📁 Project Structure

project/
│
├── project5.ipynb
├── dataset.5.csv
├── README.md
│
└── assets/
    ├── workflow.png
    ├── classification_model_comparison.png
    ├── regression_rmse_comparison.png
    ├── regression_r2_comparison.png
    ├── notebook_output_1.png
    └── notebook_output_2.png

Keep the assets folder in the same directory as README.md so the images render correctly on GitHub.

▶️ How to Run

1. Clone the repository

git clone <YOUR-REPOSITORY-URL>
cd <YOUR-REPOSITORY-FOLDER>

2. Install dependencies

pip install pandas numpy matplotlib scikit-learn lightgbm xgboost jupyter

3. Open Jupyter Notebook

jupyter notebook

4. Run

Open:

project5.ipynb

Make sure:

dataset.5.csv

is available in the expected working directory.

🧩 Project Workflow

Raw Dataset
    ↓
Data Understanding
    ↓
Missing Value Analysis
    ↓
Feature / Target Selection
    ↓
Train-Test Split
    ↓
Imputation + Encoding + Scaling
    ↓
┌─────────────────────────────┐
│ Classification              │
│ completion_status           │
└─────────────────────────────┘
    ↓
Bagging / AdaBoost / GB /
LightGBM / XGBoost / Voting /
Stacking
    ↓
Accuracy / Precision / Recall /
F1 / ROC-AUC
    ↓
Best → Stacking Classifier


┌─────────────────────────────┐
│ Regression                  │
│ final_score                 │
└─────────────────────────────┘
    ↓
Bagging / AdaBoost / GB /
LightGBM / XGBoost / Stacking
    ↓
MAE / RMSE / R²
    ↓
Best → Stacking Regressor

💡 Key Learnings

Ensemble Learning

Combining multiple models can improve predictive performance and stability compared with relying on one weak or high-variance model.

Bagging

Useful for reducing variance and stabilizing tree-based predictions.

Boosting

Builds models sequentially so later learners can focus on previous errors.

Voting

Combines independent classifier decisions through majority voting or probability averaging.

Stacking

Uses a meta-model to learn how different base models should be combined.

🏁 Final Conclusion

This project provides a complete practical comparison of ensemble learning techniques for both classification and regression.

The experiments show that:

Ensemble methods outperform the single Decision Tree baseline.

Boosting algorithms provide competitive classification and regression performance.

Hard Voting slightly outperforms Soft Voting on this dataset.

Stacking achieved the best classification F1 and ROC-AUC.

Stacking also achieved the best regression R² and lowest RMSE/MAE.

Final Models

Task

Recommended Model

Key Result

🎯 Classification

Stacking Classifier

F1 = 0.6316

📈 Regression

Stacking Regressor

R² = 0.4947

📌 Project Checklist

Dataset loaded

Classification target identified

Regression target identified

Train/Test split completed

Missing value handling

Encoding + scaling

Bagging Classifier

Bagging Regressor

Bagging vs base model

AdaBoost Classifier

AdaBoost Regressor

Weak learner improvement analysis

Gradient Boosting Classifier

Gradient Boosting Regressor

Learning rate / estimator analysis

LightGBM Classifier

LightGBM Regressor

XGBoost Classifier

XGBoost Regressor

Boosting model comparison

Hard Voting

Soft Voting

Stacking Classifier

Stacking Regressor

Classification evaluation

Regression evaluation

Best models identified

Final analysis

👨‍💻 Author

Manan

Machine Learning • Python • Data Science

<p align="center">
  <b>⭐ If you found this project useful, consider giving the repository a star!</b>
</p>
