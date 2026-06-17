# Bank Marketing Analytics — Predicting Customer Subscription to Term Deposits

A machine learning project that predicts whether a bank customer will subscribe to a term deposit, based on demographic, financial, and campaign interaction data. Built to support data-driven marketing decisions and improve campaign targeting efficiency.

---

## Overview

This project applies exploratory data analysis and four machine learning classification models to a Portuguese bank marketing dataset (~11,000 observations). A key focus of this study is **avoiding data leakage** — the `duration` variable, which is only known after a call ends, was deliberately excluded to ensure the model is realistic and deployable in real-world scenarios.

An **interactive dashboard** was also developed to allow bankers to explore customer behavior patterns, compare model performance, and predict individual customer subscription likelihood.

---

## Objectives

1. Identify the key factors influencing customer subscription to term deposits.
2. Build and compare multiple machine learning models for subscription prediction.
3. Develop a realistic, interpretable, and practically applicable model.
4. Provide actionable business insights to improve bank marketing campaign efficiency.

---

## Repository Structure

```
├── code.ipynb        # Main analysis notebook (EDA, preprocessing, model building)
├── dashboard/            # Dashboard application files
├── Report.pdf            # Full project report
└── README.md
```

---

## Dataset

- **Source:** [Bank Marketing Dataset — Kaggle](https://www.kaggle.com/datasets)
- **Size:** 11,162 observations, 17 variables
- **Target variable:** `deposit` — whether the customer subscribed to a term deposit (Yes/No)

### Variable Description

| Variable | Type | Description |
|---|---|---|
| `age` | Integer | Age of the customer |
| `job` | Categorical | Type of job |
| `marital` | Categorical | Marital status (single, married, divorced) |
| `education` | Categorical | Education level |
| `default` | Binary | Has credit in default? |
| `balance` | Integer | Average yearly balance (euros) |
| `housing` | Binary | Has a housing loan? |
| `loan` | Binary | Has a personal loan? |
| `contact` | Categorical | Contact communication type |
| `month` | Date | Last contact month |
| `duration` | Integer | Last contact duration in seconds ⚠️ excluded (data leakage) |
| `campaign` | Integer | Number of contacts during this campaign |
| `pdays` | Integer | Days since last contact from previous campaign |
| `previous` | Integer | Contacts before this campaign |
| `poutcome` | Categorical | Outcome of the previous marketing campaign |
| `deposit` | Binary | **Target** — subscribed to term deposit? |

---

## Data Preprocessing & Feature Engineering

- No missing or duplicate values were found.
- **New features created:**
  - `age_group` — customers categorised into student, early career, mid career, pre-retirement, retired
  - `balance_log` — log transformation applied to the skewed balance variable
  - `month_sin` / `month_cos` — cyclical encoding of the month variable
- **Encoding methods used:**
  - Ordinal encoding — `education`, `age_group`
  - Target encoding — `job`, `poutcome`
  - One-hot encoding — `marital`, `contact`
  - Boolean encoding — `default`, `housing`, `loan`
- Features standardised using `StandardScaler`.
- `duration`, `month`, `day`, and `age` were dropped before modelling.

---

## Models Built

Four supervised classification models were trained and evaluated:

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.71 | 0.72 | 0.64 | 0.68 | 0.76 |
| Decision Tree | 0.71 | 0.78 | 0.55 | 0.65 | 0.76 |
| **Random Forest** | **0.75** | **0.82** | **0.60** | **0.69** | **0.80** |
| XGBoost | 0.74 | 0.79 | 0.63 | 0.70 | 0.80 |

**Best model: Random Forest** — highest accuracy (75%) and precision (82%).

### Model Improvement Techniques
- `GridSearchCV` — used for Logistic Regression and Decision Tree
- `RandomizedSearchCV` — used for Random Forest and XGBoost
- Cross-validation during tuning to reduce overfitting
- Prediction threshold adjusted to 0.4 (from default 0.5) to improve recall

---

## Key Findings

- **Previous campaign outcome (`poutcome`)** is the strongest predictor — customers with a positive past response are far more likely to subscribe again.
- **Contact type** matters — customers reached via cellular showed higher subscription rates.
- **Customers without housing loans** are more likely to subscribe.
- **Single customers** tend to subscribe more than married customers.
- **Higher account balance** is weakly associated with higher subscription likelihood.
- The `duration` variable was highly predictive but was excluded as it introduces data leakage and is unavailable before a call.

---

## Dashboard

An interactive dashboard was built to support practical use by bank marketers. It includes:

- Summary metrics (total clients, deposit rate, best AUC, best accuracy)
- Exploratory analysis plots — selectable by feature
- ROC curves, confusion matrix, and threshold curve per model
- Feature importance charts (Logistic Regression and Random Forest)
- **Live prediction panel** — enter a customer's details and get a subscription prediction from any of the four models

---

## Requirements

- Python 3.8+
- Libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `plotly`

Install dependencies:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn plotly
```

---

## 👤 Author

**D.G.S.S. Gunasekara**

---

## References

- Bachmann, J. (2023). Bank Marketing Dataset. Kaggle.
- Zaki, A.M. et al. (2024). Predictive analytics and machine learning in direct marketing for anticipating bank term deposit subscription.
- Yi, H. (2024). Predicting customer subscription to fixed-term deposit products based on machine learning approach.
- Mumford, C. (2023). Market Research & Customer Conversion Analysis. Kaggle.
- Adel Gabr, M. (2023). Bank Marketing Campaign Analysis. Kaggle.
