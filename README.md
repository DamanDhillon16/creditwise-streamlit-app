# Loan Approval Prediction

A machine learning project that predicts whether a loan application will be approved, comparing Logistic Regression, KNN and Naive Bayes. The best model is served through a Streamlit app.

**Live app:** [PASTE-LIVE-LINK-HERE](https://creditwise-app-app-dcwxxnp6za9kb3pqfcs7gd.streamlit.app/)

![App screenshot](PASTE-SCREENSHOT-HERE)

## Dataset

1,000 loan applicants with 19 features (income, credit score, DTI ratio, savings, collateral value, loan amount and term, employment and demographic details) and the target `Loan_Approved`. Source: [add dataset source and link]. In the test split, about 30% of applications were approved, so the classes are imbalanced.

## Approach

1. **Cleaning:** filled missing values (about 5% per column) with the mean for numeric columns and the most frequent value for categorical columns. Dropped `Applicant_ID`.
2. **EDA:** class balance, distributions, box plots against the target, and a correlation heatmap. Credit score had the strongest correlation with approval (0.45).
3. **Encoding:** label-encoded education level and the target; one-hot encoded the other categorical columns.
4. **Split and scaling:** 80/20 train-test split (random_state 42); StandardScaler fit on the training set only.
5. **Feature engineering:** added squared DTI ratio and squared credit score, replacing the original columns.
6. **Models:** Logistic Regression, KNN (k=5) and Gaussian Naive Bayes.

## Results (test set, 200 applicants)

| Model | Accuracy | Precision | Recall |
|---|---|---|---|
| Logistic Regression (baseline) | 86.5% | 78.3% | 77.0% |
| KNN (baseline) | 76.0% | 62.7% | 52.5% |
| Naive Bayes (baseline) | 86.5% | 80.4% | 73.8% |
| **Logistic Regression (with feature engineering)** | **88.0%** | 78.5% | **83.6%** |
| KNN (with feature engineering) | 78.5% | 67.3% | 57.4% |
| Naive Bayes (with feature engineering) | 86.0% | **81.1%** | 70.5% |

Logistic Regression gave the best accuracy and recall. Naive Bayes gave the highest precision, meaning the fewest wrongly approved loans. The right choice depends on whether the lender cares more about avoiding bad approvals or missing good applicants.

## Limitations and Next Steps

- Small dataset and a single 200-row test split; add cross-validation
- Missing values were filled before the split, which leaks a little information; use a scikit-learn Pipeline
- Gender and marital status are used as features. Real credit models exclude protected attributes, so a fairness check and a rerun without them is a next step
- Report F1 and ROC-AUC alongside accuracy, since the classes are imbalanced

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit

## Run

```bash
pip install -r requirements.txt
jupyter notebook credit_wise.ipynb
streamlit run app.py
```

## Author

**Daman Dhillon**
[LinkedIn](https://www.linkedin.com/in/daman-jeet-kaur-dhillon-324191312/) | [GitHub](https://github.com/DamanDhillon16)

Copyright (c) 2026 Daman Dhillon. All rights reserved.
