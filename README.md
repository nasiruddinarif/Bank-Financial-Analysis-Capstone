# Banking Data Pipeline & Loan Default Feature Engineering (PySpark)

End-to-end data engineering and feature engineering pipeline on a European retail banking dataset (8 relational tables, 1M+ transactions, 5,369 clients, 682 loans), built in PySpark and Spark SQL.

## Overview

This project takes raw relational banking data (accounts, clients, cards, districts, orders, transactions, and loans) and builds a loan-level, ML-ready dataset for predicting loan default. It covers the full pipeline: schema loading, exploratory analysis, multi-table joins, data quality checks, feature engineering, class imbalance handling, and model comparison.

**Dataset**: 8 CSVs — Account (4,500), Client (5,369), Disposition/Relationship (5,369), Order (6,471), Transaction (1,056,320), Loan (682), Credit Card (892), District demographics (77).

## Pipeline Steps

1. **Building the Spark DataFrames** — loaded all 8 source tables with schema inference and correct delimiters.
2. **Exploratory Data Analysis** — branch-level customer counts, account growth trends, gender splits, card type preferences by demographic, loan amount distributions, salary/unemployment correlations, and transaction breakdowns (Spark SQL, visualized via Pandas/Matplotlib).
3. **Building the Modeling Dataset** — joined loan → account → district → client → transaction aggregates into a single loan-level table via layered CTEs.
4. **Preprocessing** — missing value checks, duplicate detection, and grain validation (one row per `loan_id`).
5. **Outlier & Anomaly Detection** — IQR-based outlier checks (via `percentile_approx`) across amount, payments, duration, salary, unemployment rate, crime rates, and transaction aggregates; plus checks for invalid, negative, zero, and logically inconsistent values.
6. **District Field Review** — flagged inconsistencies in historical crime-rate fields (A15/A16).
7. **Czech → English Translation** — recoded categorical fields (loan frequency, transaction type/operation) using SQL `CASE` statements.
8. **Transaction Feature Engineering** — aggregated transaction-level data into account-level features (volume, amount stats, balance stats, credit/debit and withdrawal-type counts).
9. **Target Variable Simplification** — recast the four-category loan status into a binary Defaulter / Non-defaulter label.
10. **Class Imbalance Handling** — compared under-sampling, over-sampling, and SMOTE (606 non-defaulters vs. 76 defaulters).
11–12. **Balanced Data Load & Train/Test Split** — brought the SMOTE-balanced set back into Spark and split 80/20.
13. **Model Comparison** — trained Logistic Regression, Random Forest, and Gradient Boosted Trees in Spark MLlib.
14. **Standardization Check** — tested whether feature scaling improved Logistic Regression performance.

## Results

| Model | AUC |
|---|---|
| Logistic Regression | ~0.6x |
| Random Forest | ~0.6x |
| Gradient Boosted Trees | **0.71 (best)** |

Feature standardization showed no meaningful improvement on Logistic Regression in Spark, consistent with Spark ML's internal handling of feature scale.

## Tech Stack
- **PySpark** (DataFrame API + Spark SQL)
- **Spark MLlib** (Logistic Regression, Random Forest, GBT, VectorAssembler, StandardScaler)
- **Pandas / scikit-learn** (SMOTE, LabelEncoder)
- **Matplotlib** (visualization)

## Known Limitations / Next Steps
- Class balancing (SMOTE) is currently applied before the train/test split and on a single feature (`amount`); a future iteration will apply balancing after splitting and expand the feature set to include the engineered transaction and district features.
- Under/over-sampling are computed for comparison but not yet carried through to model training.
- Evaluation currently reports AUC only; precision/recall/F1 on the minority (Defaulter) class would better reflect real-world cost of misclassification.

## Repository Contents
- `Banking_Data_Pipeline___Loan_Default_Feature_Engineering_PySpark.ipynb` — full pipeline notebook

## Author

**Nasiruddin Arif Bin Shar Sham** — Certified Enterprise Data Engineer (CADS AI)
[LinkedIn](https://www.linkedin.com/in/nasiruddinarif/)


