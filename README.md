# Bank Loan Default Prediction — PySpark Pipeline & ML Capstone

End-to-end data engineering and machine learning project built on a European bank's relational dataset (8 linked tables, 1M+ transactions, 5,369 clients, 682 loan accounts). The project covers the full pipeline: loading raw data into Spark, exploratory data analysis, merging relational tables into a single loan-level dataset, feature engineering, handling class imbalance, and training/comparing classification models to predict loan default.

## Project Overview

Banks need to assess loan default risk using fragmented operational data spread across multiple systems (accounts, clients, transactions, cards, loans, districts). This project simulates that real-world challenge: consolidating 8 relational CSV sources into a single ML-ready dataset and building a working default-prediction model.

## Dataset

**Source:** European bank financial dataset (8 CSV tables), provided as a compressed archive: `dataset/banking_capstone_data.rar`

| Table | Observations | Description |
|---|---|---|
| `account.csv` | 4,500 | Static account characteristics |
| `client.csv` | 5,369 | Client demographic data |
| `relationship.csv` (disp) | 5,369 | Client-to-account relationships |
| `order.csv` | 6,471 | Standing payment orders |
| `trans.csv` | 1,056,320 | Transaction history |
| `loan.csv` | 682 | Loan details granted to accounts |
| `card.csv` | 892 | Credit card issuance per client |
| `district.csv` | 77 | Demographic/socioeconomic data per district |

## Methodology

1. **Building the Spark DataFrames** — Loaded each of the eight source tables into Spark, cleaning up column names and casting data types where the raw CSVs didn't infer them correctly.
2. **Exploratory Data Analysis (Spark SQL)** — Investigated branch-level customer concentration, account growth trends, client gender split, card-type preferences by demographic, loan amount distributions, loan-vs-salary relationships, arrears vs. unemployment rate, and regional transaction behavior.
3. **Building the Modeling Dataset** — Merged tables around the loan applicants, preserving every loan record.
4. **Preprocessing** — Handled missing values introduced by the merge.
5. **Outlier & Anomaly Detection** — Profiled the merged dataset for irregular values.
6. **District Data Quality Check** — Investigated inconsistencies in district-level fields A15 and A16.
7. **Localization Cleanup** — Translated Czech categorical values (e.g. `POPLATEK MESICNE` → `MONTHLY`) into English using the data dictionary via SQL CASE statements.
8. **Transaction Feature Engineering** — Aggregated per-account transaction history into behavioral features (transaction volume, amount statistics, balance statistics, credit/debit transaction counts).
9. **Target Simplification** — Converted the four-category loan status into a binary target (Defaulter / Non-defaulter).
10. **Class Imbalance Handling** — Compared Random Under-Sampling (RUS), Random Over-Sampling (ROS), and SMOTE on a severely imbalanced dataset (606 non-defaulters vs. 76 defaulters).
11. **Loading Balanced Data into Spark** — Brought the SMOTE-balanced dataset back into Spark for modeling.
12. **Train/Test Split** — Assembled features and split data (80/20) for training and evaluation.
13. **Model Training & Comparison** — Trained Logistic Regression, Random Forest, and Gradient Boosted Trees (GBT) on SMOTE-balanced data.
14. **Standardization Impact Test** — Evaluated whether feature standardization improved Logistic Regression performance.

## Results

| Model | Balancing Strategy | AUC |
|---|---|---|
| Logistic Regression | SMOTE | 0.6442 |
| Random Forest | SMOTE | 0.6974 |
| **Gradient Boosted Trees** | SMOTE | **0.7056 (best performing)** |

Feature standardization was tested against Logistic Regression and did not materially change AUC (0.6442 unscaled vs. 0.6442 scaled) — full comparison and reasoning is documented in the notebook.

## Tech Stack

- **Apache Spark** (PySpark, Spark SQL, Spark MLlib)
- **Python** (Pandas for supplementary preprocessing)
- **Machine Learning:** Logistic Regression, Random Forest, Gradient Boosted Trees
- **Imbalance handling:** Random Under-Sampling, Random Over-Sampling, SMOTE

## Repository Structure

├── End_to_end_bank_loan_default_DE_project.ipynb   # Full end-to-end pipeline notebook
├── dataset/
│   └── banking_capstone_data.rar                   # Compressed source data (8 CSV tables)
├── requirements.txt
└── README.md

## How to Run

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Download `dataset/banking_capstone_data.rar` from this repo and extract it into a local `data/` folder in your project root.
4. Open `End_to_end_bank_loan_default_DE_project.ipynb` in Jupyter and run cells sequentially.

**Note on data:** The dataset is provided as a compressed archive (`dataset/banking_capstone_data.rar`) to keep the repo lightweight. Extract it before running the notebook. The notebook expects CSVs at `/content/data/` (Colab-style path) — update paths if running locally.

## Author

**Nasiruddin Arif Bin Shar Sham** — Certified Enterprise Data Engineer (CADS AI)
[LinkedIn](https://www.linkedin.com/in/nasiruddinarif/)


