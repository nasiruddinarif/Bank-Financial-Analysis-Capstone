# Bank Loan Default Prediction — PySpark Pipeline & ML Capstone

End-to-end data engineering and machine learning project built on a European bank's relational dataset (8 linked tables, 1M+ transactions, 5,369 clients, 682 loan accounts). The project covers the full pipeline: loading raw data into Spark, exploratory data analysis, merging relational tables into a single loan-level dataset, feature engineering, handling class imbalance, and training/comparing classification models to predict loan default.

## Project Overview

Banks need to assess loan default risk using fragmented operational data spread across multiple systems (accounts, clients, transactions, cards, loans, districts). This project simulates that real-world challenge: consolidating 8 relational CSV sources into a single ML-ready dataset and building a working default-prediction model.

## Dataset

Source: European bank financial dataset (8 CSV tables)

| Table | Description |
|---|---|
| `account.csv` | Static account characteristics |
| `client.csv` | Client demographic data |
| `card.csv` | Credit card issuance per client |
| `district.csv` | Demographic/socioeconomic data per district |
| `loan.csv` | Loan details granted to accounts |
| `order.csv` | Standing payment orders |
| `relationship.csv` (`disp`) | Client-to-account relationships |
| `trans.csv` | Transaction history (1M+ records) |

## Methodology

1. **Data Ingestion** — Loaded all 8 CSV sources into Spark DataFrames, corrected column names/types.
2. **Exploratory Data Analysis** (Spark SQL) — Investigated patterns across branches, client demographics, card types, loan distributions, and transaction behavior.
3. **Data Merging** — Joined all 8 tables into a single loan-level dataset, preserving every loan applicant record.
4. **Preprocessing** — Handled missing values post-merge.
5. **Outlier & Anomaly Detection** — Profiled the merged dataset for irregular values.
6. **District Data Quality Check** — Investigated anomalies in district-level variables (A15, A16).
7. **Localization Cleanup** — Translated Czech/Polish categorical column values into English using a data dictionary.
8. **Transaction Feature Engineering** — Aggregated multiple transactions per client into high-level behavioral features.
9. **Target Simplification** — Converted the 4-category loan status variable into a binary target (Defaulter / Non-defaulter).
10. **Class Imbalance Handling** — Compared Random Under-Sampling (RUS), Random Over-Sampling (ROS), and SMOTE on a severely imbalanced dataset (606 non-defaulters vs. 76 defaulters).
11. **Spark ML Pipeline** — Loaded the processed dataset into Spark for modeling.
12. **Train/Test Split** — Preprocessed and split data for model training and evaluation.
13. **Model Training & Comparison** — Trained Logistic Regression, Random Forest, and Gradient Boosted Trees (GBT) on SMOTE-balanced data.
14. **Standardization Impact Test** — Evaluated whether feature standardization improved Logistic Regression performance.

## Results

| Model | Balancing Strategy | AUC |
|---|---|---|
| Logistic Regression | SMOTE | 0.6442 |
| Random Forest | SMOTE | 0.6974 |
| **Gradient Boosted Trees** | **SMOTE** | **0.7056 (best performing)** |

Feature standardization was tested against Logistic Regression and did not materially improve AUC — full comparison and reasoning is documented in the notebook.

## Tech Stack

- **Apache Spark** (PySpark, Spark SQL, Spark MLlib)
- **Python** (Pandas for supplementary preprocessing)
- **Machine Learning**: Logistic Regression, Random Forest, Gradient Boosted Trees
- **Imbalance handling**: Random Under-Sampling, Random Over-Sampling, SMOTE

## Repository Structure
├── EDE_Capstone_SC_Nasiruddin_arif.ipynb   # Full end-to-end pipeline notebook
├── requirements.txt
└── README.md

## How to Run

1. Clone the repository.
2. Install dependencies:
pip install -r requirements.txt
3. Place the 8 source CSV files in a local `data/` folder (not included in this repo — see note below).
4. Open the notebook in Jupyter and run cells sequentially.

> **Note on data:** Raw CSV files are not included in this repository due to size. The notebook expects them at `/content/data/` (Colab-style path) — update the paths at the top of the notebook to match your local `data/` folder if running outside Colab.

## Author

Nasiruddin Arif Bin Shar Sham — Certified Enterprise Data Engineer (CADS AI)
[LinkedIn](https://linkedin.com/in/nasiruddinarif)
