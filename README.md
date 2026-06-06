# Bank Financial Analysis & Machine Learning Capstone

## Project Overview
An end-to-end data analytics project using PySpark to process a European bank's relational database. The goal is to aggregate transactional data, engineer features, handle class imbalances, and train a Logistic Regression model to predict loan defaulters.

---

## Dataset Description

### Core Relational Tables
* **account.csv:** Static account details and regional locations.
* **client.csv:** Client demographics (birthdate, gender).
* **relationship.csv:** Connects clients to their respective accounts.
* **trans.csv:** Ledger tracking over 1 million individual transactions.
* **loan.csv:** Details on loans granted, amounts, and contract statuses.
* **order.csv:** Permanent payment orders for automatic debiting.
* **card.csv:** Credit/debit cards issued to specific accounts.
* **district.csv:** Socioeconomic metrics of administrative regions.

### Engineered Outputs
* **final_dataframe.csv:** The master consolidated dataset built by joining all relational tables.
* **imbalanced_ros.csv / imbalanced_rus.csv / imbalanced_rsm.csv:** Training variations balanced using Random Over-Sampling (ROS), Random Under-Sampling (RUS), and SMOTE (RSM) to handle credit default imbalances.

---

## Key Technical Steps
* **Data Engineering:** Used PySpark to join all 8 files and aggregate row-level transactions from `trans.csv` into account-level features.
* **Feature Engineering:** Implemented StringIndexing, One-Hot Encoding, and `StandardScaler` to prepare categorical and numerical data.
* **Predictive Modeling:** Trained PySpark `LogisticRegression` models and evaluated performance using Area Under the ROC Curve (AUC).

---

## Key Conclusions
* **Feature Scaling:** Standardizing variables balances feature importance, preventing columns with large raw values from distorting model training.
* **PySpark Optimization:** Similar AUC results across unscaled and scaled PySpark Logistic Regression models showcase the mathematical robustness of Spark's native L-BFGS optimizer.
* **Sampling Effectiveness:** Balancing datasets via ROS or SMOTE drastically reduced false negatives, enhancing the model's ability to identify high-risk accounts.

---

## How to Run
1. Clone this repository.
2. Ensure PySpark and Jupyter Notebook are installed.
3. Open `EDE_Capstone_SC_Nasiruddin_arif.ipynb` and click **Kernel -> Restart & Run All**.
