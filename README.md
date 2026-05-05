# Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using **Logistic Regression** and **Random Forest**, with **SMOTE** to handle class imbalance — a real-world challenge in fraud datasets.

---

## Problem Statement

Credit card fraud causes billions in losses every year. The challenge is that fraudulent transactions are extremely rare (typically < 2% of all transactions), making this a classic **imbalanced classification** problem. A naive model that always predicts "Legit" gets 98% accuracy but catches zero fraud — this project addresses that correctly.

---

## Project Structure

```
fraud-detection/
│
├── train.py          # Full ML pipeline: load → preprocess → SMOTE → train → evaluate → save
├── predict.py        # Load saved model and score new transactions
├── eda.py            # Exploratory Data Analysis with visualizations
│
├── data/
│   └── transactions.csv        # Generated dataset (or replace with Kaggle data)
│
├── models/
│   ├── fraud_model.pkl         # Best trained model (saved by train.py)
│   └── scaler.pkl              # Fitted StandardScaler
│
├── outputs/
│   ├── class_distribution.png
│   ├── amount_distribution.png
│   ├── transactions_by_hour.png
│   ├── correlation_heatmap.png
│   └── feature_boxplots.png
│
├── requirements.txt
└── README.md
```

---

##  Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/fraud-detection.git
cd fraud-detection

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

##  Usage

### Step 1 — Exploratory Data Analysis
```bash
python train.py       # generates data/transactions.csv first
python eda.py         # produces charts in outputs/
```

### Step 2 — Train the Model
```bash
python train.py
```
This will:
- Generate a synthetic dataset (or swap in the Kaggle CSV)
- Scale features with `StandardScaler`
- Balance classes with **SMOTE**
- Train Logistic Regression and Random Forest
- Print F1 score, ROC-AUC, classification report, and confusion matrix
- Save the best model to `models/fraud_model.pkl`

### Step 3 — Predict
```bash
# Run demo with example transactions
python predict.py

# Score your own CSV
python predict.py --csv path/to/your/transactions.csv
```

---

## Features Used

| Feature | Description |
|---|---|
| `amount` | Transaction amount in USD |
| `hour` | Hour of day (0–23) |
| `v1` – `v5` | PCA-transformed behavioral features |
| `n_transactions_last_hour` | Velocity feature — number of recent transactions |

---

## ML Techniques

| Technique | Why |
|---|---|
| **StandardScaler** | Normalizes features so no single one dominates |
| **SMOTE** | Synthetically oversamples minority class (fraud) to fix imbalance |
| **Logistic Regression** | Fast, interpretable baseline |
| **Random Forest** | Handles non-linear patterns, robust to noise |
| **F1 Score + ROC-AUC** | Better metrics than accuracy for imbalanced data |

---

##  Sample Results

```
Model: Random Forest
F1 Score : 0.91
ROC-AUC  : 0.97

              precision    recall  f1-score
Legit             0.99      1.00      1.00
Fraud             0.89      0.85      0.87
```

---

##  Using Real Data (Kaggle Dataset)

Replace the synthetic data with the real Kaggle dataset for better results:

1. Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Place `creditcard.csv` in the `data/` folder
3. In `train.py`, replace the `generate_dataset()` call with:

```python
df = pd.read_csv("data/creditcard.csv")
df.rename(columns={"Time": "hour"}, inplace=True)
df["hour"] = (df["hour"] // 3600) % 24
```

---



