"""
Credit Card Fraud Detection - Training Script
==============================================
Trains multiple ML models and saves the best one.
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score
)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. GENERATE SYNTHETIC DATASET
# ─────────────────────────────────────────────

def generate_dataset(n_samples=10000, fraud_ratio=0.02, random_state=42):
    """
    Generates a synthetic credit card transaction dataset.
    In a real project, replace this with the Kaggle Credit Card dataset.
    Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    """
    np.random.seed(random_state)

    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # Legitimate transactions
    legit = pd.DataFrame({
        "amount":       np.random.exponential(scale=80, size=n_legit),
        "hour":         np.random.randint(0, 24, size=n_legit),
        "v1":           np.random.normal(0, 1, n_legit),
        "v2":           np.random.normal(0, 1, n_legit),
        "v3":           np.random.normal(0, 1, n_legit),
        "v4":           np.random.normal(0, 1, n_legit),
        "v5":           np.random.normal(0, 1, n_legit),
        "n_transactions_last_hour": np.random.poisson(3, n_legit),
        "Class":        0,
    })

    # Fraudulent transactions (different distribution)
    fraud = pd.DataFrame({
        "amount":       np.random.exponential(scale=300, size=n_fraud),
        "hour":         np.random.choice([1, 2, 3, 23], size=n_fraud),   # late night
        "v1":           np.random.normal(-3, 2, n_fraud),
        "v2":           np.random.normal(2, 2, n_fraud),
        "v3":           np.random.normal(-2, 2, n_fraud),
        "v4":           np.random.normal(1, 2, n_fraud),
        "v5":           np.random.normal(-1, 2, n_fraud),
        "n_transactions_last_hour": np.random.poisson(8, n_fraud),
        "Class":        1,
    })

    df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=random_state)
    df["amount"] = df["amount"].round(2).clip(lower=0.01)
    return df


# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(df):
    """Scales features and returns X, y."""
    FEATURES = ["amount", "hour", "v1", "v2", "v3", "v4", "v5", "n_transactions_last_hour"]
    X = df[FEATURES].copy()
    y = df["Class"].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=FEATURES)

    return X_scaled, y, scaler


# ─────────────────────────────────────────────
# 3. TRAIN MODELS
# ─────────────────────────────────────────────

def train_models(X_train, y_train):
    """Train Logistic Regression and Random Forest."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest":       RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
    }
    trained = {}
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        trained[name] = model
    return trained


# ─────────────────────────────────────────────
# 4. EVALUATE
# ─────────────────────────────────────────────

def evaluate(models, X_test, y_test):
    """Evaluate all models and return the best one by F1 score."""
    best_model, best_name, best_f1 = None, "", 0.0

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        f1  = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        print(f"\n{'='*50}")
        print(f"  Model : {name}")
        print(f"  F1    : {f1:.4f}")
        print(f"  ROC AUC: {auc:.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['Legit','Fraud'])}")
        print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

        if f1 > best_f1:
            best_f1, best_model, best_name = f1, model, name

    print(f"\n✅  Best model: {best_name}  (F1 = {best_f1:.4f})")
    return best_model, best_name


# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────

def main():
    print("\n🔍  Credit Card Fraud Detection — Training Pipeline")
    print("=" * 55)

    # Generate / load data
    print("\n[1/5] Generating dataset...")
    df = generate_dataset()
    df.to_csv("data/transactions.csv", index=False)
    print(f"      Rows: {len(df):,}  |  Fraud: {df['Class'].sum():,}  ({df['Class'].mean()*100:.1f}%)")

    # Preprocess
    print("\n[2/5] Preprocessing...")
    X, y, scaler = preprocess(df)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Handle class imbalance with SMOTE
    print("\n[3/5] Applying SMOTE to balance training data...")
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print(f"      Before SMOTE → Fraud: {y_train.sum()} | Legit: {(y_train==0).sum()}")
    print(f"      After  SMOTE → Fraud: {y_train_res.sum()} | Legit: {(y_train_res==0).sum()}")

    # Train
    print("\n[4/5] Training models...")
    models = train_models(X_train_res, y_train_res)

    # Evaluate
    print("\n[5/5] Evaluating on test set...")
    best_model, best_name = evaluate(models, X_test, y_test)

    # Save artefacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/fraud_model.pkl")
    joblib.dump(scaler,     "models/scaler.pkl")
    print(f"\n💾  Saved → models/fraud_model.pkl  &  models/scaler.pkl")
    print("\n✅  Training complete! Run predict.py to score new transactions.\n")


if __name__ == "__main__":
    main()
