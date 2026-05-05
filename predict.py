"""
Credit Card Fraud Detection - Prediction Script
================================================
Loads a trained model and scores new transactions.
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import sys


FEATURES = ["amount", "hour", "v1", "v2", "v3", "v4", "v5", "n_transactions_last_hour"]


def load_artifacts():
    try:
        model  = joblib.load("models/fraud_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        print("❌  Model not found. Please run  train.py  first.")
        sys.exit(1)


def predict(transactions: pd.DataFrame, model, scaler) -> pd.DataFrame:
    """Score a DataFrame of transactions."""
    X = scaler.transform(transactions[FEATURES])
    probs  = model.predict_proba(X)[:, 1]
    labels = model.predict(X)

    result = transactions.copy()
    result["fraud_probability"] = probs.round(4)
    result["prediction"]        = ["🚨 FRAUD" if p == 1 else "✅ Legit" for p in labels]
    return result


def demo():
    """Run a quick demo with hand-crafted transactions."""
    transactions = pd.DataFrame([
        # amount   hour  v1    v2    v3    v4    v5    n_tx_last_hr
        [  45.00,   14,  0.1,  0.2, -0.1,  0.3, -0.1,  2],   # looks legit
        [ 980.50,    2, -3.5,  3.1, -2.8,  1.9, -1.2, 10],   # looks fraudulent
        [  12.99,   11,  0.5, -0.3,  0.2, -0.1,  0.4,  1],   # legit small purchase
        [ 450.00,   23, -2.9,  2.5, -1.8,  1.2, -0.8,  9],   # suspicious late-night
    ], columns=FEATURES)

    model, scaler = load_artifacts()
    results = predict(transactions, model, scaler)

    print("\n🔍  Fraud Detection — Prediction Results")
    print("=" * 60)
    for i, row in results.iterrows():
        print(f"\n  Transaction #{i+1}")
        print(f"    Amount : ${row['amount']:.2f}")
        print(f"    Hour   : {int(row['hour']):02d}:00")
        print(f"    Result : {row['prediction']}")
        print(f"    Fraud Probability: {row['fraud_probability']*100:.1f}%")
    print()


def from_csv(path: str):
    """Score transactions from a CSV file."""
    df = pd.read_csv(path)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        print(f"❌  Missing columns in CSV: {missing}")
        sys.exit(1)

    model, scaler = load_artifacts()
    results = predict(df, model, scaler)
    out_path = path.replace(".csv", "_predictions.csv")
    results.to_csv(out_path, index=False)
    print(f"✅  Predictions saved to: {out_path}")
    print(results[["amount", "hour", "fraud_probability", "prediction"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraud Detection — Inference")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to CSV file with transactions to score")
    args = parser.parse_args()

    if args.csv:
        from_csv(args.csv)
    else:
        demo()
