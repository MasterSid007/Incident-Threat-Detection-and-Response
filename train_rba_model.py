"""
Train the model on the RBA dataset using a proper TEMPORAL split.

- TRAIN: Feb 3-4, 2020 (earliest data)
- TEST:  Feb 5+, 2020 (held out, never seen during training)

The FeatureExtractor is fit ONLY on training data to prevent label leakage
through ip_attack_rate, asn_attack_rate, etc.
"""
import sys
import os
sys.path.insert(0, 'detection')
from etl import LogLoader
from features import FeatureExtractor
from models import SupervisedAttackClassifier
import pandas as pd
import numpy as np
import joblib
import json

SPLIT_DATE = "2020-02-05"  # Everything before this = train, from this = test


def train_and_save():
    print("=" * 60)
    print("ITDR Model Training — Temporal Split")
    print("=" * 60)

    print(f"\nLoading RBA dataset...")
    loader = LogLoader('rba-dataset.csv')
    df = loader.load_to_dataframe(nrows=200000)
    print(f"Loaded {len(df)} events.")

    # ── TEMPORAL SPLIT ──────────────────────────────────────
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    cutoff = pd.Timestamp(SPLIT_DATE)

    train_df = df[df['timestamp'] < cutoff].copy()
    test_df = df[df['timestamp'] >= cutoff].copy()

    print(f"\n--- Temporal Split ---")
    print(f"Train: {len(train_df):,} events  ({train_df['timestamp'].min().date()} to {train_df['timestamp'].max().date()})")
    print(f"Test:  {len(test_df):,} events  ({test_df['timestamp'].min().date()} to {test_df['timestamp'].max().date()})")
    print(f"Train attacks: {train_df['is_attack'].sum():,}  ({train_df['is_attack'].mean()*100:.1f}%)")
    print(f"Test  attacks: {test_df['is_attack'].sum():,}  ({test_df['is_attack'].mean()*100:.1f}%)")

    # Verify no temporal leakage
    assert train_df['timestamp'].max() < test_df['timestamp'].min(), "Temporal leak!"

    # ── FIT FEATURE EXTRACTOR ON TRAIN ONLY ─────────────────
    print(f"\nFitting feature extractor on TRAINING data only...")
    extractor = FeatureExtractor()
    extractor.fit(train_df)
    
    X_train = extractor.transform(train_df)
    y_train = train_df['is_attack'].fillna(False).astype(bool)

    # ── TRAIN MODEL ON TRAIN SPLIT ──────────────────────────
    print(f"Training model on {len(X_train):,} training samples...")
    classifier = SupervisedAttackClassifier(n_estimators=200)
    classifier.train(X_train, y_train)

    # ── EVALUATE ON HELD-OUT TEST SPLIT ─────────────────────
    print(f"\nEvaluating on {len(test_df):,} TEST samples (never seen during training)...")
    X_test = extractor.transform(test_df)
    y_test = test_df['is_attack'].fillna(False).astype(int)

    results = classifier.predict(X_test)
    y_pred = results['supervised_pred'].astype(int)
    probs = results['attack_probability']

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'=' * 60}")
    print(f"TEST SET RESULTS (LEAK-FREE)")
    print(f"{'=' * 60}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Confusion Matrix:")
    print(f"  TN={cm[0][0]:,}  FP={cm[0][1]:,}")
    print(f"  FN={cm[1][0]:,}  TP={cm[1][1]:,}")
    fpr = cm[0][1] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
    print(f"False Positive Rate: {fpr:.4f}")

    # ── SAVE ARTIFACTS ──────────────────────────────────────
    os.makedirs('saved_models', exist_ok=True)
    classifier.save_model('saved_models/rba_trained_model.pkl')
    joblib.dump(extractor, 'saved_models/feature_extractor.pkl')

    # Save split metadata so the dashboard knows
    metadata = {
        "split_date": SPLIT_DATE,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_date_range": [str(train_df['timestamp'].min().date()), str(train_df['timestamp'].max().date())],
        "test_date_range": [str(test_df['timestamp'].min().date()), str(test_df['timestamp'].max().date())],
        "test_metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "fpr": round(fpr, 4),
        }
    }
    with open('saved_models/split_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved: rba_trained_model.pkl, feature_extractor.pkl, split_metadata.json")
    print("Done!")


if __name__ == "__main__":
    train_and_save()
