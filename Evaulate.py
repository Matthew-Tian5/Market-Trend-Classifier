import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from config import TICKERS
from train import load_feature_matrix

def evaluate():
    df       = load_feature_matrix(TICKERS)
    artifact = joblib.load("model.pkl")
    model    = artifact["model"]
    features = artifact["features"]
 
    X = df[features].fillna(0).values
    y = df["label"].values
 
    # Cross-validated predictions — each sample predicted on a held-out fold
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X, y, cv=cv)
 
    print("\n── Classification Report ──────────────────────────")
    print(classification_report(y, y_pred, target_names=["Down (0)", "Up (1)"]))
 
    f1 = f1_score(y, y_pred)
    print(f"Directional F1: {f1:.4f}")
 
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Pred Down", "Pred Up"],
        yticklabels=["Actual Down", "Actual Up"]
    )
    plt.title(f"Confusion Matrix  |  F1: {f1:.4f}")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("Saved → confusion_matrix.png")
 