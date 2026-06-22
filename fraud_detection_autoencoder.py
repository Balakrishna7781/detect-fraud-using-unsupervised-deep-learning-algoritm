"""
fraud_detection_autoencoder.py
 
Hands-On Assignment 4 - Advanced Artificial Intelligence (MSCS-633-A01)
Use Unsupervised Deep Learning Algorithm to Detect Fraud with PyOD
 
This script trains an unsupervised AutoEncoder (PyOD) on the Kaggle
"Credit Card Fraud Detection" dataset to flag anomalous (fraudulent)
transactions based on reconstruction error, without using fraud labels
during training.
 
Dataset (download separately, not included in this repo):
    https://www.kaggle.com/datasets/whenamancodes/fraud-detection
    File expected: creditcard.csv
    Place it in the same directory as this script, or pass a path with
    --data-path.
 
Author: Bala
Course: MSCS-633-A01, University of the Cumberlands
"""
 
import argparse
import os
import sys
import time
 
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # allows saving plots without a display/GUI backend
import matplotlib.pyplot as plt
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    RocCurveDisplay,
)
 
from pyod.models.auto_encoder import AutoEncoder
 
 
def parse_args():
    """Parse command-line arguments so the script is configurable without
    editing source code (a coding best practice for reproducible experiments).
    """
    parser = argparse.ArgumentParser(
        description="Unsupervised fraud detection using PyOD AutoEncoder."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="creditcard.csv",
        help="Path to the Kaggle creditcard.csv file (default: ./creditcard.csv)",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.0017,
        help="Expected proportion of outliers in the dataset. The Kaggle "
        "credit card dataset has ~0.17%% fraud, so 0.0017 is a sensible "
        "default (default: 0.0017)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs for the AutoEncoder (default: 30)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size (default: 64)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save plots and reports (default: ./outputs)",
    )
    return parser.parse_args()
 
 
def load_data(data_path):
    """Load the Kaggle credit card transactions dataset.
 
    The dataset contains 284,807 transactions with 28 PCA-anonymized
    features (V1-V28), plus 'Time', 'Amount', and the ground-truth label
    'Class' (1 = fraud, 0 = legitimate). The label is used only for
    evaluation after training, never for fitting the unsupervised model.
    """
    if not os.path.exists(data_path):
        sys.exit(
            f"ERROR: Could not find dataset at '{data_path}'.\n"
            "Download creditcard.csv from "
            "https://www.kaggle.com/datasets/whenamancodes/fraud-detection "
            "and place it in this directory, or pass --data-path."
        )
 
    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud cases: {df['Class'].sum()} "
          f"({100 * df['Class'].mean():.3f}% of all transactions)")
    return df
 
 
def preprocess(df, random_state):
    """Scale features and split into train/test sets.
 
    'Amount' and 'Time' are scaled because their ranges are very different
    from the PCA-transformed V1-V28 features, which are already roughly
    standardized. The AutoEncoder is trained ONLY on legitimate-looking
    data behavior in an unsupervised fashion (it does not see the 'Class'
    label during fit), consistent with real-world fraud detection where
    most transactions are unlabeled.
    """
    features = df.drop(columns=["Class"])
    labels = df["Class"]
 
    scaler = StandardScaler()
    features[["Time", "Amount"]] = scaler.fit_transform(features[["Time", "Amount"]])
 
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=random_state,
        stratify=labels,  # preserve the fraud/legit ratio in both splits
    )
 
    print(f"Training set: {X_train.shape[0]} transactions")
    print(f"Test set: {X_test.shape[0]} transactions "
          f"({y_test.sum()} fraud cases)")
 
    return X_train, X_test, y_train, y_test
 
 
def build_and_train_model(X_train, contamination, epochs, batch_size, random_state):
    """Build and fit the PyOD AutoEncoder.
 
    AutoEncoder learns to compress and reconstruct the input feature
    vectors. Because it is trained predominantly on legitimate transaction
    patterns, it reconstructs legitimate transactions well (low error) and
    struggles to reconstruct rare, anomalous transactions (high error).
    That reconstruction error becomes the anomaly score.
    """
    clf = AutoEncoder(
        contamination=contamination,
        hidden_neuron_list=[32, 16, 8, 16, 32],  # symmetric encoder/decoder
        hidden_activation_name="relu",
        epoch_num=epochs,
        batch_size=batch_size,
        random_state=random_state,
        verbose=1,
    )
 
    print("\nTraining AutoEncoder (this may take a few minutes)...")
    start = time.time()
    clf.fit(X_train)
    elapsed = time.time() - start
    print(f"Training completed in {elapsed:.1f} seconds.")
 
    return clf
 
 
def evaluate_model(clf, X_test, y_test, output_dir):
    """Evaluate the trained AutoEncoder against held-out, labeled data.
 
    Labels are used here ONLY for evaluation (computing precision, recall,
    ROC-AUC, etc.), never for training, preserving the unsupervised nature
    of the method.
    """
    os.makedirs(output_dir, exist_ok=True)
 
    # decision_function returns the raw reconstruction-error anomaly score
    y_scores = clf.decision_function(X_test)
    # predict() applies the contamination-derived threshold -> 0/1 labels
    y_pred = clf.predict(X_test)
 
    print("\n=== Classification Report (0 = legitimate, 1 = fraud) ===")
    report = classification_report(y_test, y_pred, digits=4)
    print(report)
 
    cm = confusion_matrix(y_test, y_pred)
    print("=== Confusion Matrix ===")
    print(cm)
 
    roc_auc = roc_auc_score(y_test, y_scores)
    pr_auc = average_precision_score(y_test, y_scores)
    print(f"\nROC-AUC: {roc_auc:.4f}")
    print(f"Average Precision (PR-AUC): {pr_auc:.4f}")
 
    # Save a text summary of results for the report/screenshot deliverable
    report_path = os.path.join(output_dir, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("PyOD AutoEncoder Fraud Detection - Evaluation Report\n")
        f.write("=" * 55 + "\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}\n")
        f.write(f"Average Precision (PR-AUC): {pr_auc:.4f}\n")
    print(f"Saved evaluation report to: {report_path}")
 
    # Plot and save ROC curve
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, y_scores, ax=ax)
    ax.set_title("ROC Curve - AutoEncoder Fraud Detection")
    roc_path = os.path.join(output_dir, "roc_curve.png")
    fig.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ROC curve plot to: {roc_path}")
 
    # Plot and save anomaly score distribution by class
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(y_scores[y_test == 0], bins=50, alpha=0.6, label="Legitimate", density=True)
    ax.hist(y_scores[y_test == 1], bins=50, alpha=0.6, label="Fraud", density=True)
    ax.axvline(clf.threshold_, color="red", linestyle="--", label="Decision threshold")
    ax.set_xlabel("Reconstruction error (anomaly score)")
    ax.set_ylabel("Density")
    ax.set_title("Anomaly Score Distribution by Class")
    ax.legend()
    dist_path = os.path.join(output_dir, "anomaly_score_distribution.png")
    fig.savefig(dist_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved anomaly score distribution plot to: {dist_path}")
 
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
    }
 
 
def main():
    args = parse_args()
 
    print("=" * 60)
    print("Fraud Detection with PyOD AutoEncoder (Unsupervised)")
    print("=" * 60)
 
    df = load_data(args.data_path)
    X_train, X_test, y_train, y_test = preprocess(df, args.random_state)
 
    clf = build_and_train_model(
        X_train,
        contamination=args.contamination,
        epochs=args.epochs,
        batch_size=args.batch_size,
        random_state=args.random_state,
    )
 
    evaluate_model(clf, X_test, y_test, args.output_dir)
 
if __name__ == "__main__":
    main()
 
