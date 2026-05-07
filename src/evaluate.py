"""
src/evaluate.py  —  Model evaluation, metrics, confusion matrix, ROC curve.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score
)

from src.config import CONFIG


def evaluate(model, X_test, y_test, save_dir="results"):
    """
    Full evaluation suite:
      - Accuracy, AUC, Precision, Recall, F1
      - Confusion matrix (counts + normalized)
      - ROC curve
      - Classification report
    Returns dict of metrics.
    """
    os.makedirs(save_dir, exist_ok=True)

    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= CONFIG["threshold"]).astype(int)

    test_loss, test_acc, test_auc, test_prec, test_rec = model.evaluate(
        X_test, y_test, verbose=0
    )
    f1 = f1_score(y_test, y_pred)

    print("\n" + "="*55)
    print("  TEST SET EVALUATION")
    print("="*55)
    print(f"  Accuracy   : {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"  AUC-ROC    : {test_auc:.4f}")
    print(f"  Precision  : {test_prec:.4f}")
    print(f"  Recall     : {test_rec:.4f}")
    print(f"  F1 Score   : {f1:.4f}")
    print(f"  Loss       : {test_loss:.4f}")
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=CONFIG["classes"]))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=CONFIG["classes"], yticklabels=CONFIG["classes"])
    axes[0].set_title("Confusion Matrix (Counts)", fontweight="bold")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")

    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=axes[1],
                xticklabels=CONFIG["classes"], yticklabels=CONFIG["classes"],
                vmin=0, vmax=1)
    axes[1].set_title("Confusion Matrix (Normalized)", fontweight="bold")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#2196F3", linewidth=2.5,
             label=f"ROC (AUC = {auc_val:.4f})")
    plt.plot([0,1],[0,1], color="gray", linestyle="--")
    plt.fill_between(fpr, tpr, alpha=0.07, color="#2196F3")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Malaria Classifier", fontweight="bold")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/roc_curve.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[EVAL] Plots saved to {save_dir}/")

    return {
        "accuracy": test_acc, "auc": test_auc,
        "precision": test_prec, "recall": test_rec,
        "f1": f1, "loss": test_loss
    }
