"""
DeepShield AI - Model Evaluation and Performance Metrics
Evaluates model on held-out test set, generating Accuracy, Precision, Recall,
F1-Score, Confusion Matrix, Classification Report, and ROC-AUC curve.
"""

import os
import sys

# Ensure repository root is in python search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import argparse
from typing import Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score
)

from src.utils import load_config, get_device, load_checkpoint, setup_logger
from src.model import build_model
from src.dataset import DeepfakeDataset, get_eval_transforms

logger = setup_logger("DeepShieldAI.Evaluate")


def evaluate_model(
    checkpoint_path: str = "models/best_model.pth",
    test_dir: str = "data/test",
    config_path: str = "configs/config.yaml",
    output_dir: str = "outputs"
) -> Dict[str, Any]:
    """
    Evaluates trained checkpoint on held-out test data.
    """
    config = load_config(config_path)
    prep_cfg = config.get("preprocessing", {})
    image_size = prep_cfg.get("image_size", 224)
    mean = tuple(prep_cfg.get("normalize_mean", [0.485, 0.456, 0.406]))
    std = tuple(prep_cfg.get("normalize_std", [0.229, 0.224, 0.225]))

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}. Train the model before evaluating.")

    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found at: {test_dir}.")

    transform = get_eval_transforms(image_size, mean, std)
    test_dataset = DeepfakeDataset(split_dir=test_dir, transform=transform)

    if len(test_dataset) == 0:
        raise ValueError(f"No valid test samples found in {test_dir}. Please ensure test/real and test/fake contain images.")

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    device = get_device()
    model = build_model(config)
    model, _, metadata = load_checkpoint(checkpoint_path, model, device=device)
    model.eval()

    y_true = []
    y_pred = []
    y_probs_fake = []

    logger.info(f"Evaluating {len(test_dataset)} test samples on {device}...")

    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            # Class index 1 is 'fake', Class index 0 is 'real'
            preds = torch.argmax(probs, dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_probs_fake.extend(probs[:, 1].cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs_fake = np.array(y_probs_fake)

    # Calculate standard evaluation metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # ROC AUC calculation (requires both classes present)
    unique_classes = np.unique(y_true)
    has_both_classes = len(unique_classes) > 1

    roc_auc = None
    if has_both_classes:
        try:
            roc_auc = float(roc_auc_score(y_true, y_probs_fake))
        except Exception as e:
            logger.warning(f"Could not compute ROC-AUC: {e}")

    clf_report_dict = classification_report(
        y_true, y_pred,
        target_names=["REAL (Class 0)", "FAKE (Class 1)"],
        output_dict=True,
        zero_division=0
    )
    clf_report_text = classification_report(
        y_true, y_pred,
        target_names=["REAL (Class 0)", "FAKE (Class 1)"],
        zero_division=0
    )

    os.makedirs(output_dir, exist_ok=True)

    # 1. Plot and save Confusion Matrix heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred REAL", "Pred FAKE"],
        yticklabels=["True REAL", "True FAKE"],
        cbar=False
    )
    plt.title("Confusion Matrix (Held-out Test Set)", fontsize=13, fontweight="bold")
    plt.ylabel("Actual Label", fontsize=11)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # 2. Plot ROC Curve if applicable
    roc_path = None
    if has_both_classes:
        fpr, tpr, _ = roc_curve(y_true, y_probs_fake)
        roc_auc_val = auc(fpr, tpr)

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="#2b5c8f", lw=2, label=f"ROC curve (AUC = {roc_auc_val:.3f})")
        plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Chance (0.50)")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
        plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
        plt.title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold")
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        roc_path = os.path.join(output_dir, "roc_curve.png")
        plt.savefig(roc_path, dpi=300)
        plt.close()

    # 3. Save classification report text
    report_file = os.path.join(output_dir, "classification_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("DeepShield AI - Model Evaluation Report\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Test Set: {test_dir} ({len(y_true)} samples)\n\n")
        f.write(clf_report_text)
        if roc_auc is not None:
            f.write(f"\nROC-AUC Score: {roc_auc:.4f}\n")

    # 4. Save JSON metrics summary
    summary = {
        "num_test_samples": int(len(y_true)),
        "accuracy": float(acc),
        "precision_fake": float(prec),
        "recall_fake": float(rec),
        "f1_score_fake": float(f1),
        "roc_auc": roc_auc,
        "confusion_matrix": {
            "true_real_pred_real (TN)": int(cm[0, 0]),
            "true_real_pred_fake (FP)": int(cm[0, 1]),
            "true_fake_pred_real (FN)": int(cm[1, 0]),
            "true_fake_pred_fake (TP)": int(cm[1, 1])
        },
        "classification_report": clf_report_dict,
        "checkpoint_metadata": metadata
    }

    summary_file = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    logger.info("Evaluation complete! Results summary:")
    logger.info(f"Accuracy:  {acc*100:.2f}%")
    logger.info(f"Precision: {prec*100:.2f}%")
    logger.info(f"Recall:    {rec*100:.2f}%")
    logger.info(f"F1-Score:  {f1*100:.2f}%")
    if roc_auc is not None:
        logger.info(f"ROC-AUC:   {roc_auc:.4f}")
    logger.info(f"Outputs saved to directory: {output_dir}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate DeepShield AI Model on Test Set")
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pth", help="Path to checkpoint .pth")
    parser.add_argument("--test_dir", type=str, default="data/test", help="Path to test set directory")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for plots and reports")

    args = parser.parse_args()
    try:
        evaluate_model(
            checkpoint_path=args.checkpoint,
            test_dir=args.test_dir,
            config_path=args.config,
            output_dir=args.output_dir
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
