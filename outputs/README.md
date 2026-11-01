# DeepShield AI — Evaluation and Experimentation Outputs

This directory stores experimental results, metric logs, and high-resolution plots generated during training and evaluation.

## Generated Artifacts
- `training_history.json`: Epoch-by-epoch loss and accuracy values.
- `training_curves.png`: High-resolution dual-panel plot of training vs validation loss and accuracy.
- `evaluation_summary.json`: Detailed classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix breakdown).
- `classification_report.txt`: Standard Scikit-Learn per-class classification report.
- `confusion_matrix.png`: Heatmap visualising True Positives, True Negatives, False Positives, and False Negatives.
- `roc_curve.png`: Receiver Operating Characteristic curve with Area Under the Curve (AUC).

## Generating Evaluation Artifacts
Run the evaluation script on your held-out test split:
```powershell
python src/evaluate.py --checkpoint models/best_model.pth --test_dir data/test
```
