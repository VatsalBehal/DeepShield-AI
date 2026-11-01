# DeepShield AI — Trained Model Checkpoints Directory

This directory stores serialized PyTorch model checkpoints and associated metadata.

## Expected File
- `best_model.pth`: The primary model checkpoint containing trained weights for ResNet-18.

## Checkpoint Structure
The `.pth` file is saved as a Python dictionary containing:
```python
{
    "epoch": int,                 # Epoch number with lowest validation loss
    "model_state_dict": dict,     # Learned weights of ResNet-18 layers
    "optimizer_state_dict": dict, # AdamW optimizer state
    "val_loss": float,            # Validation loss achieved
    "val_accuracy": float,        # Validation accuracy achieved
    "class_to_idx": {"real": 0, "fake": 1},
    "architecture": "resnet18",
    "preprocessing": {...},       # Normalization and image size parameters
    "timestamp": str
}
```

## How to Obtain or Generate `best_model.pth`
1. **Google Colab Training (Recommended):**
   - Open and execute `notebooks/DeepShield_AI_Training.ipynb` on a free T4 GPU.
   - Run the final cell to download `best_model.pth`.
   - Place `best_model.pth` in this directory: `D:\DEEPFAKE_DETECTION\models\best_model.pth`.

2. **Pipeline Smoke Test Fixture (Immediate UI Testing):**
   - Run `python scripts/smoke_test.py`.
   - A verified test fixture will be created automatically to allow testing the CLI and Streamlit interfaces without waiting for Colab training.
