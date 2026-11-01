"""
DeepShield AI - Utility Functions and Helper Routines
Provides configuration loading, reproducibility seeding, device management,
checkpoint serialization, and metric visualization.
"""

import os
import random
import logging
from typing import Dict, Any, Optional, Tuple
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt


def setup_logger(name: str = "DeepShieldAI", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a consistent logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s", datefmt="%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = setup_logger()


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Safely loads YAML configuration file with error handling and fallback defaults.
    """
    if not os.path.exists(config_path):
        logger.warning(f"Configuration file not found at {config_path}. Using built-in defaults.")
        return get_default_config()

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            if not isinstance(cfg, dict):
                logger.warning(f"Config at {config_path} is not a valid dictionary. Using default config.")
                return get_default_config()
            return cfg
    except Exception as e:
        logger.error(f"Failed to parse configuration from {config_path}: {e}")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Returns fallback configuration dictionary."""
    return {
        "project": {
            "name": "DeepShield AI",
            "version": "1.0.0"
        },
        "model": {
            "architecture": "resnet18",
            "pretrained": True,
            "num_classes": 2,
            "dropout_rate": 0.3,
            "freeze_backbone": false
        },
        "dataset": {
            "root_dir": "data",
            "train_dir": "data/train",
            "val_dir": "data/val",
            "test_dir": "data/test",
            "class_names": ["real", "fake"],
            "class_to_idx": {"real": 0, "fake": 1}
        },
        "preprocessing": {
            "image_size": 224,
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
            "face_detection_enabled": True,
            "face_crop_margin": 0.20,
            "min_face_size": [40, 40],
            "cascade_scale_factor": 1.1,
            "cascade_min_neighbors": 5
        },
        "training": {
            "batch_size": 16,
            "num_epochs": 10,
            "learning_rate": 0.0001,
            "weight_decay": 0.0001,
            "num_workers": 0,
            "random_seed": 42,
            "checkpoint_dir": "models",
            "checkpoint_name": "best_model.pth",
            "early_stopping_patience": 4
        },
        "inference": {
            "default_checkpoint": "models/best_model.pth",
            "confidence_threshold": 0.50,
            "video": {
                "frame_interval": 10,
                "max_frames": 30,
                "aggregation_method": "mean"
            }
        }
    }


def set_seed(seed: int = 42) -> None:
    """
    Enforces deterministic reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Deterministic random seed set to: {seed}")


def get_device(preference: Optional[str] = None) -> torch.device:
    """
    Determines optimal computing hardware (CUDA GPU or CPU).
    """
    if preference:
        preference_clean = preference.lower().strip()
        if preference_clean == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif preference_clean == "cpu":
            return torch.device("cpu")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"NVIDIA GPU detected: {gpu_name}. Using CUDA acceleration.")
        return torch.device("cuda")
    else:
        logger.info("No compatible NVIDIA GPU found. Using standard CPU execution.")
        return torch.device("cpu")


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str = "models",
    checkpoint_name: str = "best_model.pth"
) -> str:
    """
    Safely serializes model weights along with extensive training metadata.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, checkpoint_name)
    torch.save(state, filepath)
    logger.info(f"Model checkpoint successfully saved to: {filepath}")
    return filepath


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], Dict[str, Any]]:
    """
    Loads checkpoint state dictionary and restores model and optimizer parameters.
    Handles legacy checkpoints and returns accompanying metadata.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found at: {filepath}")

    if device is None:
        device = get_device()

    try:
        # Load weights on specified device safely
        checkpoint = torch.load(filepath, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint file at {filepath}: {e}")

    metadata = {}
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # Structured checkpoint with metadata
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as opt_err:
                logger.warning(f"Could not restore optimizer state: {opt_err}")
        metadata = {k: v for k, v in checkpoint.items() if k not in ["model_state_dict", "optimizer_state_dict"]}
    elif isinstance(checkpoint, dict):
        # Raw state dictionary
        model.load_state_dict(checkpoint)
        metadata = {"epoch": "unknown", "val_acc": "unknown"}
    else:
        raise ValueError(f"Unrecognized checkpoint object structure in: {filepath}")

    model.to(device)
    model.eval()
    logger.info(f"Checkpoint successfully restored from: {filepath} onto {device}")
    return model, optimizer, metadata


def plot_training_curves(
    history: Dict[str, list],
    output_dir: str = "outputs",
    filename: str = "training_curves.png"
) -> str:
    """
    Plots training and validation loss and accuracy curves and saves as high-resolution image.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)

    epochs = range(1, len(history.get("train_loss", [])) + 1)
    if len(epochs) == 0:
        logger.warning("No history epochs available to plot.")
        return ""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(epochs, history.get("train_loss", []), "o-", label="Train Loss", color="#1f77b4", linewidth=2)
    axes[0].plot(epochs, history.get("val_loss", []), "s-", label="Val Loss", color="#d62728", linewidth=2)
    axes[0].set_title("Training and Validation Loss", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Cross Entropy Loss", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=10)

    # Accuracy plot
    axes[1].plot(epochs, [a * 100 for a in history.get("train_acc", [])], "o-", label="Train Acc (%)", color="#2ca02c", linewidth=2)
    axes[1].plot(epochs, [a * 100 for a in history.get("val_acc", [])], "s-", label="Val Acc (%)", color="#ff7f0e", linewidth=2)
    axes[1].set_title("Training and Validation Accuracy", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Accuracy (%)", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    logger.info(f"Training curves successfully saved to {out_path}")
    return out_path
