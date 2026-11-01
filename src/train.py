"""
DeepShield AI - Model Training and Validation Pipeline
Executes complete deep learning training with validation tracking,
best checkpoint preservation, learning rate scheduling, and curve plotting.
"""

import os
import sys

# Ensure repository root is in python search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time
import argparse
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.utils import (
    load_config,
    set_seed,
    get_device,
    save_checkpoint,
    plot_training_curves,
    setup_logger
)
from src.model import build_model
from src.dataset import create_dataloaders

logger = setup_logger("DeepShieldAI.Train")


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Dict[str, float]:
    """Runs a single training epoch across batches."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Training Batch", leave=False)
    for images, labels, _ in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return {"loss": epoch_loss, "accuracy": epoch_acc}


def validate_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Evaluates the model on validation data without updating gradients."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Validation Batch", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return {"loss": epoch_loss, "accuracy": epoch_acc}


def run_training(
    config: Dict[str, Any],
    custom_epochs: Optional[int] = None,
    custom_batch_size: Optional[int] = None,
    custom_lr: Optional[float] = None,
    device_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main training routine.
    """
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    prep_cfg = config.get("preprocessing", {})
    ds_cfg = config.get("dataset", {})

    seed = train_cfg.get("random_seed", 42)
    set_seed(seed)

    device = get_device(device_name)
    epochs = custom_epochs if custom_epochs is not None else train_cfg.get("num_epochs", 10)
    lr = custom_lr if custom_lr is not None else float(train_cfg.get("learning_rate", 0.0001))
    weight_decay = float(train_cfg.get("weight_decay", 0.0001))
    patience = train_cfg.get("early_stopping_patience", 4)
    checkpoint_dir = train_cfg.get("checkpoint_dir", "models")
    checkpoint_name = train_cfg.get("checkpoint_name", "best_model.pth")

    if custom_batch_size is not None:
        config["training"]["batch_size"] = custom_batch_size

    logger.info(f"Initializing DataLoaders for dataset at: {ds_cfg.get('root_dir', 'data')}")
    train_loader, val_loader, _ = create_dataloaders(config)

    if train_loader is None or len(train_loader.dataset) == 0:
        raise RuntimeError("Training dataset is empty or directory was not found. Please verify dataset layout.")

    if val_loader is None or len(val_loader.dataset) == 0:
        logger.warning("Validation loader is empty! Validation will be skipped or mirrored.")

    model = build_model(config)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    logger.info(f"--- Starting Training on {device} ({epochs} Epochs, Batch Size={config['training']['batch_size']}, LR={lr}) ---")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])

        if val_loader is not None and len(val_loader.dataset) > 0:
            val_metrics = validate_one_epoch(model, val_loader, criterion, device)
            scheduler.step(val_metrics["loss"])
        else:
            val_metrics = {"loss": train_metrics["loss"], "accuracy": train_metrics["accuracy"]}

        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])

        epoch_duration = time.time() - epoch_start
        logger.info(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_duration:.1f}s) - "
            f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']*100:.2f}% | "
            f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']*100:.2f}%"
        )

        # Checkpoint if validation loss improves
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch
            patience_counter = 0

            checkpoint_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "class_to_idx": ds_cfg.get("class_to_idx", {"real": 0, "fake": 1}),
                "architecture": model_cfg.get("architecture", "resnet18"),
                "preprocessing": prep_cfg,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            save_checkpoint(checkpoint_state, checkpoint_dir=checkpoint_dir, checkpoint_name=checkpoint_name)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}. Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}.")
                break

    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time/60:.2f} minutes. Best Epoch: {best_epoch} with Val Acc: {best_val_acc*100:.2f}%")

    # Save training history and plots
    os.makedirs("outputs", exist_ok=True)
    history_file = os.path.join("outputs", "training_history.json")
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)
    logger.info(f"Training history saved to {history_file}")

    plot_training_curves(history, output_dir="outputs")

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "total_time_seconds": total_time,
        "history": history
    }


def main():
    parser = argparse.ArgumentParser(description="Train DeepShield AI ResNet-18 Model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default=None, help="Force 'cuda' or 'cpu'")
    parser.add_argument("--data_dir", type=str, default=None, help="Override dataset root directory")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.data_dir:
        config["dataset"]["root_dir"] = args.data_dir
        config["dataset"]["train_dir"] = os.path.join(args.data_dir, "train")
        config["dataset"]["val_dir"] = os.path.join(args.data_dir, "val")
        config["dataset"]["test_dir"] = os.path.join(args.data_dir, "test")

    try:
        run_training(
            config,
            custom_epochs=args.epochs,
            custom_batch_size=args.batch_size,
            custom_lr=args.lr,
            device_name=args.device
        )
    except Exception as e:
        logger.error(f"Training execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
