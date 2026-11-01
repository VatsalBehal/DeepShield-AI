"""
DeepShield AI - Unit Tests for ResNet-18 Architecture and Checkpoint Serialization
"""

import os
import pytest
import torch

from src.model import DeepShieldResNet18, build_model
from src.utils import save_checkpoint, load_checkpoint


def test_model_instantiation():
    model = DeepShieldResNet18(num_classes=2, pretrained=False)
    assert model is not None
    assert isinstance(model, torch.nn.Module)


def test_model_forward_shape():
    model = DeepShieldResNet18(num_classes=2, pretrained=False)
    model.eval()
    dummy_input = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        logits = model(dummy_input)
    assert logits.shape == (4, 2)


def test_predict_proba_sums_to_one():
    model = DeepShieldResNet18(num_classes=2, pretrained=False)
    model.eval()
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        probs = model.predict_proba(dummy_input)

    assert probs.shape == (2, 2)
    # Check that probabilities sum to 1.0 per sample
    sums = torch.sum(probs, dim=1).numpy()
    for s in sums:
        assert abs(s - 1.0) < 1e-4


def test_freeze_and_unfreeze_features():
    model = DeepShieldResNet18(num_classes=2, pretrained=False)
    model.freeze_features()
    # Check that backbone layer 1 is frozen
    for p in model.backbone.conv1.parameters():
        assert p.requires_grad is False
    # Check that classification head is trainable
    for p in model.backbone.fc.parameters():
        assert p.requires_grad is True

    model.unfreeze_features()
    for p in model.backbone.conv1.parameters():
        assert p.requires_grad is True


def test_checkpoint_roundtrip(tmp_path):
    model = DeepShieldResNet18(num_classes=2, pretrained=False)
    ckpt_dir = str(tmp_path)
    state = {
        "epoch": 5,
        "model_state_dict": model.state_dict(),
        "val_loss": 0.42,
        "val_accuracy": 0.85
    }
    saved_path = save_checkpoint(state, checkpoint_dir=ckpt_dir, checkpoint_name="test_ckpt.pth")
    assert os.path.exists(saved_path)

    new_model = DeepShieldResNet18(num_classes=2, pretrained=False)
    loaded_model, _, meta = load_checkpoint(saved_path, new_model, device=torch.device("cpu"))

    assert meta["epoch"] == 5
    assert meta["val_accuracy"] == 0.85
