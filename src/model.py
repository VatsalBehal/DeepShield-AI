"""
DeepShield AI - Deep Learning Model Architecture
Implements ResNet-18 with transfer learning for binary classification (REAL vs FAKE).
Provides customizable classification head with dropout regularization and freezing options.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

from src.utils import setup_logger

logger = setup_logger("DeepShieldAI.Model")


class DeepShieldResNet18(nn.Module):
    """
    Binary deepfake classifier using ResNet-18 feature extractor
    with a specialized regularization and classification head.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = False
    ):
        super(DeepShieldResNet18, self).__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Initialize ResNet-18 with pretrained ImageNet weights
        if pretrained:
            try:
                weights = ResNet18_Weights.DEFAULT
                self.backbone = models.resnet18(weights=weights)
                logger.info("Loaded pretrained ResNet-18 ImageNet weights.")
            except Exception as e:
                logger.warning(f"Could not download online weights ({e}). Initializing without pretrained weights.")
                self.backbone = models.resnet18(weights=None)
        else:
            self.backbone = models.resnet18(weights=None)

        if freeze_backbone:
            self.freeze_features()

        # ResNet-18 fc input has 512 channels
        in_features = self.backbone.fc.in_features

        # Replace classification head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def freeze_features(self) -> None:
        """Freezes all layers except the final classification head."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Ensure fc layer parameters remain trainable
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
        logger.info("Backbone feature extractor frozen; only classification head is trainable.")

    def unfreeze_features(self) -> None:
        """Unfreezes all model parameters for full end-to-end fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("All layers unfrozen for end-to-end fine-tuning.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
        Returns:
            Raw unnormalized logits of shape (batch_size, num_classes)
        """
        return self.backbone(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference forward pass returning calibrated softmax probabilities.
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
        Returns:
            Probabilities of shape (batch_size, num_classes) summing to 1.0.
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)


def build_model(config: Optional[Dict[str, Any]] = None) -> DeepShieldResNet18:
    """
    Factory function to instantiate the DeepShield model from configuration.
    """
    model_cfg = config.get("model", {}) if config else {}
    num_classes = model_cfg.get("num_classes", 2)
    pretrained = model_cfg.get("pretrained", True)
    dropout_rate = model_cfg.get("dropout_rate", 0.3)
    freeze_backbone = model_cfg.get("freeze_backbone", False)

    model = DeepShieldResNet18(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone
    )
    return model
