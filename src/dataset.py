"""
DeepShield AI - Dataset Loading, Video Splitting, and Leakage Prevention
Handles structured folder loading (train/val/test with real/fake subfolders),
manifest-based loading, dataset integrity verification, and video-level dataset splitting.
"""

import os
import random
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from src.preprocessing import load_image_safe, get_train_transforms, get_eval_transforms
from src.utils import setup_logger

logger = setup_logger("DeepShieldAI.Dataset")

# Standard image extensions supported
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class DeepfakeDataset(Dataset):
    """
    PyTorch Dataset for binary deepfake detection (REAL=0, FAKE=1).
    Can be initialized either from a root split directory (containing 'real' and 'fake' folders)
    or from a manifest CSV containing 'filepath' and 'label'.
    """

    def __init__(
        self,
        split_dir: Optional[str] = None,
        manifest_path: Optional[str] = None,
        transform=None,
        class_to_idx: Optional[Dict[str, int]] = None
    ):
        self.samples: List[Tuple[str, int]] = []
        self.transform = transform
        self.class_to_idx = class_to_idx or {"real": 0, "fake": 1}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        if manifest_path and os.path.exists(manifest_path):
            self._load_from_manifest(manifest_path)
        elif split_dir and os.path.exists(split_dir):
            self._load_from_directory(split_dir)
        else:
            logger.warning(f"No valid dataset found at split_dir='{split_dir}' or manifest='{manifest_path}'.")

        logger.info(f"Loaded {len(self.samples)} valid samples (Classes: {self.get_class_counts()})")

    def _load_from_directory(self, split_dir: str) -> None:
        """Loads samples from standard subdirectories (real/ and fake/)."""
        for class_name, class_idx in self.class_to_idx.items():
            class_folder = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_folder):
                continue

            for root, _, files in os.walk(class_folder):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in VALID_IMAGE_EXTENSIONS:
                        full_path = os.path.join(root, file)
                        self.samples.append((full_path, class_idx))

    def _load_from_manifest(self, manifest_path: str) -> None:
        """Loads samples from a CSV manifest."""
        df = pd.read_csv(manifest_path)
        for _, row in df.iterrows():
            img_path = str(row["filepath"])
            label_val = row["label"]
            if isinstance(label_val, str):
                class_idx = self.class_to_idx.get(label_val.lower(), 1)
            else:
                class_idx = int(label_val)

            if os.path.exists(img_path):
                self.samples.append((img_path, class_idx))

    def get_class_counts(self) -> Dict[str, int]:
        """Returns distribution of classes in the current dataset split."""
        counts = {name: 0 for name in self.class_to_idx.keys()}
        for _, label_idx in self.samples:
            class_name = self.idx_to_class.get(label_idx, "unknown")
            counts[class_name] = counts.get(class_name, 0) + 1
        return counts

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        img_path, label = self.samples[idx]
        try:
            pil_img = load_image_safe(img_path)
        except Exception as e:
            logger.error(f"Failed to read image at {img_path}: {e}. Returning blank placeholder.")
            pil_img = Image.new("RGB", (224, 224), color=(0, 0, 0))

        if self.transform is not None:
            image_tensor = self.transform(pil_img)
        else:
            default_t = get_eval_transforms()
            image_tensor = default_t(pil_img)

        return image_tensor, label, img_path


def split_video_dataset_without_leakage(
    video_manifest: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, List[Dict[str, Any]]]:
    """
    CRITICAL LEAKAGE PREVENTION:
    Splits video IDs before frame extraction so all frames from the same source
    video (and its manipulated pairs) remain strictly in the same split.

    Args:
        video_manifest: List of dicts, each with 'video_id', 'label', 'filepath'
        train_ratio: Ratio for training set (e.g., 0.70)
        val_ratio: Ratio for validation set (e.g., 0.15)
        test_ratio: Ratio for test set (e.g., 0.15)
        seed: Random seed for deterministic reproducibility

    Returns:
        Dictionary with keys 'train', 'val', 'test' containing lists of video records.
    """
    random.seed(seed)

    # Group video records by base video identity
    grouped_by_identity: Dict[str, List[Dict[str, Any]]] = {}
    for item in video_manifest:
        vid_id = str(item.get("video_id", item.get("filepath", "")))
        # Strip frame tags or manipulation suffixes if present
        base_id = vid_id.split("_")[0]
        grouped_by_identity.setdefault(base_id, []).append(item)

    unique_identities = list(grouped_by_identity.keys())
    random.shuffle(unique_identities)

    total = len(unique_identities)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_ids = set(unique_identities[:train_end])
    val_ids = set(unique_identities[train_end:val_end])
    test_ids = set(unique_identities[val_end:])

    splits = {"train": [], "val": [], "test": []}
    for base_id, records in grouped_by_identity.items():
        if base_id in train_ids:
            splits["train"].extend(records)
        elif base_id in val_ids:
            splits["val"].extend(records)
        else:
            splits["test"].extend(records)

    logger.info(
        f"Video leakage-free split complete: "
        f"Train={len(splits['train'])} videos, "
        f"Val={len(splits['val'])} videos, "
        f"Test={len(splits['test'])} videos across {total} unique identities."
    )
    return splits


def create_dataloaders(
    config: Dict[str, Any]
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """
    Instantiates PyTorch DataLoaders for train, val, and test splits
    using settings configured in config.yaml.
    """
    dataset_cfg = config.get("dataset", {})
    prep_cfg = config.get("preprocessing", {})
    train_cfg = config.get("training", {})

    image_size = prep_cfg.get("image_size", 224)
    mean = tuple(prep_cfg.get("normalize_mean", [0.485, 0.456, 0.406]))
    std = tuple(prep_cfg.get("normalize_std", [0.229, 0.224, 0.225]))

    train_transform = get_train_transforms(image_size, mean, std)
    eval_transform = get_eval_transforms(image_size, mean, std)

    batch_size = train_cfg.get("batch_size", 16)
    num_workers = train_cfg.get("num_workers", 0)

    train_loader, val_loader, test_loader = None, None, None

    train_dir = dataset_cfg.get("train_dir", "data/train")
    if os.path.isdir(train_dir):
        train_ds = DeepfakeDataset(split_dir=train_dir, transform=train_transform)
        if len(train_ds) > 0:
            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )

    val_dir = dataset_cfg.get("val_dir", "data/val")
    if os.path.isdir(val_dir):
        val_ds = DeepfakeDataset(split_dir=val_dir, transform=eval_transform)
        if len(val_ds) > 0:
            val_loader = DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )

    test_dir = dataset_cfg.get("test_dir", "data/test")
    if os.path.isdir(test_dir):
        test_ds = DeepfakeDataset(split_dir=test_dir, transform=eval_transform)
        if len(test_ds) > 0:
            test_loader = DataLoader(
                test_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )

    return train_loader, val_loader, test_loader
