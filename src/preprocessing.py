"""
DeepShield AI - Image and Face Preprocessing Pipeline
Provides robust image loading, OpenCV face detection with bounding box margin expansion,
fallback handling for non-face images, and standard PyTorch transforms with ImageNet normalization.
"""

import os
from typing import Tuple, Optional, Union, Dict, Any
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError
import torch
from torchvision import transforms


class FaceDetector:
    """
    OpenCV Haar-Cascade face detector with bounding box margin expansion.
    """

    def __init__(
        self,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (40, 40),
        margin: float = 0.20
    ):
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.margin = margin

        # Locate OpenCV Haar Cascade file with local fallback
        candidates = [
            os.path.join(getattr(cv2.data, "haarcascades", ""), "haarcascade_frontalface_default.xml"),
            os.path.abspath("configs/haarcascade_frontalface_default.xml"),
            os.path.join(os.path.dirname(__file__), "..", "configs", "haarcascade_frontalface_default.xml")
        ]
        cascade_path = None
        for p in candidates:
            if os.path.exists(p):
                cascade_path = p
                break

        if cascade_path is None:
            raise FileNotFoundError("OpenCV face cascade file not found in cv2.data or configs/ directory.")
        self.detector = cv2.CascadeClassifier(cascade_path)

    def detect_and_crop(
        self,
        image_rgb: np.ndarray
    ) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]], bool]:
        """
        Detects the primary (largest) face in an RGB image and crops it with a margin.

        Returns:
            cropped_face: Cropped face RGB image, or full image if no face detected.
            bbox: (x, y, w, h) bounding box of detected face, or None.
            face_found: Boolean flag indicating whether a face was detected.
        """
        if image_rgb is None or image_rgb.size == 0:
            raise ValueError("Input image is empty or invalid.")

        h, w, _ = image_rgb.shape
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            # No face detected: return full image with flag False
            return image_rgb, None, False

        # Select the largest face by area (w * h)
        largest_face = max(faces, key=lambda b: b[2] * b[3])
        x, y, fw, fh = largest_face

        # Add configurable margin around the detected face
        margin_x = int(fw * self.margin)
        margin_y = int(fh * self.margin)

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w, x + fw + margin_x)
        y2 = min(h, y + fh + margin_y)

        cropped_face = image_rgb[y1:y2, x1:x2]
        return cropped_face, (x, y, fw, fh), True


def load_image_safe(image_source: Union[str, Image.Image, np.ndarray]) -> Image.Image:
    """
    Safely reads and converts an image into a PIL RGB Image.
    Validates file existence, format, and color channels.
    """
    if isinstance(image_source, str):
        if not os.path.exists(image_source):
            raise FileNotFoundError(f"Image file does not exist at: {image_source}")
        try:
            with Image.open(image_source) as img:
                # Force load bytes and convert to RGB
                rgb_img = img.convert("RGB")
                return rgb_img
        except UnidentifiedImageError:
            raise ValueError(f"Corrupt or unsupported image file: {image_source}")
        except Exception as e:
            raise RuntimeError(f"Error loading image from {image_source}: {e}")

    elif isinstance(image_source, Image.Image):
        return image_source.convert("RGB")

    elif isinstance(image_source, np.ndarray):
        if image_source.ndim == 2:
            return Image.fromarray(cv2.cvtColor(image_source, cv2.COLOR_GRAY2RGB))
        elif image_source.ndim == 3 and image_source.shape[2] == 3:
            return Image.fromarray(image_source)
        elif image_source.ndim == 3 and image_source.shape[2] == 4:
            return Image.fromarray(cv2.cvtColor(image_source, cv2.COLOR_RGBA2RGB))
        else:
            raise ValueError(f"Unsupported numpy image dimensions: {image_source.shape}")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_source)}")


def get_train_transforms(
    image_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> transforms.Compose:
    """
    Transforms for training with moderate, realistic augmentations.
    Avoids aggressive warping or cutoff that would destroy facial micro-texture artifacts.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def get_eval_transforms(
    image_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> transforms.Compose:
    """
    Deterministic transforms for validation, testing, and real-time inference.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def preprocess_image(
    image_source: Union[str, Image.Image, np.ndarray],
    config: Optional[Dict[str, Any]] = None,
    detect_face: bool = True,
    is_training: bool = False
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    End-to-end preprocessing pipeline for a single image.

    Returns:
        tensor: Tensor of shape (1, 3, image_size, image_size)
        metadata: Dictionary with face detection status, bbox, and original dimensions.
    """
    pil_img = load_image_safe(image_source)
    orig_w, orig_h = pil_img.size

    cfg_prep = config.get("preprocessing", {}) if config else {}
    image_size = cfg_prep.get("image_size", 224)
    mean = tuple(cfg_prep.get("normalize_mean", [0.485, 0.456, 0.406]))
    std = tuple(cfg_prep.get("normalize_std", [0.229, 0.224, 0.225]))
    margin = cfg_prep.get("face_crop_margin", 0.20)

    face_found = False
    bbox = None
    processed_pil = pil_img

    if detect_face:
        np_img = np.array(pil_img)
        detector = FaceDetector(margin=margin)
        cropped_np, bbox, face_found = detector.detect_and_crop(np_img)
        processed_pil = Image.fromarray(cropped_np)

    if is_training:
        transform = get_train_transforms(image_size, mean, std)
    else:
        transform = get_eval_transforms(image_size, mean, std)

    tensor = transform(processed_pil)
    # Add batch dimension (1, 3, H, W)
    batch_tensor = tensor.unsqueeze(0)

    metadata = {
        "original_width": orig_w,
        "original_height": orig_h,
        "face_detected": face_found,
        "bounding_box": bbox,
        "processed_size": (image_size, image_size)
    }

    return batch_tensor, metadata
