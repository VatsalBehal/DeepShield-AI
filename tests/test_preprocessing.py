"""
DeepShield AI - Unit Tests for Preprocessing and Computer Vision
"""

import os
import pytest
import numpy as np
import torch
from PIL import Image

from src.preprocessing import (
    FaceDetector,
    load_image_safe,
    get_train_transforms,
    get_eval_transforms,
    preprocess_image
)


def test_face_detector_initialization():
    detector = FaceDetector()
    assert detector is not None
    assert detector.detector is not None


def test_face_detector_empty_image_raises_error():
    detector = FaceDetector()
    with pytest.raises(ValueError):
        detector.detect_and_crop(np.array([]))


def test_face_detector_no_face_fallback():
    detector = FaceDetector()
    # Uniform color image has no face
    dummy_img = np.full((100, 100, 3), 128, dtype=np.uint8)
    cropped, bbox, found = detector.detect_and_crop(dummy_img)
    assert found is False
    assert bbox is None
    assert cropped.shape == (100, 100, 3)


def test_load_image_safe_pil():
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    loaded = load_image_safe(img)
    assert isinstance(loaded, Image.Image)
    assert loaded.size == (64, 64)
    assert loaded.mode == "RGB"


def test_load_image_safe_numpy():
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    loaded = load_image_safe(arr)
    assert isinstance(loaded, Image.Image)
    assert loaded.size == (64, 64)


def test_load_image_safe_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        load_image_safe("path/that/does/not/exist.jpg")


def test_transforms_output_dimensions():
    train_t = get_train_transforms(image_size=224)
    eval_t = get_eval_transforms(image_size=224)

    dummy_img = Image.new("RGB", (100, 150), color=(120, 120, 120))
    train_tensor = train_t(dummy_img)
    eval_tensor = eval_t(dummy_img)

    assert isinstance(train_tensor, torch.Tensor)
    assert train_tensor.shape == (3, 224, 224)
    assert isinstance(eval_tensor, torch.Tensor)
    assert eval_tensor.shape == (3, 224, 224)


def test_preprocess_image_pipeline():
    dummy_img = Image.new("RGB", (200, 200), color=(200, 180, 160))
    tensor, meta = preprocess_image(dummy_img, detect_face=False)

    assert tensor.shape == (1, 3, 224, 224)
    assert meta["original_width"] == 200
    assert meta["original_height"] == 200
    assert meta["face_detected"] is False
