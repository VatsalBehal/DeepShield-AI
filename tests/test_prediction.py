"""
DeepShield AI - Unit Tests for Prediction Pipelines and Aggregation Logic
"""

import os
import pytest
from PIL import Image

from src.predict_image import predict_single_image
from src.predict_video import aggregate_scores, predict_video
from src.model import DeepShieldResNet18
from src.utils import save_checkpoint


def test_aggregate_scores():
    scores = [0.2, 0.4, 0.6, 0.8]
    # Mean
    mean_val = aggregate_scores(scores, method="mean")
    assert abs(mean_val - 0.5) < 1e-4

    # Median
    median_val = aggregate_scores(scores, method="median")
    assert abs(median_val - 0.5) < 1e-4

    # Max
    max_val = aggregate_scores(scores, method="max")
    assert abs(max_val - 0.8) < 1e-4

    # Empty list fallback
    empty_val = aggregate_scores([], method="mean")
    assert empty_val == 0.50


def test_predict_image_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        predict_single_image("nonexistent_img.jpg", checkpoint_path="models/best_model.pth")


def test_predict_image_missing_checkpoint_raises(tmp_path):
    img_path = str(tmp_path / "sample.jpg")
    img = Image.new("RGB", (100, 100), color=(200, 200, 200))
    img.save(img_path)

    with pytest.raises(FileNotFoundError):
        predict_single_image(img_path, checkpoint_path=str(tmp_path / "missing_model.pth"))


def test_predict_video_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        predict_video("nonexistent_vid.mp4", checkpoint_path="models/best_model.pth")


def test_predict_video_invalid_extension_raises(tmp_path):
    invalid_file = str(tmp_path / "test.txt")
    with open(invalid_file, "w") as f:
        f.write("text content")

    with pytest.raises(ValueError):
        predict_video(invalid_file, checkpoint_path="models/best_model.pth")
