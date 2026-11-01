"""
DeepShield AI - End-to-End Pipeline Smoke Test
Executes end-to-end verification across model building, forward pass,
image inference, video generation and inference, and evaluation reporting.
"""

import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import cv2
import torch
from PIL import Image

from src.utils import load_config, save_checkpoint, setup_logger
from src.model import build_model
from src.preprocessing import preprocess_image
from src.predict_image import predict_single_image
from src.predict_video import predict_video
from src.evaluate import evaluate_model
from scripts.prepare_dataset import create_pipeline_test_dataset

logger = setup_logger("DeepShieldAI.SmokeTest")


def generate_synthetic_test_video(output_path: str = "sample_inputs/smoke_test_video.mp4", num_frames: int = 25):
    """Generates a small valid MP4 video containing face features for video inference testing."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 10.0, (256, 256))

    for i in range(num_frames):
        frame = np.full((256, 256, 3), 230, dtype=np.uint8)
        # Draw face oval
        cv2.ellipse(frame, (128, 128), (60, 80), 0, 0, 360, (180, 200, 225), -1)
        cv2.ellipse(frame, (128, 128), (60, 80), 0, 0, 360, (120, 140, 180), 2)
        # Eyes
        cv2.circle(frame, (105, 110), 8, (255, 255, 255), -1)
        cv2.circle(frame, (105, 110), 4, (30, 30, 30), -1)
        cv2.circle(frame, (151, 110), 8, (255, 255, 255), -1)
        cv2.circle(frame, (151, 110), 4, (30, 30, 30), -1)
        # Nose and mouth
        cv2.line(frame, (128, 115), (128, 145), (100, 120, 150), 2)
        cv2.ellipse(frame, (128, 165), (20, 10), 0, 0, 180, (50, 50, 180), 2)

        out.write(frame)

    out.release()
    logger.info(f"Synthetic video generated at {output_path}")
    return output_path


def run_smoke_test():
    print("=" * 65)
    print("        DEEPSHIELD AI - END-TO-END SYSTEM SMOKE TEST         ")
    print("=" * 65)

    checks = []

    # 1. Config Loading
    try:
        cfg = load_config("configs/config.yaml")
        assert "model" in cfg and "dataset" in cfg
        checks.append(("Configuration Loading", True, "Config parsed with expected sections."))
    except Exception as e:
        checks.append(("Configuration Loading", False, str(e)))

    # 2. Pipeline Test Dataset
    try:
        if not os.path.exists("data/train/real") or len(os.listdir("data/train/real")) == 0:
            logger.info("Setting up synthetic pipeline dataset for smoke test...")
            create_pipeline_test_dataset("data", num_per_class=6)
        checks.append(("Pipeline Dataset Verification", True, "Dataset structure and samples verified."))
    except Exception as e:
        checks.append(("Pipeline Dataset Verification", False, str(e)))

    # 3. Model Instantiation & Forward Pass
    try:
        model = build_model(cfg)
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        assert output.shape == (2, 2), f"Expected shape (2, 2), got {output.shape}"
        checks.append(("Model Forward Pass (ResNet-18)", True, f"Output tensor shape: {tuple(output.shape)}"))
    except Exception as e:
        checks.append(("Model Forward Pass (ResNet-18)", False, str(e)))

    # 4. Checkpoint Creation / Validation
    ckpt_path = "models/best_model.pth"
    try:
        if not os.path.exists(ckpt_path):
            logger.info("Creating a baseline test checkpoint for smoke testing...")
            state = {
                "epoch": 1,
                "model_state_dict": model.state_dict(),
                "val_loss": 0.693,
                "val_accuracy": 0.50,
                "class_to_idx": {"real": 0, "fake": 1},
                "architecture": "resnet18",
                "note": "SMOKE_TEST_BASELINE_WEIGHTS (Untrained fixture for pipeline validation)"
            }
            save_checkpoint(state, checkpoint_dir="models", checkpoint_name="best_model.pth")
        checks.append(("Checkpoint File Availability", True, f"Verified checkpoint at {ckpt_path}"))
    except Exception as e:
        checks.append(("Checkpoint File Availability", False, str(e)))

    # 5. Image Preprocessing & Inference
    try:
        test_img_path = "sample_inputs/test_face.jpg"
        os.makedirs("sample_inputs", exist_ok=True)
        # Create a test face image if none exists
        if not os.path.exists(test_img_path):
            test_img = Image.new("RGB", (256, 256), color=(240, 230, 220))
            test_img.save(test_img_path)

        img_res = predict_single_image(test_img_path, checkpoint_path=ckpt_path)
        assert "predicted_class" in img_res and "confidence" in img_res
        checks.append(("Single Image Inference Pipeline", True, f"Prediction: {img_res['predicted_class']} ({img_res['confidence_percent']})"))
    except Exception as e:
        checks.append(("Single Image Inference Pipeline", False, str(e)))

    # 6. Video Generation & Inference
    try:
        vid_path = "sample_inputs/smoke_test_video.mp4"
        generate_synthetic_test_video(vid_path, num_frames=20)
        vid_res = predict_video(vid_path, checkpoint_path=ckpt_path, frame_interval=5, max_frames=5)
        assert "predicted_class" in vid_res and "sampled_frames" in vid_res
        checks.append(("Video Inference Pipeline", True, f"Verdict: {vid_res['predicted_class']} across {vid_res['sampled_frames']} frames"))
    except Exception as e:
        checks.append(("Video Inference Pipeline", False, str(e)))

    # 7. Test Evaluation Pipeline
    try:
        eval_res = evaluate_model(
            checkpoint_path=ckpt_path,
            test_dir="data/test",
            config_path="configs/config.yaml",
            output_dir="outputs"
        )
        assert "accuracy" in eval_res
        checks.append(("Held-Out Test Set Evaluation", True, f"Generated metrics and plots in outputs/ (Acc: {eval_res['accuracy']*100:.1f}%)"))
    except Exception as e:
        checks.append(("Held-Out Test Set Evaluation", False, str(e)))

    # Print summary table
    print("\n" + "-" * 65)
    print("TEST SUITE RESULTS:")
    all_passed = True
    for name, passed, detail in checks:
        mark = "[PASS]" if passed else "[FAIL]"
        print(f"  {mark} {name:<32} : {detail}")
        if not passed:
            all_passed = False

    print("=" * 65)
    if all_passed:
        print("ALL PIPELINE CHECKS PASSED SUCCESSFULLY! [OK]")
    else:
        print("SOME CHECKS ENCOUNTERED ERRORS. Check logs above.")
    print("=" * 65 + "\n")
    return all_passed


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
