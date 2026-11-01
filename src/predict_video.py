"""
DeepShield AI - Video Deepfake Prediction Pipeline
Samples video frames at configurable intervals, detects and crops faces,
performs ResNet-18 frame-level inferences, and aggregates frame probabilities
into a calibrated video-level authenticity verdict.
"""

import os
import sys

# Ensure repository root is in python search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import argparse
from typing import Dict, Any, List, Optional
import cv2
import numpy as np
import torch
from PIL import Image

from src.utils import load_config, get_device, load_checkpoint, setup_logger
from src.model import build_model
from src.preprocessing import FaceDetector, get_eval_transforms

logger = setup_logger("DeepShieldAI.PredictVideo")

VALID_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def aggregate_scores(scores: List[float], method: str = "mean") -> float:
    """
    Combines frame-level fake probabilities into a single video-level score.

    Methods:
        mean: Standard arithmetic average. Resilient to transient single-frame noise.
        median: 50th percentile. Highly robust to extreme outlier frames.
        max: Peak fake probability. Sensitive but prone to false alarms from blurs.
        trimmed_mean: Mean excluding highest and lowest 10% values.
    """
    if not scores:
        return 0.50

    scores_arr = np.array(scores)
    method_clean = method.lower().strip()

    if method_clean == "median":
        return float(np.median(scores_arr))
    elif method_clean == "max":
        return float(np.max(scores_arr))
    elif method_clean == "trimmed_mean":
        if len(scores_arr) >= 5:
            low_cutoff = np.percentile(scores_arr, 10)
            high_cutoff = np.percentile(scores_arr, 90)
            trimmed = scores_arr[(scores_arr >= low_cutoff) & (scores_arr <= high_cutoff)]
            if len(trimmed) > 0:
                return float(np.mean(trimmed))
        return float(np.mean(scores_arr))
    else:  # default "mean"
        return float(np.mean(scores_arr))


def predict_video(
    video_path: str,
    checkpoint_path: str = "models/best_model.pth",
    config_path: str = "configs/config.yaml",
    frame_interval: Optional[int] = None,
    max_frames: Optional[int] = None,
    aggregation_method: Optional[str] = None,
    device_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes deepfake analysis across frames of an input video file.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    ext = os.path.splitext(video_path)[1].lower()
    if ext not in VALID_VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video extension '{ext}'. Supported: {VALID_VIDEO_EXTENSIONS}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint file '{checkpoint_path}' not found! "
            "Please train or provide model weights before running inference."
        )

    config = load_config(config_path)
    device = get_device(device_name)

    vid_cfg = config.get("inference", {}).get("video", {})
    prep_cfg = config.get("preprocessing", {})

    interval = frame_interval if frame_interval is not None else vid_cfg.get("frame_interval", 10)
    max_f = max_frames if max_frames is not None else vid_cfg.get("max_frames", 30)
    agg_method = aggregation_method if aggregation_method is not None else vid_cfg.get("aggregation_method", "mean")

    image_size = prep_cfg.get("image_size", 224)
    mean = tuple(prep_cfg.get("normalize_mean", [0.485, 0.456, 0.406]))
    std = tuple(prep_cfg.get("normalize_std", [0.229, 0.224, 0.225]))
    margin = prep_cfg.get("face_crop_margin", 0.20)

    # Initialize model
    model = build_model(config)
    model, _, _ = load_checkpoint(checkpoint_path, model, device=device)
    model.eval()

    transform = get_eval_transforms(image_size, mean, std)
    detector = FaceDetector(margin=margin)

    # Open video capture safely
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"OpenCV could not open video stream for file: {video_path}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_index = 0
    sampled_count = 0
    face_count = 0
    no_face_count = 0
    frame_results: List[Dict[str, Any]] = []

    try:
        while sampled_count < max_f:
            ret, bgr_frame = cap.read()
            if not ret:
                break

            # Sample every Nth frame
            if frame_index % interval == 0:
                sampled_count += 1
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

                # Attempt face detection
                cropped_face, bbox, face_found = detector.detect_and_crop(rgb_frame)

                if face_found:
                    face_count += 1
                    pil_face = Image.fromarray(cropped_face)
                    tensor = transform(pil_face).unsqueeze(0).to(device)

                    with torch.no_grad():
                        logits = model(tensor)
                        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

                    p_real = float(probs[0])
                    p_fake = float(probs[1])

                    frame_results.append({
                        "frame_index": frame_index,
                        "timestamp_seconds": round(frame_index / max(fps, 1.0), 2),
                        "face_detected": True,
                        "bounding_box": bbox,
                        "prob_fake": round(p_fake, 4),
                        "prob_real": round(p_real, 4)
                    })
                else:
                    no_face_count += 1
                    frame_results.append({
                        "frame_index": frame_index,
                        "timestamp_seconds": round(frame_index / max(fps, 1.0), 2),
                        "face_detected": False,
                        "bounding_box": None,
                        "prob_fake": None,
                        "prob_real": None
                    })

            frame_index += 1

    finally:
        cap.release()

    # Aggregate frame probabilities
    usable_scores = [f["prob_fake"] for f in frame_results if f["face_detected"] and f["prob_fake"] is not None]

    if len(usable_scores) == 0:
        logger.warning(f"No faces detected across {sampled_count} sampled frames in {video_path}!")
        video_prob_fake = 0.50
        video_prob_real = 0.50
        verdict = "INCONCLUSIVE (NO FACE DETECTED)"
        confidence = 0.50
    else:
        video_prob_fake = aggregate_scores(usable_scores, method=agg_method)
        video_prob_real = 1.0 - video_prob_fake

        threshold = config.get("inference", {}).get("confidence_threshold", 0.50)
        if video_prob_fake >= threshold:
            verdict = "FAKE"
            confidence = video_prob_fake
        else:
            verdict = "REAL"
            confidence = video_prob_real

    result = {
        "video_path": os.path.abspath(video_path),
        "total_video_frames": total_video_frames,
        "sampled_frames": sampled_count,
        "processed_face_frames": face_count,
        "skipped_no_face_frames": no_face_count,
        "aggregation_method": agg_method,
        "predicted_class": verdict,
        "confidence": round(confidence, 4),
        "confidence_percent": f"{confidence * 100:.2f}%",
        "probabilities": {
            "REAL": round(video_prob_real, 4),
            "FAKE": round(video_prob_fake, 4)
        },
        "frame_breakdown": frame_results,
        "disclaimer": (
            "NOTICE: Video predictions aggregate frame-level facial features. "
            "Temporal consistency, audio manipulation, and synthetic voices are not assessed by this visual classifier."
        )
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Predict whether a video is REAL or FAKE using DeepShield AI")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pth", help="Path to checkpoint .pth")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--frame_interval", type=int, default=None, help="Sample every N frames")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum frames to evaluate")
    parser.add_argument("--aggregation", type=str, default="mean", choices=["mean", "median", "max", "trimmed_mean"], help="Score aggregation method")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu' or 'cuda')")
    parser.add_argument("--output", type=str, default=None, help="Path to save output JSON")

    args = parser.parse_args()

    try:
        res = predict_video(
            video_path=args.video,
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            frame_interval=args.frame_interval,
            max_frames=args.max_frames,
            aggregation_method=args.aggregation,
            device_name=args.device
        )

        print("\n" + "=" * 60)
        print("          DEEPSHIELD AI - VIDEO ANALYSIS RESULT          ")
        print("=" * 60)
        print(f"File:                  {res['video_path']}")
        print(f"Total Video Frames:    {res['total_video_frames']}")
        print(f"Sampled Frames:        {res['sampled_frames']}")
        print(f"Frames with Faces:     {res['processed_face_frames']}")
        print(f"Skipped (No Face):     {res['skipped_no_face_frames']}")
        print(f"Aggregation Strategy:  {res['aggregation_method'].upper()}")
        print(f"Verdict:               {res['predicted_class']}")
        print(f"Confidence:            {res['confidence_percent']}")
        print(f"  - Prob(REAL):        {res['probabilities']['REAL'] * 100:.2f}%")
        print(f"  - Prob(FAKE):        {res['probabilities']['FAKE'] * 100:.2f}%")
        print("-" * 60)
        print(res['disclaimer'])
        print("=" * 60 + "\n")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=4)
            print(f"Saved JSON result to {args.output}")

    except Exception as e:
        logger.error(f"Video inference error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
