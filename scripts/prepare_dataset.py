"""
DeepShield AI - Dataset Preparation, Verification, and Pipeline Generation Tool
Provides dataset integrity checking, corruption filtering, manifest generation,
leakage-free splitting guidance, and an offline pipeline test dataset generator.
"""

import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import argparse
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw

from src.utils import setup_logger
from src.preprocessing import load_image_safe

logger = setup_logger("DeepShieldAI.PrepareDataset")

ACADEMIC_DATASETS_INFO = {
    "FaceForensics++": {
        "url": "https://github.com/ondyari/FaceForensics",
        "description": "Standard benchmark with 1000 original YouTube videos manipulated with 4 methods (Deepfakes, Face2Face, FaceSwap, NeuralTextures).",
        "access": "Submit Google Form agreement to authors to receive official download script.",
        "storage": "c23 (light compression): ~40 GB. Raw (c0): >1 TB.",
        "recommended_subset": "Sample 100 Real videos and 100 Deepfakes videos, extract 20 frames per video."
    },
    "Celeb-DF v2": {
        "url": "https://github.com/yuezunli/celeb-deepfakeforensics",
        "description": "High-quality celebrity deepfakes with realistic facial boundaries and lighting.",
        "access": "Request access link via official GitHub repo request form.",
        "storage": "Full dataset: ~16.8 GB.",
        "recommended_subset": "100 Celeb-real videos + 100 Celeb-synthesis videos (approx 3-5 GB)."
    },
    "DFDC (Deepfake Detection Challenge)": {
        "url": "https://ai.meta.com/datasets/dfdc/",
        "description": "Meta AI benchmark featuring diverse demographic actors, settings, and manipulations.",
        "access": "Direct registration and acceptance of competition terms.",
        "storage": "Full dataset: >470 GB. Preview dataset: ~4 GB.",
        "recommended_subset": "DFDC Preview subset (~4 GB, 5000 video clips)."
    }
}


def print_acquisition_guide():
    """Prints academic dataset acquisition documentation."""
    print("=" * 70)
    print("         DEEPSHIELD AI - DATASET ACQUISITION & RESEARCH GUIDE         ")
    print("=" * 70)
    for name, info in ACADEMIC_DATASETS_INFO.items():
        print(f"\n[{name}]")
        print(f"  Description:         {info['description']}")
        print(f"  Official Portal:     {info['url']}")
        print(f"  Access Procedure:    {info['access']}")
        print(f"  Storage Requirement: {info['storage']}")
        print(f"  Recommended Subset:  {info['recommended_subset']}")

    print("\n" + "-" * 70)
    print("CRITICAL DATA LEAKAGE PREVENTION PRINCIPLE:")
    print("  When preparing frames from video datasets:")
    print("  1. NEVER randomly split frames into train/val/test.")
    print("  2. Split SOURCE VIDEOS first (e.g., 70% train, 15% val, 15% test).")
    print("  3. Keep ALL frames of a given video within the same split.")
    print("  4. Group manipulated videos with their original donor video.")
    print("=" * 70 + "\n")


def setup_directory_structure(root_dir: str = "data") -> None:
    """Creates the standard train/val/test real/fake folders."""
    splits = ["train", "val", "test"]
    classes = ["real", "fake"]
    for s in splits:
        for c in classes:
            folder = os.path.join(root_dir, s, c)
            os.makedirs(folder, exist_ok=True)
    logger.info(f"Directory structure initialized at '{root_dir}/'")


def validate_and_manifest(root_dir: str = "data") -> Dict[str, Any]:
    """
    Scans data directory, verifies image readability, detects corrupt files,
    and produces data/manifest.csv and data/dataset_summary.json.
    """
    records: List[Dict[str, Any]] = []
    corrupt_files: List[str] = []
    class_counts: Dict[str, Dict[str, int]] = {
        "train": {"real": 0, "fake": 0},
        "val": {"real": 0, "fake": 0},
        "test": {"real": 0, "fake": 0}
    }

    for split in ["train", "val", "test"]:
        for cls in ["real", "fake"]:
            cls_folder = os.path.join(root_dir, split, cls)
            if not os.path.isdir(cls_folder):
                continue

            for root, _, files in os.walk(cls_folder):
                for f in files:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                        fpath = os.path.join(root, f)
                        try:
                            # Test image opening and byte verification
                            with Image.open(fpath) as img:
                                img.verify()
                            # Safe load to ensure RGB conversion works
                            load_image_safe(fpath)
                            records.append({
                                "filepath": fpath,
                                "filename": f,
                                "split": split,
                                "label": cls,
                                "class_index": 0 if cls == "real" else 1
                            })
                            class_counts[split][cls] += 1
                        except Exception as e:
                            logger.warning(f"Corrupt or unreadable image removed from index: {fpath} ({e})")
                            corrupt_files.append(fpath)

    df = pd.DataFrame(records)
    manifest_path = os.path.join(root_dir, "manifest.csv")
    df.to_csv(manifest_path, index=False)

    summary = {
        "total_valid_samples": len(records),
        "total_corrupted_samples": len(corrupt_files),
        "split_counts": class_counts,
        "manifest_path": manifest_path
    }

    summary_path = os.path.join(root_dir, "dataset_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    logger.info(f"Dataset verification complete! Summary written to {summary_path}")
    print(f"\nDataset Summary: {summary['total_valid_samples']} valid images indexed.")
    for s, counts in class_counts.items():
        print(f"  {s.upper():<5} -> Real: {counts['real']}, Fake: {counts['fake']}")
    if corrupt_files:
        print(f"  Warning: {len(corrupt_files)} corrupt files detected.")

    return summary


def create_pipeline_test_dataset(root_dir: str = "data", num_per_class: int = 10) -> None:
    """
    Creates a clearly labeled synthetic test dataset for verifying the end-to-end
    pipeline (smoke tests, data loading, training loop, inference, and Streamlit)
    WITHOUT downloading a 15GB academic dataset.

    HONESTY NOTICE:
    This dataset is strictly for software pipeline verification and smoke testing.
    It does NOT represent trained real-world deepfake forensic capabilities.
    """
    logger.info(f"Generating synthetic pipeline test dataset ({num_per_class} samples/class per split)...")
    setup_directory_structure(root_dir)

    splits = {
        "train": num_per_class,
        "val": max(2, num_per_class // 3),
        "test": max(2, num_per_class // 3)
    }

    for split, count in splits.items():
        for cls in ["real", "fake"]:
            folder = os.path.join(root_dir, split, cls)
            for i in range(count):
                # Create a 256x256 test face-like canvas
                img = Image.new("RGB", (256, 256), color=(240, 230, 220))
                draw = ImageDraw.Draw(img)

                # Draw simplified head and facial features
                # Head oval
                draw.ellipse([50, 40, 206, 216], fill=(225, 195, 175), outline=(180, 140, 120), width=2)
                # Eyes
                draw.ellipse([85, 100, 115, 120], fill=(255, 255, 255), outline=(50, 50, 50))
                draw.ellipse([95, 105, 105, 115], fill=(30, 30, 30))
                draw.ellipse([141, 100, 171, 120], fill=(255, 255, 255), outline=(50, 50, 50))
                draw.ellipse([151, 105, 161, 115], fill=(30, 30, 30))
                # Nose
                draw.line([(128, 120), (124, 150), (132, 150)], fill=(160, 110, 90), width=2)
                # Mouth
                draw.arc([100, 160, 156, 185], start=0, end=180, fill=(180, 60, 60), width=3)

                if cls == "fake":
                    # Add digital manipulation artifacts (color band / boundary seam)
                    draw.rectangle([70, 90, 186, 190], outline=(0, 255, 200), width=2)
                    # Gaussian-like noise patches
                    arr = np.array(img)
                    noise = np.random.randint(-30, 30, arr.shape, dtype=np.int16)
                    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                    img = Image.fromarray(arr)

                fname = f"sample_{cls}_{split}_{i:03d}.jpg"
                fpath = os.path.join(folder, fname)
                img.save(fpath, quality=95)

    validate_and_manifest(root_dir)
    logger.info("Pipeline test dataset generation finished.")


def main():
    parser = argparse.ArgumentParser(description="Prepare and validate dataset for DeepShield AI")
    parser.add_argument("--guide", action="store_true", help="Print academic dataset acquisition instructions")
    parser.add_argument("--root", type=str, default="data", help="Root data folder")
    parser.add_argument("--validate", action="store_true", help="Scan and validate images in data folder")
    parser.add_argument("--generate_pipeline_dataset", action="store_true", help="Create pipeline test dataset for smoke testing")
    parser.add_argument("--samples", type=int, default=12, help="Number of samples per class for pipeline generator")

    args = parser.parse_args()

    if args.guide:
        print_acquisition_guide()
        return

    if args.generate_pipeline_dataset:
        create_pipeline_test_dataset(root_dir=args.root, num_per_class=args.samples)
        return

    if args.validate or not any([args.guide, args.generate_pipeline_dataset]):
        setup_directory_structure(args.root)
        validate_and_manifest(args.root)


if __name__ == "__main__":
    main()
