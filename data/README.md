# DeepShield AI — Dataset Storage Directory

This directory stores training, validation, and test media partitions.

## Directory Structure
```
data/
├── train/
│   ├── real/     # Authentic training images
│   └── fake/     # Manipulated training images
├── val/
│   ├── real/     # Authentic validation images
│   └── fake/     # Manipulated validation images
├── test/
│   ├── real/     # Authentic held-out test images
│   └── fake/     # Manipulated held-out test images
├── manifest.csv           # Generated metadata manifest
└── dataset_summary.json   # Class counts and integrity logs
```

## Dataset Acquisition Options

### 1. Academic Datasets
To obtain established research benchmarks:
- **FaceForensics++:** [https://github.com/ondyari/FaceForensics](https://github.com/ondyari/FaceForensics) (Submit the official Google Form)
- **Celeb-DF v2:** [https://github.com/yuezunli/celeb-deepfakeforensics](https://github.com/yuezunli/celeb-deepfakeforensics) (Request via repo)
- **DFDC:** [https://ai.meta.com/datasets/dfdc/](https://ai.meta.com/datasets/dfdc/)

Run the acquisition guide script for detailed procedures:
```powershell
python scripts/prepare_dataset.py --guide
```

### 2. Pipeline Test Dataset (For Immediate Testing & Smoke Tests)
To generate an immediate offline test dataset without downloading 15 GB of media:
```powershell
python scripts/prepare_dataset.py --generate_pipeline_dataset --samples 15
```

### 3. Validating Image Integrity
To scan the data directory, verify image readability, detect corrupt files, and generate `manifest.csv`:
```powershell
python scripts/prepare_dataset.py --validate
```
