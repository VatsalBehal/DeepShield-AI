# DeepShield AI — Deepfake Image and Video Detection System

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-brightgreen.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An academic and practical deepfake detection system developed for a second-year **B.Tech Computer Science and Engineering (AI & Data Science)** capstone project. 

DeepShield AI detects facial manipulations across both static images and video sequences using a transfer-learned **ResNet-18** convolutional neural network, automated OpenCV facial boundary extraction with margin compensation, and an interactive Streamlit graphical dashboard.

---

## 🌟 Key Features

- **Transfer Learning with ResNet-18:** Pretrained on ImageNet with 11.2M parameters; lightweight and optimized for CPU inference on an 8 GB RAM laptop.
- **OpenCV Face Localization & Margin Expansion:** Detects the primary human face and expands the bounding box by 20% to capture critical blending seams along the jawline and hairline.
- **Data Leakage Prevention:** Partitions media strictly at the source-video level before frame extraction.
- **Temporal Video Inference:** Decodes video streams, samples frames at configurable intervals, scores facial regions, and aggregates probabilities using robust statistical methods (mean, median, trimmed mean).
- **Google Colab Training Workflow:** Complete GPU-accelerated notebook (`notebooks/DeepShield_AI_Training.ipynb`) ready for one-click model training on Google Colab's free T4 GPU.
- **Interactive Streamlit Web Dashboard:** Polished user interface (`app.py`) featuring image drag-and-drop, video timeline charts, face crop galleries, model metrics, and viva exam prep.
- **Automated Verification:** Comprehensive `pytest` test suite and an end-to-end `smoke_test.py` script.

---

## 📁 Project Directory Structure

```
DeepShield-AI/
├── app.py                      # Streamlit graphical application
├── requirements.txt            # Dependency specification
├── README.md                   # Project documentation and quickstart
├── .gitignore                  # Git exclusion rules
├── configs/
│   └── config.yaml             # Global system configurations
├── data/
│   └── README.md               # Dataset partitioning and acquisition guide
├── models/
│   └── README.md               # Trained model checkpoint storage
├── notebooks/
│   └── DeepShield_AI_Training.ipynb # Google Colab GPU training notebook
├── src/
│   ├── __init__.py             # Package marker
│   ├── dataset.py              # Dataset loader and leakage-free splitting
│   ├── preprocessing.py        # Face detection, cropping, and transforms
│   ├── model.py                # ResNet-18 architecture with dropout
│   ├── train.py                # Model training and validation loops
│   ├── evaluate.py             # Performance evaluation and metrics
│   ├── predict_image.py        # CLI single image inference
│   ├── predict_video.py        # CLI video sequence inference
│   └── utils.py                # Checkpoint serialization, logging, plotting
├── scripts/
│   ├── check_environment.py    # Environment and dependency diagnostic tool
│   ├── prepare_dataset.py      # Dataset validation and pipeline generator
│   └── smoke_test.py           # End-to-end integration smoke test
├── tests/
│   ├── test_preprocessing.py   # Preprocessing and face detector tests
│   ├── test_model.py           # ResNet-18 forward pass and checkpoint tests
│   └── test_prediction.py      # Inference and score aggregation tests
├── outputs/
│   └── README.md               # Generated confusion matrices, plots, and logs
├── docs/
│   ├── project_report.md       # Full B.Tech academic project report
│   ├── architecture.md         # Mermaid system architecture and DFD
│   ├── viva_questions.md       # 45+ comprehensive viva questions & answers
│   ├── demo_script.md          # 5-minute teacher presentation script
│   └── submission_checklist.md # Final submission verification checklist
└── sample_inputs/
    └── README.md               # Sample images and video clips for testing
```

---

## 🚀 Windows Installation and Execution Guide (Step-by-Step)

Follow these exact steps in Windows PowerShell to set up and run the project locally.

### Step 1: Open PowerShell in the Project Directory
Press `Win + X`, select **Terminal** or **PowerShell**, and navigate to your project directory:
```powershell
cd D:\DEEPFAKE_DETECTION
```

### Step 2: Create and Activate Virtual Environment
```powershell
py -3.10 -m venv .venv
```
Activate the virtual environment:
```powershell
.\.venv\Scripts\Activate.ps1
```
> *Troubleshooting Note:* If PowerShell gives an execution policy error (`running scripts is disabled`), run:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and then run the activation script again.

### Step 3: Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Run Environment Diagnostics
Verify that all packages, OpenCV cascades, and hardware resources are detected:
```powershell
python scripts/check_environment.py
```
Expected output: `DIAGNOSTIC VERDICT: System is fully configured and ready! [PASS]`

---

## 🧪 Quick Pipeline Smoke Test (Zero-Download Verification)

To immediately test the entire software pipeline—including model creation, image inference, video inference, and evaluation—without downloading a 15 GB academic dataset, execute:

```powershell
python scripts/smoke_test.py
```
This generates a pipeline test dataset, initializes a baseline model checkpoint, creates a sample video, runs predictions, and outputs evaluation plots to `outputs/`.

---

## 🏋️ Model Training via Google Colab (GPU Acceleration)

Because laptops with 8 GB RAM and integrated Intel GPUs are constrained for deep convolutional training, model training is conducted on Google Colab using a free NVIDIA T4 GPU:

1. Open Google Colab: [https://colab.research.google.com/](https://colab.research.google.com/)
2. Upload `notebooks/DeepShield_AI_Training.ipynb`.
3. Set runtime to GPU: **Runtime** → **Change runtime type** → **T4 GPU**.
4. Run the notebook cells sequentially to train the model, validate epochs, and evaluate the held-out test set.
5. The final notebook cell downloads `best_model.pth`.
6. Place `best_model.pth` in your local directory: `D:\DEEPFAKE_DETECTION\models\best_model.pth`.

---

## 🔍 Command-Line Inference

### 1. Single Image Inference
```powershell
python src/predict_image.py --image sample_inputs/test_face.jpg --checkpoint models/best_model.pth
```
Example Output:
```
=======================================================
          DEEPSHIELD AI - IMAGE ANALYSIS RESULT          
=======================================================
File:             sample_inputs/test_face.jpg
Face Detected:    Yes
Face Bounding Box: [x=50, y=40, w=156, h=176]
Verdict:          AUTHENTIC (REAL)
Confidence:       87.42%
  - Prob(REAL):   87.42%
  - Prob(FAKE):   12.58%
-------------------------------------------------------
NOTICE: Predictions are probabilistic classifications...
=======================================================
```

### 2. Video Sequence Inference
```powershell
python src/predict_video.py --video sample_inputs/smoke_test_video.mp4 --checkpoint models/best_model.pth --frame_interval 10 --aggregation mean
```

---

## 🖥️ Launching the Streamlit Web Application

To launch the interactive dashboard:
```powershell
streamlit run app.py
```
The browser will automatically open at `http://localhost:8501`.

The web application features:
- **Image Analysis Tab:** Drag-and-drop image upload, face bounding box display, real vs fake probability meters.
- **Video Analysis Tab:** Video player, frame extraction progress bar, timeline probability chart, face crop gallery.
- **Evaluation & Metrics Tab:** Real-time rendering of test confusion matrices and ROC curves.
- **Architecture & Methodology Tab:** Mathematical formulas and pipeline explanations.
- **Viva & College Guide Tab:** Interactive viva questions and presentation notes.

---

## 🧪 Running Automated Unit Tests

Run the Pytest suite to verify system integrity:
```powershell
python -m pytest -q
```
Expected output: All unit tests in `tests/` pass cleanly.

---

## 📚 Academic Documentation & University Submission Materials

Comprehensive university documentation is located in the `docs/` folder:
- **Project Report:** `docs/project_report.md` (Complete capstone report with academic references).
- **System Architecture:** `docs/architecture.md` (Mermaid architectural diagrams and DFDs).
- **Viva Questions & Answers:** `docs/viva_questions.md` (45+ viva questions with student-friendly answers).
- **Live Demo Script:** `docs/demo_script.md` (5-minute teacher presentation guide).
- **Submission Checklist:** `docs/submission_checklist.md` (Final pre-submission verification).

---

## ⚠️ Scientific & Forensic Disclaimer

DeepShield AI provides automated probabilistic assessments based on learned pattern discrimination in pixel micro-textures. Predictions do **not** constitute definitive legal or forensic proof of authenticity. Model accuracy can be affected by lossy video compression, unusual illumination, or unseen generative algorithms.

---

## 📜 License
This project is developed for educational and academic research purposes under the MIT License.
