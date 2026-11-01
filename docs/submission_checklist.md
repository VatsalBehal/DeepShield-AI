# DeepShield AI — University Project Submission Checklist

Use this checklist to ensure all components, files, tests, and documentation are complete and verified prior to submitting your project.

---

## 1. Source Code & Environment Verification
- [ ] Python 3.10 virtual environment created at `.venv/`.
- [ ] Dependencies installed via `requirements.txt`.
- [ ] `scripts/check_environment.py` runs with `[PASS]` status.
- [ ] Automated pytest suite runs with all tests passing:
  ```powershell
  python -m pytest -q
  ```
- [ ] `scripts/smoke_test.py` completes with all checks passed.

---

## 2. Model Checkpoint & Training Assets
- [ ] `models/best_model.pth` is present and valid.
- [ ] If trained on Google Colab: `notebooks/DeepShield_AI_Training.ipynb` runs from start to finish on a T4 GPU without errors.
- [ ] Model architecture conforms to ResNet-18 with 2 output classes (`REAL` and `FAKE`).
- [ ] Checkpoint metadata includes `class_to_idx`, `architecture`, and `preprocessing` settings.

---

## 3. Dataset Integrity & Leakage Prevention
- [ ] Dataset directories follow structure:
  - `data/train/real/` and `data/train/fake/`
  - `data/val/real/` and `data/val/fake/`
  - `data/test/real/` and `data/test/fake/`
- [ ] Source videos were partitioned **before** frame extraction (no identity leakage across splits).
- [ ] `data/manifest.csv` and `data/dataset_summary.json` generated via `python scripts/prepare_dataset.py --validate`.

---

## 4. Evaluation Outputs
- [ ] `python src/evaluate.py` executed successfully.
- [ ] `outputs/evaluation_summary.json` contains quantitative metrics.
- [ ] `outputs/confusion_matrix.png` generated and verified.
- [ ] `outputs/roc_curve.png` generated and verified.
- [ ] `outputs/classification_report.txt` saved.

---

## 5. Web Application (Streamlit)
- [ ] Application launches cleanly:
  ```powershell
  streamlit run app.py
  ```
- [ ] Single Image Analysis tested and verified.
- [ ] Video Analysis tested with frame sampling and timeline plotting.
- [ ] Model & Architecture documentation tabs render properly.
- [ ] Privacy notice and forensic disclaimers clearly visible.

---

## 6. Academic Documentation & Viva Materials
- [ ] `docs/project_report.md` customized with student name, roll number, and college details.
- [ ] `docs/architecture.md` reviewed and understood.
- [ ] `docs/viva_questions.md` (40+ questions) studied for viva defense.
- [ ] `docs/demo_script.md` rehearsed for the 5-minute presentation.
- [ ] Project screenshots captured according to the checklist.
