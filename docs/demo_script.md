# DeepShield AI — 5-Minute College Demonstration Script

This script provides an exact minute-by-minute blueprint for demonstrating your project before faculty examiners and viva evaluators.

---

## Preparation Checklist (Before the Presentation)

1. Open PowerShell and navigate to `D:\DEEPFAKE_DETECTION`.
2. Activate your virtual environment: `.\.venv\Scripts\Activate.ps1`.
3. Verify that `models/best_model.pth` exists.
4. Launch the application:
   ```powershell
   streamlit run app.py
   ```
5. Keep your browser open at `http://localhost:8501`.
6. Have sample images and videos ready in the `sample_inputs/` folder.

---

## 5-Minute Demonstration Walkthrough

### [00:00 - 01:00] Introduction & Problem Motivation
- **What to say:**
  > "Good morning, respected teachers. Today I am presenting **DeepShield AI**, a deep learning system engineered to detect manipulated facial media—commonly known as deepfakes—in both still images and video streams.
  > With recent advances in Generative Adversarial Networks (GANs) and diffusion models, hyper-realistic face replacements pose severe security challenges including financial fraud, identity manipulation, and media disinformation.
  > DeepShield AI employs a transfer-learned ResNet-18 convolutional neural network combined with computer-vision face extraction to accurately classify media as either AUTHENTIC (REAL) or MANIPULATED (FAKE)."
- **What to show on screen:**
  - The Streamlit web application homepage (`http://localhost:8501`).
  - Highlight the clean interface, system status indicator showing "Model weights loaded", and privacy guarantee in the sidebar.

---

### [01:00 - 02:15] Single Image Analysis Live Demonstration
- **What to do:**
  1. Click on the **"Image Analysis"** tab.
  2. Upload a test image from `sample_inputs/` or your prepared dataset.
  3. Click **"Analyze Image"**.
- **What to say:**
  > "Notice how our preprocessing pipeline operates automatically:
  > First, the OpenCV Haar Cascade localizes the primary human face in the image and adds a 20% margin around the bounding box. This margin is crucial because deepfake synthesis algorithms frequently leave seam artifacts along the outer jawline and hairline.
  > Next, the face is resized to 224×224 pixels, normalized with ImageNet statistics, and passed through ResNet-18.
  > The system outputs a clear verdict—here classified as AUTHENTIC or MANIPULATED—along with calibrated Softmax probabilities for both classes, inference latency in milliseconds, and the isolated facial crop."

---

### [02:15 - 03:30] Video Sequence Analysis Live Demonstration
- **What to do:**
  1. Switch to the **"Video Analysis"** tab.
  2. Upload an MP4 video clip (e.g. `sample_inputs/smoke_test_video.mp4` or a sample video).
  3. Click **"Analyze Video Sequence"**.
- **What to say:**
  > "For video media, evaluating every single frame at 30 frames-per-second is computationally redundant. Instead, DeepShield AI employs intelligent temporal sampling.
  > The system samples every 10th frame, verifies face presence, scores each frame independently, and aggregates the scores across the timeline using a robust mean or trimmed-mean function.
  > On screen, you can see the dynamic timeline chart showing fake probability fluctuations across timestamps, a summary of total vs sampled frames, and a visual gallery of inspected face crops."

---

### [03:30 - 04:15] Architecture & Performance Metrics
- **What to do:**
  1. Switch to the **"Evaluation & Metrics"** tab.
  2. Show the Confusion Matrix and ROC Curve.
- **What to say:**
  > "Under our technical methodology:
  > We chose ResNet-18 due to its residual skip connections, which solve the vanishing gradient problem while maintaining a lightweight footprint of 11.2 million parameters. This allows it to execute on standard laptop hardware without requiring an expensive discrete GPU.
  > To ensure scientific rigor, we strictly prevented **data leakage** by partitioning at the source video level before extracting frames.
  > Here on the evaluation dashboard, you can see our held-out test set metrics: Accuracy, Precision, Recall, F1-Score, the Confusion Matrix heatmap, and the ROC curve."

---

### [04:15 - 05:00] Limitations, Future Scope & Conclusion
- **What to say:**
  > "Finally, we maintain complete scientific honesty regarding system limitations:
  > The model outputs statistical confidence rather than forensic proof. Video re-compression and unseen generative algorithms can induce domain shift.
  > In future iterations, we plan to incorporate multi-modal audio forensics to detect voice cloning and temporal RNNs to track unnatural eye-blinking patterns.
  > Thank you. I am now glad to answer any questions."

---

## University Submission Screenshot Checklist

Capture and include the following screenshots in your final report and PowerPoint presentation:

- [ ] **Screenshot 1:** Streamlit Dashboard Landing Page showing sidebar configuration and status badges.
- [ ] **Screenshot 2:** Image Analysis Result for a REAL image (Green authentic banner, bounding box, probability breakdown).
- [ ] **Screenshot 3:** Image Analysis Result for a FAKE image (Red manipulated banner, bounding box, probability breakdown).
- [ ] **Screenshot 4:** Video Analysis Result showing the interactive frame-by-frame probability timeline chart and face gallery.
- [ ] **Screenshot 5:** Evaluation Tab showing the Confusion Matrix heatmap (`outputs/confusion_matrix.png`).
- [ ] **Screenshot 6:** ROC Curve plot (`outputs/roc_curve.png`).
- [ ] **Screenshot 7:** Training and Validation convergence loss/accuracy curve (`outputs/training_curves.png`).
- [ ] **Screenshot 8:** Terminal execution showing successful automated tests (`pytest`).
