"""
DeepShield AI - Streamlit Web Application
Interactive user interface for deepfake image and video detection,
visualization of facial bounding boxes, frame-by-frame analysis, and metrics.
"""

import os
import tempfile
import time
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import cv2
import streamlit as st
import torch

from src.utils import load_config, get_device, load_checkpoint
from src.model import build_model
from src.preprocessing import preprocess_image, FaceDetector, get_eval_transforms
from src.predict_video import aggregate_scores, VALID_VIDEO_EXTENSIONS

# ------------------------------------------------------------------------------
# Page Configuration & Styling
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="DeepShield AI - Deepfake Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #2563EB;
        margin-bottom: 12px;
    }
    .verdict-fake {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        border-radius: 8px;
        padding: 18px;
        color: #991B1B;
        font-weight: bold;
        font-size: 1.3rem;
        text-align: center;
    }
    .verdict-real {
        background-color: #F0FDF4;
        border: 2px solid #22C55E;
        border-radius: 8px;
        padding: 18px;
        color: #166534;
        font-weight: bold;
        font-size: 1.3rem;
        text-align: center;
    }
    .disclaimer-box {
        background-color: #FFFBEB;
        border: 1px solid #F59E0B;
        border-radius: 6px;
        padding: 12px;
        font-size: 0.85rem;
        color: #92400E;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# Cached Model Loader
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cached_model(checkpoint_path: str, config_path: str = "configs/config.yaml"):
    """Loads and caches the PyTorch model to avoid repeated disk reads."""
    if not os.path.exists(checkpoint_path):
        return None, None, f"Checkpoint not found at: {checkpoint_path}"

    config = load_config(config_path)
    device = get_device()
    try:
        model = build_model(config)
        model, _, meta = load_checkpoint(checkpoint_path, model, device=device)
        model.eval()
        return model, config, meta
    except Exception as e:
        return None, None, str(e)


# ------------------------------------------------------------------------------
# Sidebar Controls
# ------------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("DeepShield AI")
    st.caption("AI & DS B.Tech Capstone Project")

    st.markdown("---")
    st.subheader("⚙️ System Configuration")

    checkpoint_path = st.text_input("Checkpoint Path", value="models/best_model.pth")
    confidence_thresh = st.slider("Classification Threshold", 0.10, 0.90, 0.50, 0.05,
                                 help="Probability cutoff for FAKE classification.")
    face_detect_toggle = st.checkbox("Enable Face Detection (OpenCV)", value=True,
                                    help="Crops detected face ROI before classification.")

    st.markdown("---")
    st.subheader("🎥 Video Analysis Parameters")
    frame_interval = st.slider("Frame Interval", 1, 30, 10, help="Analyze every Nth frame.")
    max_frames = st.slider("Max Frames", 5, 60, 30, help="Upper limit of frames sampled.")
    agg_method = st.selectbox("Aggregation Strategy", ["mean", "median", "max", "trimmed_mean"], index=0)

    st.markdown("---")
    # Status display
    if os.path.exists(checkpoint_path):
        st.success("✅ Model weights loaded")
    else:
        st.warning("⚠️ Weights not found. Run smoke test or download checkpoint.")

    st.markdown("---")
    st.markdown("🔒 **Privacy Guarantee**  \nUploaded media is processed locally in RAM and temporary memory. No files are retained or transmitted externally.")


# ------------------------------------------------------------------------------
# Main Application Header
# ------------------------------------------------------------------------------
st.markdown('<div class="main-title">DeepShield AI — Deepfake Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Automated forensic face analysis powered by Deep Learning (ResNet-18 Transfer Learning)</div>', unsafe_allow_html=True)

# Main Navigation Tabs
tabs = st.tabs([
    "🖼️ Image Analysis",
    "🎬 Video Analysis",
    "📊 Evaluation & Metrics",
    "🧠 Model Architecture & Methodology",
    "🎓 Viva & College Guide"
])


# ==============================================================================
# TAB 1: IMAGE INFERENCE
# ==============================================================================
with tabs[0]:
    st.subheader("Single Image Authenticity Verification")
    st.write("Upload an image containing a human face to analyze facial micro-textures and boundary artifacts.")

    uploaded_image = st.file_uploader("Choose an image (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"], key="img_uploader")

    if uploaded_image is not None:
        col_img1, col_img2 = st.columns([1, 1])

        try:
            image = Image.open(uploaded_image).convert("RGB")
            with col_img1:
                st.image(image, caption="Uploaded Image", use_container_width=True)
        except Exception as e:
            st.error(f"Could not read image: {e}")
            st.stop()

        if st.button("🚀 Analyze Image", type="primary", key="btn_img_analyze"):
            model, cfg, meta_or_err = get_cached_model(checkpoint_path)

            if model is None:
                st.error(f"Cannot run analysis: {meta_or_err}")
                st.info("Tip: If you haven't trained on Colab yet, run `python scripts/smoke_test.py` to create a test checkpoint for testing the interface.")
                st.stop()

            with st.spinner("Executing face detection and ResNet-18 feature extraction..."):
                start_t = time.time()
                prep_cfg = cfg.get("preprocessing", {}) if cfg else {}
                margin = prep_cfg.get("face_crop_margin", 0.20)
                image_size = prep_cfg.get("image_size", 224)

                img_np = np.array(image)
                face_found = False
                bbox = None
                display_img = image.copy()

                if face_detect_toggle:
                    detector = FaceDetector(margin=margin)
                    face_crop, bbox, face_found = detector.detect_and_crop(img_np)
                    if face_found:
                        pil_crop = Image.fromarray(face_crop)
                        # Draw bounding box on display preview
                        draw = ImageDraw.Draw(display_img)
                        x, y, w, h = bbox
                        draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
                    else:
                        pil_crop = image
                else:
                    pil_crop = image

                # Tensor transform and model evaluation
                transform = get_eval_transforms(image_size)
                tensor = transform(pil_crop).unsqueeze(0).to(get_device())

                with torch.no_grad():
                    logits = model(tensor)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

                elapsed = time.time() - start_t
                prob_real = float(probs[0])
                prob_fake = float(probs[1])

                is_fake = prob_fake >= confidence_thresh
                verdict = "FAKE" if is_fake else "REAL"
                confidence = prob_fake if is_fake else prob_real

            with col_img2:
                st.markdown("### Analysis Results")
                if is_fake:
                    st.markdown(f'<div class="verdict-fake">🚨 VERDICT: MANIPULATED (FAKE)<br><span style="font-size:1rem;">Estimated Fake Confidence: {prob_fake*100:.2f}%</span></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="verdict-real">🛡️ VERDICT: AUTHENTIC (REAL)<br><span style="font-size:1rem;">Estimated Real Confidence: {prob_real*100:.2f}%</span></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Probabilities breakdown
                col_p1, col_p2 = st.columns(2)
                col_p1.metric("Real Probability", f"{prob_real * 100:.2f}%")
                col_p2.metric("Fake Probability", f"{prob_fake * 100:.2f}%")

                st.progress(float(prob_fake), text=f"Fake Risk Score ({prob_fake*100:.1f}%)")

                # Detection details
                st.markdown("#### Detection Metadata")
                st.write(f"- **Face Detected:** {'Yes (Bounding box highlighted)' if face_found else 'No (Analyzed entire frame)'}")
                st.write(f"- **Input Dimensions:** {image.width} × {image.height} px")
                st.write(f"- **Inference Latency:** {elapsed * 1000:.1f} ms")
                st.write(f"- **Threshold Used:** {confidence_thresh:.2f}")

                if face_found:
                    with st.expander("🔍 View Cropped Facial Region"):
                        st.image(pil_crop, caption="Cropped Face with Margin", width=180)

            st.markdown("""
            <div class="disclaimer-box">
                <b>⚠️ University Demonstration Notice:</b> This result is an automated statistical inference generated by a machine-learning binary classifier. Deepfake detection models are susceptible to compression artifacts, unusual lighting, and unseen generative techniques. Predictions do not constitute conclusive forensic evidence.
            </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# TAB 2: VIDEO INFERENCE
# ==============================================================================
with tabs[1]:
    st.subheader("Video Sequence Authenticity Verification")
    st.write("Upload an MP4, AVI, or MOV video clip. The system samples frames, detects faces, and aggregates probabilities across the timeline.")

    uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv", "webm"], key="vid_uploader")

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_video.name)[1])
        tfile.write(uploaded_video.read())
        temp_video_path = tfile.name
        tfile.close()

        st.video(temp_video_path)

        if st.button("🚀 Analyze Video Sequence", type="primary", key="btn_vid_analyze"):
            model, cfg, meta_or_err = get_cached_model(checkpoint_path)

            if model is None:
                st.error(f"Cannot run analysis: {meta_or_err}")
                st.stop()

            prep_cfg = cfg.get("preprocessing", {}) if cfg else {}
            margin = prep_cfg.get("face_crop_margin", 0.20)
            image_size = prep_cfg.get("image_size", 224)

            detector = FaceDetector(margin=margin)
            transform = get_eval_transforms(image_size)

            cap = cv2.VideoCapture(temp_video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = max(1.0, cap.get(cv2.CAP_PROP_FPS))

            frame_idx = 0
            sampled_idx = 0
            frame_records = []
            preview_crops = []

            progress_bar = st.progress(0, text="Sampling frames and evaluating facial regions...")

            try:
                while sampled_idx < max_frames:
                    ret, bgr_frame = cap.read()
                    if not ret:
                        break

                    if frame_idx % frame_interval == 0:
                        sampled_idx += 1
                        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                        cropped_face, bbox, found = detector.detect_and_crop(rgb_frame)

                        if found:
                            pil_crop = Image.fromarray(cropped_face)
                            tensor = transform(pil_crop).unsqueeze(0).to(get_device())

                            with torch.no_grad():
                                logits = model(tensor)
                                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

                            p_fake = float(probs[1])
                            p_real = float(probs[0])

                            frame_records.append({
                                "Frame Index": frame_idx,
                                "Timestamp (s)": round(frame_idx / fps, 2),
                                "Face Detected": True,
                                "Fake Probability": p_fake,
                                "Real Probability": p_real
                            })

                            if len(preview_crops) < 6:
                                preview_crops.append((pil_crop, f"Frame {frame_idx} (Fake: {p_fake*100:.1f}%)"))
                        else:
                            frame_records.append({
                                "Frame Index": frame_idx,
                                "Timestamp (s)": round(frame_idx / fps, 2),
                                "Face Detected": False,
                                "Fake Probability": None,
                                "Real Probability": None
                            })

                        progress_bar.progress(min(1.0, sampled_idx / max_frames))

                    frame_idx += 1

            finally:
                cap.release()
                if os.path.exists(temp_video_path):
                    try:
                        os.unlink(temp_video_path)
                    except Exception:
                        pass

            progress_bar.empty()

            df_frames = pd.DataFrame(frame_records)
            usable_fakes = [r["Fake Probability"] for r in frame_records if r["Face Detected"] and r["Fake Probability"] is not None]

            if len(usable_fakes) == 0:
                st.warning("⚠️ No detectable human faces found in any sampled video frames. Unable to compute authenticity verdict.")
            else:
                agg_fake = aggregate_scores(usable_fakes, method=agg_method)
                agg_real = 1.0 - agg_fake
                is_fake = agg_fake >= confidence_thresh

                st.markdown("### Video Authenticity Summary")
                if is_fake:
                    st.markdown(f'<div class="verdict-fake">🚨 VIDEO VERDICT: MANIPULATED (FAKE)<br><span style="font-size:1rem;">Aggregate Fake Score: {agg_fake*100:.2f}% ({agg_method.upper()} aggregation)</span></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="verdict-real">🛡️ VIDEO VERDICT: AUTHENTIC (REAL)<br><span style="font-size:1rem;">Aggregate Real Score: {agg_real*100:.2f}% ({agg_method.upper()} aggregation)</span></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                col_v1, col_v2, col_v3 = st.columns(3)
                col_v1.metric("Total Video Frames", total_frames)
                col_v2.metric("Sampled Frames", len(frame_records))
                col_v3.metric("Frames with Faces", len(usable_fakes))

                # Timeline Chart
                st.markdown("#### Frame-by-Frame Fake Probability Timeline")
                chart_data = df_frames.dropna(subset=["Fake Probability"])[["Timestamp (s)", "Fake Probability"]]
                if not chart_data.empty:
                    st.line_chart(chart_data.set_index("Timestamp (s)"))

                # Sampled Crops Gallery
                if preview_crops:
                    st.markdown("#### Sampled Face Frames")
                    cols = st.columns(len(preview_crops))
                    for idx, (crp, lbl) in enumerate(preview_crops):
                        cols[idx].image(crp, caption=lbl, use_container_width=True)

                # Tabular Breakdown
                with st.expander("📋 View Full Frame-by-Frame Data Table"):
                    st.dataframe(df_frames, use_container_width=True)


# ==============================================================================
# TAB 3: EVALUATION & METRICS
# ==============================================================================
with tabs[2]:
    st.subheader("Model Performance on Held-Out Test Set")
    st.write("Quantitative experimental metrics evaluated against held-out validation and test data.")

    summary_file = "outputs/evaluation_summary.json"
    cm_file = "outputs/confusion_matrix.png"
    roc_file = "outputs/roc_curve.png"
    history_file = "outputs/training_history.json"
    curves_file = "outputs/training_curves.png"

    if os.path.exists(summary_file):
        with open(summary_file, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Test Accuracy", f"{eval_data['accuracy']*100:.2f}%")
        col_m2.metric("Precision (Fake)", f"{eval_data['precision_fake']*100:.2f}%")
        col_m3.metric("Recall (Fake)", f"{eval_data['recall_fake']*100:.2f}%")
        col_m4.metric("F1-Score (Fake)", f"{eval_data['f1_score_fake']*100:.2f}%")

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            if os.path.exists(cm_file):
                st.image(cm_file, caption="Confusion Matrix Heatmap", use_container_width=True)
        with col_g2:
            if os.path.exists(roc_file):
                st.image(roc_file, caption="Receiver Operating Characteristic (ROC)", use_container_width=True)

        if os.path.exists(curves_file):
            st.markdown("---")
            st.markdown("#### Training and Validation Convergence Curves")
            st.image(curves_file, caption="Loss and Accuracy History across Training Epochs", use_container_width=True)
    else:
        st.info("ℹ️ Evaluation metrics have not been generated yet.")
        st.markdown("""
        To compute test metrics and generate plots, execute the following command in PowerShell:
        ```powershell
        python src/evaluate.py --checkpoint models/best_model.pth --test_dir data/test
        ```
        """)


# ==============================================================================
# TAB 4: ARCHITECTURE & METHODOLOGY
# ==============================================================================
with tabs[3]:
    st.subheader("DeepShield AI System Architecture & Engineering Principles")

    st.markdown("""
    ### 1. Convolutional Neural Network Backbone: ResNet-18
    DeepShield AI utilizes **ResNet-18** (Residual Network with 18 layers) as its core feature extractor:
    - **Residual Skip Connections:** Solves the vanishing gradient problem in deep networks through identity mapping:
      $$y = \mathcal{F}(x, \{W_i\}) + x$$
    - **Computational Efficiency:** ResNet-18 contains approximately 11.2 million parameters, allowing fast inference on standard consumer laptops (8 GB RAM / CPU) without requiring a high-end discrete GPU.
    - **Transfer Learning:** Pretrained on ImageNet (1.2 million images), the network has already learned low-level visual primitives (edges, gradients, textures) and high-level structural features.

    ### 2. End-to-End Pipeline
    1. **Input Acquisition:** Image or video frame received.
    2. **Face Localization & Padding:** OpenCV Haar Cascade detects the face bounding box; a 20% margin is added to incorporate jawline and boundary artifacts.
    3. **Normalization:** Transformed to 224×224 pixels and standardized using ImageNet color statistics:
       $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
    4. **Forward Inference:** Dense features are mapped to 2 output logits, followed by Softmax:
       $$P(\text{Fake}) = \frac{e^{z_{\text{fake}}}}{e^{z_{\text{real}}} + e^{z_{\text{fake}}}}$$
    5. **Aggregation:** Video frames are combined via arithmetic mean or trimmed mean.

    ### 3. Understanding Detector Limitations & Domain Shift
    - **Domain Shift:** Models trained on a specific generator (e.g. FaceSwap) may struggle when tested on completely unseen models (e.g. Diffusion models or StyleGAN3).
    - **Compression Artifacts:** Aggressive social media video re-encoding (H.264/H.265 compression) removes subtle high-frequency artifacts, degrading detection accuracy.
    """)


# ==============================================================================
# TAB 5: VIVA & COLLEGE GUIDE
# ==============================================================================
with tabs[4]:
    st.subheader("University Project Submission & Viva Voce Preparation")

    st.markdown("""
    ### Key Concepts for Teacher Demonstration

    #### Q1: Why did you choose ResNet-18 instead of a larger model like ResNet-50 or ViT?
    **Answer:** ResNet-18 has 11.2 million parameters and runs in under 50ms on CPU, making it perfectly suited for an 8 GB RAM laptop during live university viva presentations. Larger models risk overfitting on modest student datasets and cause memory exhaustion.

    #### Q2: What is Data Leakage, and how did you prevent it?
    **Answer:** If consecutive video frames from the same video are randomly scattered between training and test sets, the model simply memorizes the person's face rather than deepfake artifacts. We prevented this by **splitting at the source video level**, ensuring all frames of any video remain strictly in one split.

    #### Q3: Why is Precision and Recall more important than raw Accuracy?
    **Answer:** In deepfake detection, class distributions are often imbalanced. A False Positive incorrectly labels a genuine person as a fraud, whereas a False Negative allows a malicious deepfake through. Precision measures correctness of fake alerts, while Recall measures how many fakes were caught.

    #### Q4: Why can't the model guarantee 100% authenticity?
    **Answer:** The model outputs a statistical probability, not a cryptographic signature. Compression noise, lighting variations, and new generation algorithms introduce domain shifts where the model may misclassify.
    """)
