# DeepShield AI — Comprehensive Viva Voce Questions & Answers
### B.Tech Artificial Intelligence & Data Science Capstone Project

This document contains 45 essential viva voce examination questions organized into 6 thematic categories to help you confidently demonstrate and defend your project before faculty evaluators.

---

## Category 1: Project Motivation, Objectives & Fundamentals

### Q1: What is a Deepfake, and how did the term originate?
**Answer:** The term "Deepfake" is a portmanteau of "Deep Learning" and "Fake." It refers to synthetic digital media (images, videos, or audio) generated or manipulated using artificial intelligence techniques—primarily Deep Neural Networks, Autoencoders, and Generative Adversarial Networks (GANs)—to replace or alter a person's likeness, voice, or expressions convincingly.

### Q2: What specific problem does DeepShield AI solve?
**Answer:** Generative AI tools allow malicious actors to fabricate synthetic videos for financial fraud, political disinformation, identity theft, and non-consensual media. DeepShield AI solves this problem by providing an automated, reproducible computer vision pipeline that extracts facial features and classifies media as either AUTHENTIC (REAL) or MANIPULATED (FAKE), producing calibrated probability scores and visual diagnostic evidence.

### Q3: Why is deepfake detection considered an "arms race" in cybersecurity?
**Answer:** As generative algorithms (GANs, diffusion models) improve, they produce fewer perceptual artifacts. In response, detection models must evolve to identify subtle spatial, spectral, and temporal inconsistencies. Every advance in detection pushes creators of deepfakes to refine their models, creating a continuous adversarial cycle.

### Q4: Can your system guarantee 100% authenticity of any uploaded video?
**Answer:** No. In computer vision and machine learning, no binary classifier can guarantee 100% certainty. DeepShield AI outputs a calibrated statistical likelihood under its learned parameter space. Unseen generation algorithms, severe video re-compression, motion blur, and poor illumination can cause classification errors.

### Q5: What is Supervised Learning, and how does it apply to this project?
**Answer:** Supervised learning is a machine learning paradigm where the algorithm is trained on input-output pairs. In DeepShield AI, each training image is labeled either `0` for `REAL` or `1` for `FAKE`. The model learns a mathematical mapping from pixel intensities to these two discrete categories.

---

## Category 2: Dataset Methodology & Data Leakage Prevention

### Q6: Which academic benchmark datasets did you investigate for this project?
**Answer:** We evaluated three primary academic benchmarks:
1. **FaceForensics++ (TUM):** 1,000 pristine YouTube videos altered via 4 manipulation methods (Deepfakes, Face2Face, FaceSwap, NeuralTextures).
2. **Celeb-DF v2:** High-visual-quality celebrity manipulations addressing boundary and color inconsistencies.
3. **Deepfake Detection Challenge (DFDC by Meta):** Large-scale diverse dataset with over 100,000 clips across various lighting conditions and demographics.

### Q7: What is "Data Leakage" in video-based deepfake detection, and how did you prevent it?
**Answer:** Data leakage occurs when consecutive frames extracted from the same source video appear in both the training set and the test set. Because adjacent video frames are virtually identical, the model merely memorizes the person's identity and background rather than learning manipulation artifacts, creating artificially inflated test accuracy. We prevented this by **splitting at the source video level before extracting frames**, ensuring all derivatives of a specific video remain strictly in one split.

### Q8: What dataset split ratio did you choose, and why?
**Answer:** We adopted a 70% Training, 15% Validation, and 15% Test split. The training set updates model weights; the validation set tunes hyperparameters and detects overfitting during training epochs; and the held-out test set provides unbiased performance measurement.

### Q9: Why shouldn't a student project use all 470 GB of the full DFDC dataset?
**Answer:** Training on a 470 GB dataset requires multi-GPU cloud clusters, enterprise storage, and hundreds of dollars in compute credits. For a student capstone, training on a curated, leakage-free subset of 2,000 to 5,000 validated frames on Google Colab's free NVIDIA T4 GPU provides sound empirical validation without exceeding practical constraints.

### Q10: How do you handle corrupted or unreadable images in your dataset pipeline?
**Answer:** Our `prepare_dataset.py` script validates every file using Pillow's `Image.verify()` and safe RGB array conversion. Any zero-byte files, corrupted headers, or non-image files are excluded from `manifest.csv` and logged in `dataset_summary.json`.

---

## Category 3: Computer Vision & Face Preprocessing

### Q11: Why is face detection and cropping a crucial preprocessing step?
**Answer:** In deepfake videos, manipulation is concentrated entirely on facial landmarks (eyes, mouth, jawline, skin texture), while background scenery remains unaltered. If we feed full frames into the CNN, the network may learn spurious correlations with background objects. Cropping isolates the region of interest (ROI) where artifacts reside.

### Q12: Why did you add a 20% margin around the detected face bounding box?
**Answer:** Deepfake synthesis algorithms (such as DeepFaceLab or FaceSwap) paste a manipulated face mask onto a source body, frequently leaving visible color mismatches, blurring, or seam artifacts along the jawline and forehead hairline. A 20% margin ensures these critical peripheral boundary artifacts are captured inside the crop.

### Q13: What happens if an uploaded image contains no detectable human face?
**Answer:** Our `FaceDetector` handles this gracefully: if OpenCV finds zero faces, it logs a warning, flags `face_detected = False`, and falls back to evaluating the centered image frame while alerting the user in the Streamlit UI that no face was localized.

### Q14: Why do we resize images to 224×224 pixels?
**Answer:** Pretrained Torchvision convolutional architectures (including ResNet-18) were designed and optimized for ImageNet images scaled to $224 \times 224$ pixels. Using standard dimensions ensures compatibility with the architecture's convolutional filter geometry.

### Q15: Why is image normalization necessary, and what are ImageNet mean and std values?
**Answer:** Normalization scales pixel values from $[0, 255]$ to standardized distributions with zero mean and unit variance. Because our ResNet-18 weights were pretrained on ImageNet, we must normalize using ImageNet's exact channel statistics: Mean = $[0.485, 0.456, 0.406]$ and Std = $[0.229, 0.224, 0.225]$. This stabilizes gradient descent during training.

### Q16: What data augmentations did you apply during training, and why?
**Answer:** We used `RandomHorizontalFlip` (p=0.5) and slight `ColorJitter` (brightness, contrast, saturation = 0.1). We avoided heavy Gaussian blurs or severe random perspective warping because those could artificially destroy the subtle high-frequency manipulation artifacts we need the model to learn.

---

## Category 4: Deep Learning Model Architecture (ResNet-18)

### Q17: What is a Convolutional Neural Network (CNN)?
**Answer:** A CNN is a specialized deep neural network designed for spatial data such as images. It uses parameter-sharing convolution kernels (filters) that slide across pixel matrices to extract hierarchical visual features—ranging from low-level edges and textures in early layers to complex facial geometries in deeper layers.

### Q18: What is the Vanishing Gradient Problem in deep neural networks?
**Answer:** During backpropagation, gradients are propagated backward by multiplying weight matrices layer by layer using the chain rule. In very deep networks, repeated multiplication of small fractional derivatives causes gradients to shrink exponentially toward zero, preventing earlier layers from updating their weights.

### Q19: How does ResNet (Residual Network) solve the vanishing gradient problem?
**Answer:** ResNet introduces **Residual Skip Connections** (shortcut connections) that bypass one or more convolutional layers. Instead of learning an underlying mapping $\mathcal{H}(x)$, the residual block learns a residual mapping $\mathcal{F}(x) = \mathcal{H}(x) - x$. The output becomes $\mathcal{F}(x) + x$. During backpropagation, gradients can flow unimpeded directly through the shortcut ($+1$ term), enabling stable training of deep networks.

### Q20: Why did you choose ResNet-18 specifically over ResNet-50 or Vision Transformers?
**Answer:** ResNet-18 has approximately 11.2 million parameters and a compact model size of ~45 MB, compared to 25 million for ResNet-50 and 86+ million for ViT. It achieves fast inference (under 50ms per frame on a standard CPU) and low memory consumption, making it ideal for deployment on an 8 GB RAM laptop without sacrificial loss of feature representation.

### Q21: What is Transfer Learning, and why is it beneficial here?
**Answer:** Transfer learning leverages a model trained on a massive source dataset (ImageNet with 1.2M images across 1,000 categories) and adapts its learned feature representations to a new target domain (deepfake detection). Instead of training randomly initialized weights from scratch—which requires hundreds of thousands of images—transfer learning converges faster and achieves higher accuracy with smaller datasets.

### Q22: What modifications did you make to the standard ResNet-18 architecture?
**Answer:** We replaced the default 1,000-class output fully-connected (`fc`) layer with:
1. A **Dropout layer** (`p = 0.3`) to prevent co-adaptation of neurons and mitigate overfitting.
2. A **Linear layer** mapping the 512 backbone features to `2` output classes (`REAL` and `FAKE`).

### Q23: What is the purpose of Dropout in the classification head?
**Answer:** During training, Dropout randomly deactivates 30% of neuron outputs with each forward pass. This forces the remaining neurons to learn independent, robust representations rather than memorizing redundant combinations of features, effectively acting as an ensemble regularizer.

---

## Category 5: Training, Loss & Optimization

### Q24: What Loss Function did you use, and what is its mathematical formula?
**Answer:** We used **Binary Cross-Entropy Loss** implemented via PyTorch's `nn.CrossEntropyLoss()`, which combines LogSoftmax with Negative Log-Likelihood:
$$\mathcal{L} = - \sum_{i=1}^{C} y_i \log(\hat{y}_i)$$
Where $C=2$ classes, $y_i$ is the ground-truth one-hot label, and $\hat{y}_i$ is the predicted probability.

### Q25: Which optimizer did you select, and how does AdamW differ from standard Adam?
**Answer:** We selected **AdamW** (Adaptive Moment Estimation with Decoupled Weight Decay). Standard Adam applies $L_2$ regularization directly to the gradient, which distorts adaptive moment estimates. AdamW decouples weight decay directly from gradient updates, resulting in better generalization and cleaner parameter shrinkage.

### Q26: What is a Learning Rate Scheduler, and why did you use `ReduceLROnPlateau`?
**Answer:** A learning rate scheduler dynamically adjusts the step size during training. `ReduceLROnPlateau` monitors validation loss: when the loss ceases to decrease for a designated number of epochs (patience = 2), the scheduler cuts the learning rate in half (factor = 0.5), allowing the optimizer to settle into narrow local minima.

### Q27: What is an Epoch, and what is Batch Size?
**Answer:**
- **Epoch:** One complete pass through the entire training dataset.
- **Batch Size:** The number of training samples processed in one forward and backward pass before updating model weights (we used 16 locally, 32 in Google Colab).

### Q28: What is Overfitting, and what mechanisms prevent it in your project?
**Answer:** Overfitting occurs when a model memorizes idiosyncrasies and noise in the training set, performing poorly on unseen validation data. We counter overfitting using:
1. Pretrained feature representations via transfer learning.
2. Dropout regularization ($p=0.3$).
3. Weight decay (AdamW regularization = $0.0001$).
4. Data augmentation (random flips and color variations).
5. Early stopping based on validation loss.

### Q29: What information is stored inside `models/best_model.pth`?
**Answer:** The checkpoint file contains:
- `model_state_dict`: Learned weights and biases of all convolutional and linear layers.
- `optimizer_state_dict`: Optimizer momentum and variance states.
- Metadata: Best epoch number, validation loss, validation accuracy, class mapping dictionary (`{"real": 0, "fake": 1}`), and preprocessing parameters.

---

## Category 6: Evaluation Metrics, Video Inference & Deployment

### Q30: What is a Confusion Matrix, and what are its four quadrants?
**Answer:** A confusion matrix is a $2 \times 2$ table summarizing classification performance:
- **True Negative (TN):** Genuine images correctly classified as REAL.
- **False Positive (FP):** Genuine images incorrectly flagged as FAKE (Type I error).
- **False Negative (FN):** Manipulated deepfakes incorrectly missed as REAL (Type II error).
- **True Positive (TP):** Manipulated deepfakes correctly detected as FAKE.

### Q31: In deepfake detection, is Precision or Recall more important?
**Answer:** Both are critical depending on the application context:
- **High Recall** is crucial in security-sensitive screening to catch every possible deepfake (minimizing False Negatives).
- **High Precision** is vital in social media moderation to ensure authentic users are not unfairly censored or accused of fraud (minimizing False Positives).
- The **F1-Score** balances both as their harmonic mean:
$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### Q32: What is an ROC Curve and what does AUC signify?
**Answer:** The Receiver Operating Characteristic (ROC) curve plots the True Positive Rate (Sensitivity) against the False Positive Rate ($1 - \text{Specificity}$) across all classification thresholds from 0.0 to 1.0. The Area Under the Curve (AUC) summarizes diagnostic discrimination ability: an AUC of 0.5 represents random guessing, while 1.0 represents perfect classification.

### Q33: How does video inference work in DeepShield AI?
**Answer:** Videos are decoded using OpenCV's `cv2.VideoCapture`. Rather than analyzing all 30 frames per second (which is redundant and slow), we sample frames at configurable intervals (e.g. every 10th frame). Faces are localized and cropped, passed through ResNet-18 to compute frame-level fake probabilities, and aggregated into an overall authenticity score.

### Q34: What score aggregation methods does your video pipeline support?
**Answer:**
1. **Mean (Default):** Arithmetic average of all frame probabilities. Resilient to isolated noise.
2. **Median:** 50th percentile score. Highly robust to single-frame outliers or momentary blurs.
3. **Max:** Flags the maximum detected fake probability. Sensitive to single-frame manipulations, but prone to false alarms.
4. **Trimmed Mean:** Discards the top and bottom 10% outlier scores before averaging.

### Q35: Why shouldn't a single anomalous frame cause an entire video to be flagged as FAKE?
**Answer:** Transient video compression artifacts, rapid head movements, or momentary camera defocus can cause a single frame to look degraded, resulting in a false-positive spike. Requiring sustained high scores across multiple frames ensures robust detection.

### Q36: What is "Domain Shift" or "Cross-Dataset Degradation"?
**Answer:** A machine learning model performs best when test data comes from the same distribution as its training data. If a detector is trained exclusively on FaceForensics++ (YouTube videos manipulated with FaceSwap), its accuracy drops when tested on Celeb-DF or diffusion-based deepfakes because lighting, compression algorithms, and synthesis artifacts differ fundamentally.

### Q37: How does your Streamlit web application handle user media privacy?
**Answer:** DeepShield AI processes all uploaded images and videos locally on the host machine in volatile memory and temporary system directories. No data is stored in permanent databases or transmitted to third-party cloud APIs. Temporary files are unlinked and released after analysis.

### Q38: Why is CPU execution appropriate for the local deployment on Windows?
**Answer:** Model training requires high parallel throughput (calculating millions of gradient updates across batches), which necessitates a GPU. In contrast, inference executes a single forward pass with frozen weights on a single image ($1 \times 3 \times 224 \times 224$), which ResNet-18 executes in under 50 milliseconds on modern Intel/AMD laptop CPUs.

### Q39: What are the ethical implications of deepfake detection technology?
**Answer:** Detectors must be deployed responsibly. False accusations of deepfaking can damage reputations, undermine genuine evidence in legal proceedings, or cause political controversy. Automated systems should serve as decision-support tools alongside human forensic experts rather than acting as unilateral truth arbiters.

### Q40: What are the primary future enhancements for DeepShield AI?
**Answer:**
1. **Multi-Modal Analysis:** Integrating audio forensics (detecting synthetic voice cloned with models like ElevenLabs).
2. **Temporal Modeling:** Adding an LSTM, GRU, or Temporal Convolutional Network (TCN) to analyze unnatural eye-blinking rates and head pose inconsistencies across frames.
3. **Explainability (XAI):** Implementing Grad-CAM (Gradient-weighted Class Activation Mapping) heatmaps to visually highlight the exact facial regions influencing the model's decision.
