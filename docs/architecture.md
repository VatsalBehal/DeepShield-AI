# DeepShield AI — System Architecture and Data-Flow Specifications

## 1. High-Level System Architecture

The following diagram illustrates the complete software architecture of DeepShield AI, from dataset acquisition to web application serving.

```mermaid
flowchart TD
    subgraph Data_Pipeline["1. Data Acquisition & Preprocessing"]
        A["Academic Video Dataset (FaceForensics++ / Celeb-DF)"] --> B["Video-Level Split (70% Train / 15% Val / 15% Test)"]
        B -->|Leakage Prevention| C["Frame Extraction (OpenCV)"]
        C --> D["Haar-Cascade Face Detection & Margin Padding"]
        D --> E["Image Normalization (224x224, ImageNet μ/σ)"]
    end

    subgraph Training_Cloud["2. Cloud Model Training (Google Colab T4 GPU)"]
        E --> F["DeepShield ResNet-18 Backbone"]
        F --> G["Custom Classification Head (Dropout + Linear)"]
        G --> H["Cross-Entropy Loss & AdamW Optimizer"]
        H --> I["Validation & Early Stopping"]
        I --> J["Saved Checkpoint (models/best_model.pth)"]
    end

    subgraph Local_Inference["3. Local Windows Deployment (8GB RAM / CPU)"]
        J -->|Transfer Checkpoint| K["Local Model Weights"]
        L["User Input Media (Image or Video)"] --> M["Face Localization & Preprocessing"]
        K & M --> N["ResNet-18 Feed-Forward Inference"]
        N --> O["Calibrated Softmax Probabilities"]
        O --> P["Streamlit Interactive Web Application (app.py)"]
    end
```

---

## 2. Image Inference Data-Flow Diagram (DFD)

```mermaid
flowchart LR
    Input["Input Image File"] --> Read["Safe Image Loader (PIL/OpenCV)"]
    Read --> Detect{"Face Detected?"}
    Detect -->|Yes| Crop["Crop Face ROI + 20% Margin"]
    Detect -->|No| Full["Fallback: Full Frame with Warning"]
    Crop --> Trans["Transform & Normalize (3x224x224)"]
    Full --> Trans
    Trans --> Model["ResNet-18 Forward Pass"]
    Model --> Logits["Raw Logits [z0, z1]"]
    Logits --> Softmax["Softmax Layer"]
    Softmax --> Probs["P(Real) vs P(Fake)"]
    Probs --> Threshold{"P(Fake) >= Threshold?"}
    Threshold -->|Yes| Fake["Verdict: MANIPULATED (FAKE)"]
    Threshold -->|No| Real["Verdict: AUTHENTIC (REAL)"]
```

---

## 3. Video Frame Sampling and Score Aggregation Pipeline

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant App as Streamlit Web App / CLI
    participant Video as OpenCV VideoCapture
    participant Detector as Haar Face Detector
    participant Model as ResNet-18 Classifier
    participant Aggregator as Score Aggregator

    User->>App: Upload Video File (.mp4 / .avi)
    App->>Video: Open Video Stream & Extract Metadata
    loop For Every Nth Frame (Sample Interval = 10)
        Video->>App: Read BGR Frame
        App->>Detector: Detect Human Face
        alt Face Detected
            Detector-->>App: Face ROI Coordinates & Crop
            App->>Model: Forward Pass (Cropped Face)
            Model-->>App: Frame Fake Probability P_fake
        else No Face Found
            Detector-->>App: None (Skip Frame from Aggregation)
        end
    end
    App->>Aggregator: Collect List of [P_fake_1, P_fake_2, ...]
    Aggregator->>App: Compute Aggregated Score (Mean / Trimmed Mean)
    App-->>User: Display Final Verdict, Timeline Chart & Face Previews
```

---

## 4. Google Colab to Local Windows Deployment Architecture

```mermaid
flowchart TD
    subgraph Colab_Environment["Google Colab Training Workflow"]
        A1["DeepShield_AI_Training.ipynb"] --> A2["Download Dataset / Pipeline Test Data"]
        A2 --> A3["Train ResNet-18 on NVIDIA T4 GPU"]
        A3 --> A4["Evaluate Test Metrics (outputs/confusion_matrix.png)"]
        A4 --> A5["Export Artifact: best_model.pth"]
    end

    subgraph Transfer["Artifact Migration"]
        A5 -->|Download to PC| B1["Local Project Folder: models/best_model.pth"]
    end

    subgraph Local_Windows["Local Windows Environment (Python 3.10)"]
        B1 --> C1["Streamlit Web Server (app.py)"]
        B1 --> C2["CLI Image Inference (src/predict_image.py)"]
        B1 --> C3["CLI Video Inference (src/predict_video.py)"]
        B1 --> C4["Automated Pytest Suite (tests/)"]
    end
```
