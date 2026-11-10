# Hierarchical Multimodal Biometric Verification System

## 1. Project Overview

This project implements a **Hierarchical Multimodal Biometric Verification System** that verifies a person's identity using two biometric modalities:

- **Face Recognition:** Extracting facial features using Histogram of Oriented Gradients (HOG) and Local Binary Patterns (LBP).
- **Voice Recognition:** Extracting acoustic features using Mel-Frequency Cepstral Coefficients (MFCC), Zero-Crossing Rate (ZCR), and Root Mean Square Energy (RMS).

The system explores unimodal and multimodal biometric verification approaches using traditional Machine Learning algorithms.

It implements four main verification approaches:

1. Face-only Verification.
2. Voice-only Verification.
3. Early Feature Fusion.
4. Hierarchical Face-Voice Verification.

The main objective is to compare individual biometric verification with multimodal verification and investigate whether combining facial and voice information improves identity verification performance.

**Important:** This project uses traditional Machine Learning techniques and handcrafted feature extraction methods rather than Deep Learning models.

---

## 2. Dataset

### VidTIMIT Dataset

The project uses the publicly available **VidTIMIT dataset**, which contains video and audio recordings of multiple speakers.

Dataset characteristics:

- 43 speakers.
- 19 female and 24 male participants.
- 3 recording sessions.
- 10 speech recordings per speaker.
- 430 audio recordings in total.
- 103,542 extracted video frames.

### Dataset Split

To prevent recording leakage and evaluate the system across different recording sessions, the dataset is divided as follows:

| Session | Recordings per Speaker | Purpose |
|---|---:|---|
| Session 1 | 6 | Training and Enrollment |
| Session 2 | 2 | Validation and Threshold Selection |
| Session 3 | 2 | Final Testing |

The same speaker identities are represented in all three sessions.

The system uses Session 1 to establish enrolled identities, Session 2 to select operating thresholds, and Session 3 for independent final evaluation.

### Dataset Citation

C. Sanderson and B. C. Lovell, "Multi-Region Probabilistic Histograms for Robust and Scalable Identity Inference," in *Lecture Notes in Computer Science*, vol. 5558, pp. 199–208, 2009.

---

## 3. System Architecture

The system consists of five major stages:

1. Data Preprocessing.
2. Feature Extraction.
3. Unimodal Verification.
4. Multimodal Fusion.
5. Evaluation and Comparison.

### Verification Approaches

#### A. Face-only Verification

Facial features are extracted from video frames and used to train traditional Machine Learning classifiers.

Feature extraction methods:

- HOG (Histogram of Oriented Gradients).
- LBP (Local Binary Patterns).

Models evaluated:

- Support Vector Machine (SVM).
- Random Forest (RF).

#### B. Voice-only Verification

Acoustic features are extracted from WAV recordings.

Feature extraction methods:

- MFCC (Mel-Frequency Cepstral Coefficients).
- ZCR (Zero-Crossing Rate).
- RMS (Root Mean Square Energy).

Models evaluated:

- Support Vector Machine (SVM).
- Random Forest (RF).

#### C. Early Feature Fusion

Early Fusion combines the extracted Face and Voice feature vectors into a single multimodal representation before classification.

The combined feature vector contains:

- 3,548 Face features.
- 30 Voice features.
- 3,578 total multimodal features.

The fused representation is used to train a traditional Machine Learning verification classifier.

#### D. Hierarchical Face-Voice Verification

The Hierarchical approach uses a sequential two-stage verification pipeline.

**Stage 1 — Face Verification**

The Face SVM evaluates the claimed identity.

Based on its verification score:

- High-confidence scores are accepted directly.
- Low-confidence scores are rejected directly.
- Uncertain scores are escalated to Voice verification.

**Stage 2 — Voice Verification**

Only uncertain Face trials are passed to the Voice SVM.

The Voice model evaluates the claim and produces the final accept/reject decision.

This approach avoids unnecessary Voice inference when the Face result is already sufficiently confident.

---

## 4. Feature Extraction

### Face Features

Face preprocessing includes:

- Face detection using Haar Cascade classifiers.
- Selection of the largest detected face.
- Face-region padding.
- Grayscale conversion.
- Image resizing to 128 × 128 pixels.

Feature extraction uses:

| Method | Description |
|---|---|
| HOG | Extracts local gradient and edge-orientation information. |
| LBP | Describes local texture patterns using neighborhood intensity comparisons. |

The final Face representation contains 3,548 features per recording.

### Voice Features

Audio preprocessing and feature extraction use:

| Feature | Description |
|---|---|
| MFCC | Represents the spectral characteristics of speech. |
| ZCR | Measures the rate of signal sign changes. |
| RMS | Represents the signal's energy level. |

The final Voice representation contains 30 features per recording.

---

## 5. Machine Learning Models

The project uses traditional Machine Learning algorithms.

| Modality | Models |
|---|---|
| Face | Support Vector Machine (SVM), Random Forest |
| Voice | Support Vector Machine (SVM), Random Forest |
| Early Fusion | SVM, Random Forest |
| Hierarchical | Face SVM + Voice SVM |

The SVM pipelines use feature scaling and dimensionality reduction where configured.

The Random Forest models use multiple decision trees and class balancing.

---

## 6. Verification Methodology

The project implements biometric verification rather than relying only on multiclass identity classification.

Each verification trial compares a probe recording against an enrolled identity.

The pairwise feature representation is based on the absolute difference between the probe feature vector and the enrolled identity's reference representation.

The system produces a binary decision:

- **Genuine:** The probe is accepted as belonging to the claimed identity.
- **Impostor:** The probe is rejected as not belonging to the claimed identity.

### Experimental Protocol

| Dataset Session | Role |
|---|---|
| Session 1 | Training and Enrollment |
| Session 2 | Validation and Threshold Selection |
| Session 3 | Final Evaluation |

Thresholds are selected using validation data, while the final reported performance is measured on the independent test session.

---

## 7. Evaluation Metrics

The system is evaluated using the following metrics:

| Metric | Description |
|---|---|
| Accuracy | Overall proportion of correct decisions. |
| Precision | Proportion of accepted genuine claims among all accepted claims. |
| Recall | Proportion of genuine claims correctly accepted. |
| F1-Score | Harmonic mean of Precision and Recall. |
| FAR | False Acceptance Rate: proportion of impostor claims incorrectly accepted. |
| FRR | False Rejection Rate: proportion of genuine claims incorrectly rejected. |
| ROC-AUC | Area under the Receiver Operating Characteristic curve, where continuous scores are available. |

Confusion matrices, ROC curves, and FAR-FRR curves are generated where applicable.

For the final Hierarchical cascade, the system produces binary decisions. Therefore, ROC-AUC is not reported for that cascade unless a valid continuous combined score is defined.

---

## 8. Experimental Results

The following table reports the current Session 3 evaluation results.

| System | Accuracy | Precision | Recall | F1-Score | FAR | FRR |
|---|---:|---:|---:|---:|---:|---:|
| Face SVM | 98.59% | 68.09% | 74.42% | 71.11% | 0.83% | 25.58% |
| Voice SVM | 97.21% | 29.27% | 13.95% | 18.90% | 0.80% | 86.05% |
| Early Feature Fusion SVM | 98.40% | 62.86% | 76.74% | 69.11% | 1.08% | 23.26% |
| Hierarchical Face-Voice | 98.57% | 66.67% | 76.74% | 71.35% | 0.91% | 23.26% |

### Results Discussion

- Face SVM provides strong overall verification performance.
- Voice-only verification shows weaker genuine-user detection under the selected security threshold.
- Early Feature Fusion improves genuine-user recall compared with Face-only verification.
- Hierarchical verification achieves a higher F1-score than the Face-only and Early Fusion SVM baselines in the current experiment.
- The Hierarchical approach also achieves a lower FAR than Early Feature Fusion.
- The results demonstrate that multimodal integration can improve selected performance measures, although improvements are not universal across all metrics.

### Hierarchical Inference Statistics

The final Session 3 experiment contains 3,698 verification trials.

- Total trials: 3,698.
- Direct Face decisions: 3,668.
- Voice escalations: 30.

Voice inference is performed only for trials escalated by the Face stage.

### Important Limitation

The Session 2 validation set contains only a small number of genuine trials escalated to Voice verification. Therefore, the observed contribution of Voice within the Hierarchical pipeline should be interpreted cautiously.

---

## 9. Project Structure

```text
Biometric-Verification-System/
│
├── data/
│   ├── raw/
│   │   ├── archives/
│   │   └── vidtimit/
│   │
│   ├── processed/
│   │   ├── face/
│   │   ├── voice/
│   │   ├── splits/
│   │   └── ...
│   │
│   └── metadata/
│
├── src/
│   ├── preprocessing/
│   ├── features/
│   ├── models/
│   ├── verification/
│   ├── fusion/
│   ├── evaluation/
│   └── ...
│
├── saved_models/
│   ├── face/
│   ├── voice/
│   ├── fusion/
│   └── hierarchical/
│
├── results/
│   ├── face/
│   ├── voice/
│   ├── fusion/
│   └── hierarchical/
│
├── notebooks/
├── requirements.txt
└── README.md
```

---

## 10. Installation and Setup

### Prerequisites

- Python 3.12.
- pip.
- Git.

### Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd Biometric-Verification-System
```

### Create a Virtual Environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Ensure that the VidTIMIT dataset is downloaded and extracted into the expected data directories before running preprocessing or feature extraction scripts.

---

## 11. Running the Project

Run the scripts from the repository root directory.

### Step 1 — Dataset Preprocessing

```bash
python src/preprocessing/face_preprocessing.py
```

### Step 2 — Feature Extraction

```bash
python src/features/face_features.py
python src/features/voice_features.py
```

### Step 3 — Session Mapping and Dataset Preparation

```bash
python src/preprocessing/session_split.py
python src/preprocessing/prepare_datasets.py
```

### Step 4 — Unimodal Verification

```bash
python src/verification/face_verification.py
python src/verification/voice_verification.py
```

### Step 5 — Security Threshold Evaluation

```bash
python src/verification/face_security_thresholds.py
python src/verification/voice_security_thresholds.py
```

### Step 6 — Early Feature Fusion

```bash
python src/verification/early_fusion_verification.py
```

### Step 7 — Hierarchical Verification

```bash
python src/verification/hierarchical_verification.py
```

The Hierarchical implementation uses the validated audit policy and produces sequential evaluation outputs.

**Note:** The exact execution order and required input artifacts should be followed. If the dataset, processed features, or saved models already exist, the corresponding earlier stages do not need to be repeated.

---

## 12. Technologies Used

- Python 3.12
- NumPy
- Pandas
- OpenCV
- scikit-image
- Librosa
- SoundFile
- SciPy
- scikit-learn
- Matplotlib
- Seaborn
- Joblib

---

## 13. Conclusion

This project demonstrates a traditional Machine Learning-based multimodal biometric verification system using Face and Voice information.

It compares unimodal verification with Early Feature Fusion and Hierarchical Face-Voice verification under a session-based evaluation protocol.

The experiments highlight the advantages and limitations of individual biometric modalities and multimodal integration.

The Hierarchical approach demonstrates how a sequential verification pipeline can use Face as the primary modality and invoke Voice only when additional verification is needed.

The results provide an experimental comparison of security and usability trade-offs in multimodal biometric verification.