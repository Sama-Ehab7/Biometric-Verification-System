# Biometric Verification System

## Project overview

This project compares traditional Machine Learning biometric verification using VidTIMIT face and voice recordings. It implements Face-only, Voice-only, Early Feature Fusion, and sequential Hierarchical Face-to-Voice verification. It uses handcrafted features with SVM and Random Forest models; Deep Learning is not used by the verification pipelines.

This README includes the technical summary and results; the detailed completion inventory and final report are in `PROJECT_COMPLETION_AUDIT.md` and `FINAL_PROJECT_COMPLETION_REPORT.md`.

## Data and evaluation protocol

The main experiments use VidTIMIT: 43 speakers, three sessions, 430 recordings. Session 1 (258 recordings) is used for training and enrollment; Session 2 (86) is used for validation and threshold selection; Session 3 (86) is held for final testing. Pairwise evaluation expands held-out recordings into **3,698 trials**: 86 genuine and 3,612 impostor. Genuine labels are 1; impostor labels are 0.

The enrollment/validation/test roles are kept separate. Thresholds are selected from Session 2 and frozen for Session 3. FAR is FP/(FP+TN); FRR is FN/(FN+TP). Because the trials are imbalanced, accuracy should be read with precision, recall, FAR, and FRR.

## Features and verification systems

- **Face:** 3,548 features per recording from HOG and LBP, computed from preprocessed 128×128 face images.
- **Voice:** 30 features per recording: 13 MFCC coefficients with mean and standard deviation, plus mean and standard deviation for ZCR and RMS.
- **Early Feature Fusion:** concatenates the 3,548 Face and 30 Voice features into 3,578 values before pairwise verification-model training. This is feature-level fusion.
- **Hierarchical:** scores Face first. Face scores above the high threshold accept directly; scores below the low threshold reject directly; only uncertain trials are sent to Voice. Voice inference is sequential and is run only on escalated trials.

Accuracy, precision, recall/genuine acceptance rate, F1, FAR, FRR, ROC-AUC, and confusion matrices are reported where applicable. The final Hierarchical system returns binary cascade decisions, so **ROC-AUC is not reported** for that system.

## Session 3 results

Thresholds for Face, Voice, and Early Fusion use the 1% target-FAR operating point; achieved test FAR can differ from target. Hierarchical lists its three cascade thresholds. EER is not reported in the VidTIMIT result artifacts.

| System | Model | Accuracy | Precision | Recall | F1 | FAR | FRR | ROC-AUC | Threshold(s) | TN / FP / FN / TP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Face | SVM | 98.59% | 68.09% | 74.42% | 71.11% | 0.83% | 25.58% | 0.9657 | 0.290093 | 3582 / 30 / 22 / 64 |
| Face | Random Forest | 98.57% | 75.38% | 56.98% | 64.90% | 0.44% | 43.02% | 0.9182 | 0.166667 | 3596 / 16 / 37 / 49 |
| Voice | SVM | 97.21% | 29.27% | 13.95% | 18.90% | 0.80% | 86.05% | 0.8010 | 0.816066 | 3583 / 29 / 74 / 12 |
| Voice | Random Forest | 96.92% | 22.00% | 12.79% | 16.18% | 1.08% | 87.21% | 0.8140 | 0.476667 | 3573 / 39 / 75 / 11 |
| Early Fusion | SVM | 98.40% | 62.86% | 76.74% | 69.11% | 1.08% | 23.26% | 0.9684 | 0.217984 | 3573 / 39 / 20 / 66 |
| Early Fusion | Random Forest | 98.62% | 73.97% | 62.79% | 67.92% | 0.53% | 37.21% | 0.9267 | 0.160000 | 3593 / 19 / 32 / 54 |
| Hierarchical | Face SVM → Voice SVM | 98.57% | 66.67% | 76.74% | 71.35% | 0.91% | 23.26% | Not reported | 0.171650 / 0.290093 / 0.202839 | 3579 / 33 / 20 / 66 |

Hierarchical Session 2 FAR was 0.8029%. On Session 3 it made 3,668 direct Face decisions and escalated 30 trials to Voice. The sequential result matches the validated audit result; see the audit report for stage counts and comparison details.

## Fingerprint bonus

Fingerprint is a separate optional modality using FVC2002 Set B. DB1_B–DB3_B are the captured data used for the reported real-data evaluation; DB4_B is synthetic and kept separate. Identity names include the database prefix. The pipeline uses grayscale, CLAHE, median filtering, aspect-preserving resize/padding to 256×256, then HOG (8,100 values) and LBP (10 values), for **8,110 features** total.

The saved SVM and Random Forest results have high FRR at the validation-selected low-FAR operating point. A separate controlled comparison did not improve test FAR, so the original models remain authoritative. Fingerprint is not fused with VidTIMIT because the datasets do not share real identities.

Reusable, non-training inference for a claimed enrolled fingerprint is described in the Fingerprint audit section below. With the saved model and enrollment template available, run from the repository root:

```powershell
.venv\Scripts\python.exe src/verification/fingerprint_inference.py data/raw/fingerprint/DB1_B/101_7.tif DB1_B_101
```

This command reads the supplied image and saved artifacts and returns a score, frozen threshold, and ACCEPT/REJECT result. Do not pass `--create-enrollment-artifact` when using an existing submission workspace; that option writes a new template artifact.

## Optional feature status

| Component | Status |
|---|---|
| Face-only, Voice-only, Early Fusion, Hierarchical | Implemented and evaluated |
| Fingerprint bonus | Implemented as a separate modality; high FRR limitation documented |
| Speech-command bonus | Two-speaker Traditional ML classifier trained; small held-out set. Protected execution remains fail-closed |
| Real-time custom demo | Browser camera and microphone capture are implemented; physical browser/device end-to-end testing is pending |
| Custom Face | Separate grouped-split SVM/RF experiment and API; small held-out evaluation, see caveats below |
| Custom Voice | Experimental SVM speaker verifier and closed-set identifier; unknown-speaker rejection is not validated |
| Combined custom demo | Server-side all-of-three policy; assumed fingerprint mapping, not scientifically validated fusion |

Custom Face images and Voice/command recordings are present under `data/raw/custom/`. Windows Media Foundation decoded all 40 MP4 files to a separate 32 kHz mono WAV corpus. Voice data remains very small after duplicate grouping. FVC identities are not proven to correspond to team members, so the project does not claim scientifically validated three-modality fusion.

## Project layout

This is a conceptual guide to the main populated areas; it does not assert that every possible project folder exists.

```text
Biometric-Verification-System/
├── data/                 # Raw and processed datasets (not all may be distributed)
├── notebooks/            # Exploratory notebooks
├── results/              # Saved evaluation results and plots
├── saved_models/         # Saved estimators and policies
├── src/
│   ├── evaluation/
│   ├── features/
│   ├── preprocessing/
│   └── verification/
├── tests/                # Focused inference/model smoke tests
└── requirements.txt      # Core dependencies plus notebook/inspection tools
```

The original FastAPI backend and static HTML/JavaScript UI are in `backend/` and `ui/`. Open `ui/app.html` for the original research system and `ui/custom_team_demo.html` for custom Face, Voice, combined, command, and live capture demos. The custom fingerprint page is separate at `ui/custom_fingerprint_demo.html`.

### Start the application (no training)

From the repository root, use two terminals. These commands only start the API and static file server; they do not retrain models or regenerate evaluation artifacts.

```powershell
.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

```powershell
.venv\Scripts\python.exe -m http.server 5500 --bind 127.0.0.1
```

Open `http://127.0.0.1:5500/ui/app.html` for the original system, `http://127.0.0.1:5500/ui/custom_team_demo.html` for the custom demos, or `http://127.0.0.1:5500/ui/custom_fingerprint_demo.html` for the standalone Fingerprint page. The project slide deck is [PRESENTATION.html](PRESENTATION.html); open it through the same local server to present or print to PDF.

### Submission demonstration guide

Start from the repository root in two terminals. If a port is already occupied, stop the prior project server from the terminal where it was started; do not terminate an unknown process. The standard API port was occupied by an older backend during final verification, and that process returned 404 for the custom routes. The current repository backend was verified separately on port 8001. Before the demo, confirm that `/api/custom/voice/status`, `/api/custom/combined/status`, `/api/custom/commands/status`, and `/api/custom-fingerprint/test-images` return 200 from the selected API port.

Terminal 1 (API):

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Terminal 2 (static UI):

```powershell
.\.venv\Scripts\python.exe -m http.server 5500 --bind 127.0.0.1
```

Open `http://127.0.0.1:5500/ui/app.html` for original VidTIMIT Face→Voice verification, `http://127.0.0.1:5500/ui/custom_team_demo.html` for custom Face/Voice/combined/command/live-capture demos, and `http://127.0.0.1:5500/ui/custom_fingerprint_demo.html` for standalone Fingerprint. API checks are `http://127.0.0.1:8000/api/health` and `http://127.0.0.1:8000/api/identities`. To present the slides, open `http://127.0.0.1:5500/PRESENTATION.html`.

Recommended sequence and known outcomes from actual API requests against the current repository backend on port 8001:

1. Check health and identities first. The observed health response reported `mode=hierarchical`, the saved Face/Voice SVMs, `hierarchical_policy_audit.joblib`, and 43 VidTIMIT identities; `/api/identities` returned 43 speakers and no runtime-enrolled users.
2. Demonstrate original VidTIMIT Face→Voice using a claimed VidTIMIT identity such as `fadg0` and that speaker’s raw video frame plus matching audio from the same session. Session 3 files are final-test material; do not use them to tune anything. Actual output varies by probe; no universal ACCEPT claim is made here.
3. In the custom-team page, choose Nour or Sama first, start the webcam, capture a frame, and press **Verify Face**. The page shows score and threshold. If a physical camera is unavailable, use `data/raw/custom/face/test/nour/face_01.jpg` for an API smoke check; it previously returned score `0.8334847442`, threshold `0.6508407787`, ACCEPT. A file-backed sample is not a webcam test.
4. After the Face check, record or upload a WAV and press **Verify Voice**. For a deterministic API smoke check, use `data/processed/custom_audio_v3/converted/voice/test/nour/WhatsApp Audio 2026-11-11 at 10.46.47 PM.wav`; the manifest assigns it to test. It previously returned score `0.9531987383`, validation-derived threshold `0.8866929907`, ACCEPT. The test corpus contains only three unique identity recordings overall.
5. After Voice, upload a fingerprint image and press **Verify Fingerprint**. The live custom page accepts an uploaded image and calls the saved SVM/template through `/api/custom-fingerprint/verify-upload`. A repeatable test-split probe is `custom-data/fingerprint/test/nour/101_8.tif` (ACCEPT, score `0.5`, threshold `0.4540606025`); `101_7.tif` was REJECT (score `0.0018696814`). Nour/Sama↔FVC is only an assumed mapping.
6. The custom page displays each result and shows overall ACCEPT only when all three accept. The separate `/api/custom/combined/verify` endpoint remains available for mapped test-split probes; its earlier sample response accepted all three. Neither interface proves real identity association or scientifically validated fusion.
7. Demonstrate command classification with `data/processed/custom_audio_v3/converted/commands/test/nour/stop/stop_004.wav`. The actual probe was classified as `start` (score `0.4857059232`), illustrating the classifier’s limitations. Protected execution must remain disabled; `/api/custom/commands/execute` returned HTTP 503.
8. Camera endpoint smoke check: the `face_01.jpg` still image passed the one-face guard; `face_02.jpg` was rejected with HTTP 422 because no face was detected. This is an API test using image files, **not** a physical browser-camera test. Browser navigation, webcam capture, and microphone capture were not physically tested.

These results are sample-level API observations, separate from the saved offline evaluation metrics. Do not use the custom small-sample results to claim robust performance or unknown-speaker rejection. Leave protected actions disabled.

## Custom Team Face demo

Collected team Face images are present at `data/raw/custom/face/{train,validation,test}/{nour,sama}` (the named `custom-data/face/` location is absent). All 40 files decode as images. The project Haar detector finds faces in 33/40; the isolated custom pipeline therefore uses its own centered square crop, resizes to 128×128 grayscale, and extracts the 3,548-value HOG/LBP sequence-compatible representation. It does not change the original VidTIMIT preprocessing.

The active isolated SVM and Random Forest models are in `saved_models/custom/face_v4/`; evaluation JSON files are in `results/custom/face_v4/`. A pHash screen grouped candidate near-duplicates before fitting. The adjusted split contains 25 train, 7 validation, and 8 test images. Thresholds were selected on validation before test evaluation. On 8 held-out images (16 correlated claims), SVM achieved 87.50% accuracy (TN/FP/FN/TP = 8/0/2/6), and RF achieved 100% (8/0/0/8). These are preliminary: only two known identities and no unknown-person examples exist; some test images form a near-duplicate pair within the test split.

Run the backend and serve `ui/` from the project root using the commands below, then open `http://127.0.0.1:5500/ui/custom_team_demo.html`. The custom Face endpoints are `/api/custom/face/verify` and `/api/custom/face/identify`. Fingerprint aliases Nour/Sama are an assumed demo mapping only; they do not prove that FVC identities correspond to the team members.

### Custom Voice, commands, and combined demo

`src/preprocessing/prepare_custom_audio.py` decoded 40 AAC MP4 and 20 WAV recordings to isolated 32 kHz mono PCM WAV under `data/processed/custom_audio_v3/`. Original files were not modified. Five exact duplicate pairs were assigned to a single split, leaving 15 unique identity clips: Nour 3/1/1 and Sama 6/2/2 for train/validation/test. The custom Voice SVM was selected on validation; on six held-out pairwise claims it achieved 83.33% accuracy, precision 100%, recall 66.67%, F1 0.80, FAR 0%, FRR 33.33%, AUC 1.00, TN/FP/FN/TP = 3/0/1/2. The test set has only three recordings (one Nour, two Sama), so these estimates are highly uncertain. No independent unknown-speaker samples exist.

The command classifier v3 uses 20/10/10 recordings across train/validation/test (four/two/two per command; both speakers occur in each split). Validation selected SVM (macro-F1 0.8933, accuracy 90%; RF tied). Test accuracy was 70%, macro-F1 0.6476, coverage 100%. The 6×6 command/unknown confusion matrix is saved in `results/custom/commands_v3/evaluation.json`. Only two test clips per command support a descriptive result. Protected command execution returns HTTP 503 because the small speaker corpus lacks unknown-speaker validation and the permission workflow remains intentionally disabled.

The custom page presents a sequential identity → live Face capture/verification → Voice recording/upload/verification → uploaded Fingerprint verification flow. Its fingerprint upload route (`/api/custom-fingerprint/verify-upload`) resolves the selected Nour/Sama alias from the mapping CSV and calls the existing saved SVM/template without changing its threshold or inference behavior. The separate `/api/custom/combined/verify` API still accepts a mapped test-split Fingerprint probe and applies the server-side all-of-three rule. Both use an assumed demo association, not proof of shared real identities or scientific fusion. Physical browser/camera/microphone testing remains pending. Isolated model/results paths are `saved_models/custom/voice_v1/`, `results/custom/voice_v1/`, `saved_models/custom/commands_v3/`, and `results/custom/commands_v3/`; earlier Face and command pilots remain preserved.

See [PROJECT_COMPLETION_AUDIT.md](PROJECT_COMPLETION_AUDIT.md) for the requirement status and preservation baseline. The original VidTIMIT, Fingerprint, and custom fingerprint artifacts remain separate from the custom Face models.

## Environment and dependencies

Use Python 3.12 and a project virtual environment. The requirements files declare packages; they are not a tested environment lockfile, and installing them was not part of the audit. Saved scikit-learn models can be sensitive to library-version changes, so use the project’s intended environment when loading them.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Before running scripts that import Pandas, check that Pandas loads in the target environment:

```powershell
python -c "import pandas as pd; print(pd.__version__)"
```

A Windows Application Control error has blocked the Pandas native `pandas_parser` extension in one observed environment. If that occurs, Pandas metadata may still appear installed while imports fail. This audit did not establish a package-upgrade fix; do not treat upgrading Pandas or pip as a remedy.

`requirements.txt` contains the core dependencies (NumPy, Pandas, OpenCV, scikit-image, librosa, scikit-learn, SciPy, SoundFile, Matplotlib, and joblib) plus notebook and auxiliary inspection/visualization tools (Jupyter, ipykernel, Pillow, and Seaborn). It does not include Deep Learning or UI/backend packages.

## Reproduction commands — may overwrite existing outputs

> **Warning:** These are reproduction/training commands, not read-only checks. Preprocessing, feature extraction, split creation, model training, threshold selection, and evaluation scripts write to existing project paths. Running them can replace processed data, splits, saved models, threshold records, CSV results, or plots. Do not run them in a workspace where validated artifacts must be preserved unless you have made and verified separate backups.

Run from the repository root, using the intended Python environment. The following commands are retained for reproducing the pipeline when overwriting outputs is acceptable:

```powershell
python src/preprocessing/face_preprocessing.py
python src/features/face_features.py
python src/features/voice_features.py
python src/preprocessing/session_split.py
python src/preprocessing/prepare_datasets.py
python src/verification/face_verification.py
python src/verification/voice_verification.py
python src/verification/face_security_thresholds.py
python src/verification/voice_security_thresholds.py
python src/verification/early_fusion_verification.py
python src/verification/hierarchical_verification.py
```

The Face/Voice/Fusion scripts train or select thresholds and write model/results artifacts. The Hierarchical script writes its `_sequential` files and sequential policy; it preserves the legacy and `_audit` files but overwrites the sequential names on rerun. `src/verification/fingerprint_verification.py` also trains and overwrites the baseline Fingerprint model/result paths; it is intentionally omitted from this reproduction command list because the Fingerprint outputs are already finalized.

## Comprehensive audit record

**Overall status: PARTIALLY COMPLETE; do not claim all bonuses are complete.** The mandatory VidTIMIT systems and independent Fingerprint pipeline remain intact. Custom Face/Voice experiments, a two-speaker command classifier, a fail-closed all-of-three demo policy, and webcam/microphone capture controls are implemented with isolated artifacts. The custom datasets are too small for robust unknown-speaker or generalization claims; protected command execution is disabled; hardware/browser workflows were not manually tested.

### Dataset and protocol audit

- VidTIMIT contains 43 speakers and 430 recordings. Session 1 has 258 recordings for training/enrollment; Session 2 has 86 for validation/threshold selection; Session 3 has 86 for final evaluation.
- Session 3 verification expands to 86 genuine and 3,612 impostor trials (3,698 total). Genuine is label 1; impostor is label 0. FAR = FP/(FP+TN), FRR = FN/(FN+TP).
- The split construction is in `src/preprocessing/session_split.py` and `src/preprocessing/prepare_datasets.py`. The observed split files use the stated session protocol.
- Face and Voice use the same VidTIMIT identities, held-out sessions, claimed identity trial protocol, and 3,698 Session 3 trials. Their chosen thresholds and operating points are model-specific. Early Fusion uses the same trials with concatenated features. Hierarchical uses the same trials with a cascade policy.
- Trial imbalance means accuracy alone is insufficient; FAR estimates are also sensitive to a small number of false accepts.

### Authoritative Session 3 comparison

Values below are from the saved Face, Voice, Fusion, and sequential Hierarchical evaluation files. The Face final CSV omits accuracy, precision, recall, and F1; those four values are derived from its stored confusion counts. EER is unavailable in the VidTIMIT result artifacts. Hierarchical produces binary cascade decisions, so ROC-AUC is not reported.

| System | Model | Threshold | Accuracy | Precision | Recall | F1 | FAR | FRR | ROC-AUC | TN / FP / FN / TP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Face | SVM | 0.290093 | 98.59% | 68.09% | 74.42% | 71.11% | 0.83% | 25.58% | 0.9657 | 3582 / 30 / 22 / 64 |
| Face | Random Forest | 0.166667 | 98.57% | 75.38% | 56.98% | 64.90% | 0.44% | 43.02% | 0.9182 | 3596 / 16 / 37 / 49 |
| Voice | SVM | 0.816066 | 97.21% | 29.27% | 13.95% | 18.90% | 0.80% | 86.05% | 0.8010 | 3583 / 29 / 74 / 12 |
| Voice | Random Forest | 0.476667 | 96.92% | 22.00% | 12.79% | 16.18% | 1.08% | 87.21% | 0.8140 | 3573 / 39 / 75 / 11 |
| Early Fusion | SVM | 0.217984 | 98.40% | 62.86% | 76.74% | 69.11% | 1.08% | 23.26% | 0.9684 | 3573 / 39 / 20 / 66 |
| Early Fusion | Random Forest | 0.160000 | 98.62% | 73.97% | 62.79% | 67.92% | 0.53% | 37.21% | 0.9267 | 3593 / 19 / 32 / 54 |
| Hierarchical | Face SVM → Voice SVM | 0.171650 / 0.290093 / 0.202839 | 98.57% | 66.67% | 76.74% | 71.35% | 0.91% | 23.26% | Not reported | 3579 / 33 / 20 / 66 |

All listed confusion rows sum to 3,698. Face/Voice/Fusion thresholds are the saved 1% target-FAR operating points; test FAR may differ from target. The Hierarchical threshold record is `results/hierarchical/selected_thresholds_audit.json`: Session 2 FAR 0.8029%, Face low 0.17164957, Face high 0.290093, Voice 0.20283867.

### Hierarchical policy and artifact authority

`src/verification/hierarchical_verification.py` loads `saved_models/hierarchical/hierarchical_policy_audit.joblib`, uses Face SVM first, and runs Voice only for uncertain trials. Session 3 stage counts are 3,574 direct Face rejects, 94 direct Face accepts, and 30 Voice escalations. Each trial receives one final binary decision; runtime decisions are generated before evaluation labels are attached.

The authoritative evaluation outputs use the `_sequential` suffix, including `results/hierarchical/hierarchical_trial_decisions_sequential.csv` and `results/hierarchical/hierarchical_comparison_sequential.csv`. `results/hierarchical/sequential_vs_audit_comparison.json` records identical Session 3 decisions and labels versus the validated audit, zero differences in reported metrics, and maximum escalated Voice score delta 2.06e-8. The `_audit` threshold record and policy are authoritative inputs. Canonical legacy hierarchy outputs are historical and are retained; do not mix them with the sequential results.

The validated Face/Voice security models are the `*_security.joblib` files under `saved_models/face/` and `saved_models/voice/`; Early Fusion models are in `saved_models/fusion/`. These are distinct from older baseline, session-based, or tuned model artifacts. No model was retrained for this audit.

### Assignment requirements checklist

| Requirement | Status | Evidence and limitation |
|---|---|---|
| Face preprocessing | PASS | `src/preprocessing/face_preprocessing.py`; processed Face data and metadata inspected. |
| Face HOG | PASS | `src/features/face_features.py`; HOG contributes to 3,548 dimensions. |
| Face LBP | PASS | Same feature source; LBP accompanies HOG. |
| Voice MFCC | PASS | `src/features/voice_features.py`; 13 coefficients, mean and standard deviation. |
| Voice ZCR | PASS | Same source; mean and standard deviation. |
| Voice RMS | PASS | Same source; mean and standard deviation. |
| Early Feature Fusion | PASS | `src/verification/early_fusion_verification.py`; concatenates Face+Voice features before training. |
| Face SVM | PASS | Security model and Session 3 metrics in `results/face/`. |
| Face Random Forest | PASS | Security model and Session 3 metrics in `results/face/`. |
| Voice SVM | PASS | Security model and Session 3 metrics in `results/voice/`. |
| Voice Random Forest | PASS | Security model and Session 3 metrics in `results/voice/`. |
| Fusion SVM | PASS | Security model and Session 3 metrics in `results/fusion/`. |
| Fusion Random Forest | PASS | Security model and Session 3 metrics in `results/fusion/`. |
| Face-only verification | PASS | Pairwise genuine/impostor trials; 3,698 Session 3 decisions. |
| Voice-only verification | PASS | Same held-out trial protocol; Voice genuine recall is low at the security threshold. |
| Early Fusion verification | PASS | Same trial protocol; feature-level concatenation. |
| Hierarchical verification | PASS | Sequential Face-to-Voice policy and saved sequential result files. |
| Sequential Face → Voice decisions | PASS | 30 of 3,698 Session 3 trials escalated; no Voice scores on direct decisions. |
| Validation-only threshold selection | PASS | Session 2 policy record; Session 3 used only for final evaluation. |
| Session 3 evaluation | PASS | Confusion matrices sum to 3,698 for all four compared systems. |
| Metrics calculations | PASS | Confusion counts align with accuracy, precision, recall, F1, FAR, and FRR. |
| Confusion matrices | PASS | Saved result artifacts; Hierarchical TN/FP/FN/TP = 3579/33/20/66. |
| ROC curves | PASS | Face, Voice, Fusion, and Fingerprint continuous-score outputs have ROC plots. Hierarchical ROC-AUC is not applicable to its binary cascade output. |
| Comparison across required systems | PASS | Same VidTIMIT trial protocol; thresholds/operating points are model-specific. |
| Fingerprint dataset | PASS | 320 images; 40 database-qualified identities; DB4_B synthetic and separate. |
| Fingerprint preprocessing | PASS | CLAHE, median filter, aspect-preserving resize/padding to 256×256; 320 processed successfully. |
| Fingerprint HOG/LBP | PASS | HOG 8,100 + LBP 10; feature ordering and inference parity checked. |
| Fingerprint SVM/RF | PASS | Saved estimators, thresholds, target FAR, and feature ordering. |
| Fingerprint evaluation | PASS | Validation threshold frozen before test; high FRR is documented. |
| Fingerprint new-image inference | PASS | `src/verification/fingerprint_inference.py`; smoke-tested on a raw dataset image. A separately collected external sample was unavailable. |
| No Deep Learning in verification | PASS | Mandatory pipelines use handcrafted features and SVM/RF. Neural packages are not required dependencies. |
| Speech-command bonus | PARTIAL | Two-speaker command classifier; test has two clips per command; protected execution is disabled. |
| Real-time bonus | PARTIAL | Custom webcam/microphone capture calls real endpoints; no physical browser/device test. |
| Custom Team Dataset | PARTIAL | Face and Voice models trained on grouped splits; only two identities and very small test sets. |

Focused validation executed 34 tests successfully with `.venv/Scripts/python.exe -m unittest discover -s tests -v`. These include original saved-model input dimensions and score inference, hierarchical decision-policy checks, fingerprint feature parity and alias tests, custom Face, Voice and combined inference/API checks, manifest integrity checks, and fail-closed command endpoint tests. Local Uvicorn returned successful Voice/command/combined results and camera verification; the camera guard rejected a no-face frame, and protected command execution returned HTTP 503. The static server returned the original, custom, and fingerprint pages with HTTP 200. `node --check` passed for both custom demo scripts. This does not establish real-browser/camera/microphone hardware behavior. No original-model training, threshold-selection, or evaluation script was rerun.

## Fingerprint technical report

### Dataset, splits, and features

FVC2002 Set B contains DB1_B, DB2_B, DB3_B, and DB4_B, each with 10 identities and eight impressions: 320 images total. DB1_B–DB3_B are captured data (30 database-qualified identities) and DB4_B is synthetic (10 identities); they are not merged. The preprocessing report is `data/metadata/fingerprint_preprocessing_report.csv`; feature records are `data/processed/fingerprint/fingerprint_features.csv`.

The split builder `src/preprocessing/prepare_fingerprint_dataset.py` assigns impressions 1–4 to enrollment, 5–6 to validation, and 7–8 to test. Real split counts are 120/60/60 images; synthetic split counts are 40/20/20. All identities occur in all splits, and image-level overlap checks passed. All source images were readable, all processed images were 256×256, and the 320×8,110 feature matrix had no duplicates, missing values, or non-finite entries.

Preprocessing converts to grayscale, applies CLAHE (clip limit 2.0, tile grid 8×8), median filtering (size 3), then aspect-preserving resize and white padding to 256×256. HOG uses 9 orientations, 16×16 pixels per cell, 2×2 cells per block, L2-Hys normalization, and square-root transform, giving 8,100 values. Uniform LBP uses P=8, R=1 and a normalized 10-bin histogram. Feature order is HOG indices 0000–8099 followed by LBP indices 00–09.

### Fingerprint trials and baseline metrics

Four enrollment images form each identity template. Training genuine trials use a leave-one-out template made from the other three samples; impostor trials compare against other identity templates. This yields 3,600 training trials (120 genuine, 3,480 impostor). Validation and test probes are each scored against the full four-image enrollment templates, yielding 1,800 trials each (60 genuine, 1,740 impostor). Thresholds are selected from validation at target FAR 1%, then frozen for test.

| Model | Threshold | Validation FAR | Validation FRR | Test accuracy | Precision | Recall | F1 | FAR | FRR | ROC-AUC | TN / FP / FN / TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SVM | 0.4540606025 | 0.9770% | 86.6667% | 96.3889% | 44.4444% | 33.3333% | 38.0952% | 1.4368% | 66.6667% | 0.905125 | 1715 / 25 / 40 / 20 |
| Random Forest | 0.2366666667 | 0.9195% | 86.6667% | 95.5000% | 32.2034% | 31.6667% | 31.9328% | 2.2989% | 68.3333% | 0.924866 | 1700 / 40 / 41 / 19 |

The discrete test-score EER summaries are 0.1664 (SVM) and 0.1575 (Random Forest); they are descriptive only and were not used for threshold selection. Confusion matrices and ROC plots are under `results/fingerprint/SVM/` and `results/fingerprint/RandomForest/`.

The separate controlled comparison at `results/fingerprint/controlled_audit_v1/` tested block-balanced HOG/LBP distances with linear SVM and Logistic Regression. Validation FAR was 0.9770%; validation FRR was 86.6667% and 85.0000%, respectively. Test FAR increased to 4.0230% and 4.1379%, so neither alternative replaced the saved baseline. The baseline audit found false accepts among same-database impostor claims and none among cross-database impostors, suggesting the cross-database negative trials are easier. Small identity/impression counts and high FRR limit conclusions.

### Fingerprint inference usage

The original model artifacts contain estimator, threshold, feature names/order, and target FAR, but not templates. `saved_models/fingerprint/fingerprint_enrollment_templates_v1.npz` contains 30 mean templates derived from the real enrollment split. Inference in `src/verification/fingerprint_inference.py` loads the saved SVM and template artifact, applies the same image preprocessing and HOG/LBP parameters, scores the absolute feature difference, and returns claim, probability score, frozen threshold, and ACCEPT/REJECT. It does not train or calibrate. The module refuses to overwrite an existing template artifact. A duplicate `.joblib` export is retained but is not used.

```powershell
.venv\Scripts\python.exe src/verification/fingerprint_inference.py data/raw/fingerprint/DB1_B/101_7.tif DB1_B_101
```

The smoke-test image scored 0.00186968 against threshold 0.45406060 and was rejected, matching its saved test trial. This validates the code path on an existing dataset image, not generalization to a new user or sensor. Fingerprint identities are unrelated to VidTIMIT participants, so genuine three-modality fusion cannot be claimed.

### Fingerprint limitations

There are 30 real identities, four enrollment impressions, and only two impressions for each of validation and test. HOG is spatially sensitive; rotation, pressure, placement, ridge quality, and sensor shift may affect it, but these factors were not individually isolated. The alternatives increased genuine acceptance slightly while substantially worsening test FAR. Keep the original results and threshold; do not tune on test data. DB4_B is synthetic and is never represented as captured data.

## Environment notes and references

`requirements.txt` lists core dependencies plus notebook and auxiliary tools. It is not a tested lockfile, and no clean installation was tested. The active Windows environment once reported an Application Control denial while loading Pandas' native `pandas_parser` extension. Package metadata can exist even when this DLL import fails; upgrading Pandas or pip is not an established fix. Check Pandas importability in the target environment before running dependent scripts.

VidTIMIT reference: C. Sanderson and B. C. Lovell, “Multi-Region Probabilistic Histograms for Robust and Scalable Identity Inference,” LNCS 5558, 199–208 (2009).
