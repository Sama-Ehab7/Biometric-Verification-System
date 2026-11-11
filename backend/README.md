# Backend: live face + voice verification

A small FastAPI server that verifies a claimed identity with the project's
**trained models**. It reuses the project's own code, so a live photo or
recording goes through exactly the same steps as the training data.

```
backend/
├── app.py         FastAPI app and routes (/api/verify, /api/enroll, ...)
├── config.py      file paths and settings
├── models.py      loads the trained SVMs, Session-1 enrollment references and the policy
├── face.py        image -> face crop -> HOG + LBP sequence features
├── voice.py       audio -> 32 kHz WAV -> MFCC + ZCR + RMS features
├── policy.py      the hierarchical decision rules (tested in tests/test_backend_policy.py)
├── enrollment.py  people enrolled through the API
└── server.py      old entry point, kept for compatibility
```

| Step | Face | Voice |
|---|---|---|
| Input | photo or camera frames | WAV (the UI converts recordings to 32 kHz WAV) |
| Preprocessing | `face_preprocessing.detect_face` + 15% padded crop, 128×128 gray | resampled to 32 kHz mono |
| Features | `face_features` HOG + LBP, mean/std over frames | `voice_features.extract_audio_features` (MFCC + ZCR + RMS) |
| Comparison | \|probe − enrolled reference\| → face SVM | \|probe − enrolled reference\| → voice SVM |

**Decision**: the validated hierarchical policy (Session-2 thresholds):

- face score ≥ **high** → accepted by face
- face score ≤ **low** → rejected by face
- otherwise → **voice decides** (voice score ≥ voice threshold → accepted)

## 1. Files it needs (git-ignored, from the machine that ran the pipeline)

```
saved_models/face/svm_face_security.joblib                   required
data/processed/splits/face_train.csv                         required (Session-1 enrollment)
saved_models/voice/svm_voice_security.joblib                 optional: enables voice
data/processed/splits/voice_train.csv                        optional: enables voice
saved_models/hierarchical/hierarchical_policy_audit.joblib   optional: otherwise
                                                             results/hierarchical/selected_thresholds_audit.json is used
```

- With only the face files, the server runs in **face-only** mode, using the threshold for 1% FAR from the face model.
- With the voice files too, it runs the full **hierarchical** face → voice policy.

## 2. Install

```
pip install fastapi uvicorn python-multipart numpy pandas scikit-learn scikit-image librosa soundfile joblib "opencv-python<5"
```

Tests for the decision rules (no models needed):

```
python -m unittest tests.test_backend_policy
```

> OpenCV **5.x removed `cv2.CascadeClassifier`**, which the face detector needs. Use OpenCV 4.x.

## 3. Run

From the project root:

```
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

(`backend.server:app` still works too.)

In a second terminal, serve the UI:

```
python -m http.server 5500
```

Then open <http://127.0.0.1:5500/ui/app.html>. The badge in the top right changes from
"Demo mode" to **"Face + voice models online"** (or "Face model online").

## 4. API

| Method | Path | What it does |
|---|---|---|
| GET | `/api/health` | mode (`hierarchical` / `face_only`), loaded models, thresholds |
| GET | `/api/identities` | the 43 VidTIMIT speakers + enrolled people |
| POST | `/api/verify` | form: `claimed_id`, `face` (one or more images), `voice` (optional WAV) |
| POST | `/api/enroll` | form: `name`, `faces` (photos), `voices` (optional WAV) |
| DELETE | `/api/enroll/{name}` | remove an enrolled person |

`/api/verify` returns `accepted`, `stage` (`face_accept`, `face_reject`, `voice_accept`,
`voice_reject` or `voice_required`), `face_score`, `voice_score`, the thresholds, and the detected face crop.

## Notes

- **Enrolled people** are saved in `backend/enrolled/enrolled.json` (git-ignored, because it is biometric data).
- A **single photo** is augmented into 8 slightly varied frames, standing in for the video frames the model was trained on. The camera burst in the UI (10 frames) is closer to the training data.
- The models were trained on VidTIMIT studio video from 2001, so webcam photos in other lighting can score lower than the reported Session-3 results.
