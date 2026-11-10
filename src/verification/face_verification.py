
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ==========================================
# 1. CONFIGURATION AND PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "splits"

TRAIN_FILE = SPLITS_DIR / "face_train.csv"
TEST_FILE = SPLITS_DIR / "face_test.csv"

MODEL_DIR = PROJECT_ROOT / "saved_models" / "face"
RESULTS_DIR = PROJECT_ROOT / "results" / "face"

RANDOM_STATE = 42
N_IMPOSTOR_TRAIN = 5
PCA_COMPONENTS = 100
THRESHOLD = 0.5


# ==========================================
# 2. LOAD DATA
# ==========================================

def load_data():

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    feature_cols = [
        col for col in train_df.columns
        if col.startswith("face_feature_")
    ]

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)

    y_train = train_df["speaker"].astype(str).to_numpy()
    y_test = test_df["speaker"].astype(str).to_numpy()

    print("Training recordings:", len(train_df))
    print("Testing recordings:", len(test_df))
    print("Speakers:", len(np.unique(y_train)))
    print("Face features:", len(feature_cols))

    return train_df, test_df, X_train, X_test, y_train, y_test


# ==========================================
# 3. BUILD TRAINING TRIALS
# ==========================================

def build_training_trials(X_train, y_train):

    rng = np.random.default_rng(RANDOM_STATE)

    speakers = np.unique(y_train)

    speaker_data = {
        speaker: X_train[y_train == speaker]
        for speaker in speakers
    }

    pair_features = []
    pair_labels = []

    for speaker in speakers:

        recordings = speaker_data[speaker]

        for index, probe in enumerate(recordings):

            # Leave-one-out enrollment reference:
            # The probe must not be part of its own reference.
            reference = np.mean(
                np.delete(recordings, index, axis=0),
                axis=0,
            )

            # Genuine trial
            pair_features.append(np.abs(probe - reference))
            pair_labels.append(1)

            # Random impostor claims
            other_speakers = speakers[speakers != speaker]

            selected_impostors = rng.choice(
                other_speakers,
                size=min(N_IMPOSTOR_TRAIN, len(other_speakers)),
                replace=False,
            )

            for impostor in selected_impostors:

                impostor_reference = np.mean(
                    speaker_data[impostor],
                    axis=0,
                )

                pair_features.append(
                    np.abs(probe - impostor_reference)
                )
                pair_labels.append(0)

    X_pairs = np.asarray(pair_features, dtype=np.float32)
    y_pairs = np.asarray(pair_labels, dtype=np.int32)

    return X_pairs, y_pairs


# ==========================================
# 4. BUILD TESTING TRIALS
# ==========================================

def build_test_trials(X_train, y_train, X_test, y_test):

    speakers = np.unique(y_train)

    # Enrollment references use Session 1 only
    enrollment = {
        speaker: np.mean(
            X_train[y_train == speaker],
            axis=0,
        )
        for speaker in speakers
    }

    pair_features = []
    pair_labels = []

    for probe, true_speaker in zip(X_test, y_test):

        # Test each probe against all 43 claimed identities
        for claimed_speaker in speakers:

            reference = enrollment[claimed_speaker]

            pair_features.append(
                np.abs(probe - reference)
            )

            pair_labels.append(
                int(true_speaker == claimed_speaker)
            )

    X_pairs = np.asarray(pair_features, dtype=np.float32)
    y_pairs = np.asarray(pair_labels, dtype=np.int32)

    return X_pairs, y_pairs


# ==========================================
# 5. DEFINE MODELS
# ==========================================

def create_models():

    svm_model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "pca",
            PCA(
                n_components=PCA_COMPONENTS,
                svd_solver="randomized",
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "classifier",
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return {
        "SVM": svm_model,
        "Random Forest": rf_model,
    }


# ==========================================
# 6. EVALUATE VERIFICATION MODEL
# ==========================================

def evaluate_model(model, X_test, y_test, model_name):

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    far = fp / (fp + tn) if (fp + tn) else 0.0
    frr = fn / (fn + tp) if (fn + tp) else 0.0

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "FAR": far,
        "FRR": frr,
        "ROC-AUC": roc_auc_score(y_test, probabilities),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }

    print(f"\n========== {model_name.upper()} RESULTS ==========")

    for metric, value in metrics.items():
        if metric != "Model":
            print(f"{metric}: {value:.4f}" if isinstance(value, float)
                  else f"{metric}: {value}")

    return metrics


# ==========================================
# 7. MAIN PIPELINE
# ==========================================

def main():

    print("=" * 60)
    print("FACE BIOMETRIC VERIFICATION")
    print("=" * 60)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    (
        train_df,
        test_df,
        X_train,
        X_test,
        y_train,
        y_test,
    ) = load_data()

    print("\nBuilding training trials...")

    X_train_pairs, y_train_pairs = build_training_trials(
        X_train,
        y_train,
    )

    print("Training trial shape:", X_train_pairs.shape)
    print("Genuine training trials:", np.sum(y_train_pairs == 1))
    print("Impostor training trials:", np.sum(y_train_pairs == 0))

    print("\nBuilding testing trials...")

    X_test_pairs, y_test_pairs = build_test_trials(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    print("Testing trial shape:", X_test_pairs.shape)
    print("Genuine testing trials:", np.sum(y_test_pairs == 1))
    print("Impostor testing trials:", np.sum(y_test_pairs == 0))

    models = create_models()

    all_results = []

    for model_name, model in models.items():

        print(f"\nTraining {model_name}...")

        model.fit(X_train_pairs, y_train_pairs)

        metrics = evaluate_model(
            model,
            X_test_pairs,
            y_test_pairs,
            model_name,
        )

        all_results.append(metrics)

        model_filename = (
            "svm_face.joblib"
            if model_name == "SVM"
            else "random_forest_face.joblib"
        )

        joblib.dump(
            model,
            MODEL_DIR / model_filename,
        )

        print("Model saved:", MODEL_DIR / model_filename)

    results_df = pd.DataFrame(all_results)

    results_path = RESULTS_DIR / "face_verification_results.csv"

    results_df.to_csv(results_path, index=False)

    print("\n========== MODEL COMPARISON ==========")
    print(
        results_df[
            ["Model", "Accuracy", "Precision", "Recall",
             "F1 Score", "FAR", "FRR", "ROC-AUC"]
        ].to_string(index=False)
    )

    print("\nResults saved:", results_path)
    print("\nFace Verification completed.")


if __name__ == "__main__":
    main()
