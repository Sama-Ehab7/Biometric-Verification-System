
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
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ==========================================
# 1. CONFIGURATION AND PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "splits"

TRAIN_FILE = SPLITS_DIR / "voice_train.csv"
TEST_FILE = SPLITS_DIR / "voice_test.csv"

MODEL_DIR = PROJECT_ROOT / "saved_models" / "voice"
RESULTS_DIR = PROJECT_ROOT / "results" / "voice"

RANDOM_STATE = 42
N_IMPOSTOR_TRAIN = 5


# ==========================================
# 2. LOAD DATA
# ==========================================

def load_data():

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")

    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Testing file not found: {TEST_FILE}")

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    metadata_cols = {
        "speaker",
        "sequence",
        "session",
    }

    feature_cols = [
        col for col in train_df.columns
        if col not in metadata_cols
    ]

    if len(feature_cols) != 30:
        raise ValueError(
            f"Expected 30 voice features, found {len(feature_cols)}. "
            f"Check the columns in {TRAIN_FILE}"
        )

    missing_test_cols = [
        col for col in feature_cols
        if col not in test_df.columns
    ]

    if missing_test_cols:
        raise ValueError(
            f"Missing feature columns in test data: {missing_test_cols}"
        )

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)

    y_train = train_df["speaker"].astype(str).to_numpy()
    y_test = test_df["speaker"].astype(str).to_numpy()

    if not np.isfinite(X_train).all():
        raise ValueError("Training data contains NaN or infinite values.")

    if not np.isfinite(X_test).all():
        raise ValueError("Testing data contains NaN or infinite values.")

    print("Training recordings:", len(train_df))
    print("Testing recordings:", len(test_df))
    print("Speakers:", len(np.unique(y_train)))
    print("Voice features:", len(feature_cols))

    return (
        train_df,
        test_df,
        X_train,
        X_test,
        y_train,
        y_test,
        feature_cols,
    )


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

            # Leave-one-out genuine enrollment reference
            reference = np.mean(
                np.delete(recordings, index, axis=0),
                axis=0,
            )

            # Genuine trial
            pair_features.append(
                np.abs(probe - reference)
            )
            pair_labels.append(1)

            # Sample impostor identities
            other_speakers = speakers[speakers != speaker]

            selected_impostors = rng.choice(
                other_speakers,
                size=min(
                    N_IMPOSTOR_TRAIN,
                    len(other_speakers),
                ),
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

    X_pairs = np.asarray(
        pair_features,
        dtype=np.float32,
    )

    y_pairs = np.asarray(
        pair_labels,
        dtype=np.int32,
    )

    return X_pairs, y_pairs


# ==========================================
# 4. BUILD VALIDATION / TEST TRIALS
# ==========================================

def build_test_trials(
    X_enrollment,
    y_enrollment,
    X_probes,
    y_probes,
):

    speakers = np.unique(y_enrollment)

    # Build enrollment reference for each speaker
    enrollment = {
        speaker: np.mean(
            X_enrollment[y_enrollment == speaker],
            axis=0,
        )
        for speaker in speakers
    }

    pair_features = []
    pair_labels = []

    for probe, true_speaker in zip(X_probes, y_probes):

        # Compare each probe with all enrolled identities
        for claimed_speaker in speakers:

            reference = enrollment[claimed_speaker]

            pair_features.append(
                np.abs(probe - reference)
            )

            pair_labels.append(
                int(true_speaker == claimed_speaker)
            )

    X_pairs = np.asarray(
        pair_features,
        dtype=np.float32,
    )

    y_pairs = np.asarray(
        pair_labels,
        dtype=np.int32,
    )

    return X_pairs, y_pairs


# ==========================================
# 5. CREATE MODELS
# ==========================================

def create_models():

    svm_model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "pca",
            PCA(
                n_components=0.95,
                svd_solver="full",
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
# 6. GET GENUINE SCORES
# ==========================================

def get_scores(model, X):

    positive_index = list(model.classes_).index(1)

    return model.predict_proba(X)[:, positive_index]


# ==========================================
# 7. FIND EER THRESHOLD
# ==========================================

def find_eer_threshold(y_true, scores):

    fpr, tpr, thresholds = roc_curve(
        y_true,
        scores,
    )

    fnr = 1 - tpr

    index = np.nanargmin(
        np.abs(fpr - fnr)
    )

    threshold = thresholds[index]

    eer = (
        fpr[index] + fnr[index]
    ) / 2

    return threshold, eer


# ==========================================
# 8. EVALUATE MODEL
# ==========================================

def evaluate_model(
    model,
    X_test,
    y_test,
    threshold,
    model_name,
):

    scores = get_scores(model, X_test)

    predictions = (
        scores >= threshold
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
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "F1 Score": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "FAR": far,
        "FRR": frr,
        "ROC-AUC": roc_auc_score(y_test, scores),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }

    print(f"\n========== {model_name.upper()} ==========")

    for metric, value in metrics.items():

        if metric == "Model":
            continue

        if isinstance(value, (float, np.floating)):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    return metrics


# ==========================================
# 9. MAIN PIPELINE
# ==========================================

def main():

    print("=" * 65)
    print("VOICE BIOMETRIC VERIFICATION")
    print("=" * 65)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------
    # A. Load data
    # --------------------------------------

    (
        train_df,
        test_df,
        X_train,
        X_test,
        y_train,
        y_test,
        feature_cols,
    ) = load_data()

    # --------------------------------------
    # B. Separate Session 2 and Session 3
    # --------------------------------------

    test_df = test_df.copy()

    test_df["session"] = (
        test_df["session"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    session_2 = test_df[
        test_df["session"] == "session_2"
    ].copy()

    session_3 = test_df[
        test_df["session"] == "session_3"
    ].copy()

    if session_2.empty or session_3.empty:
        raise ValueError(
            "Session 2 or Session 3 is empty. "
            f"Available sessions: {test_df['session'].unique()}"
        )

    X_val = session_2[feature_cols].to_numpy(
        dtype=np.float32
    )

    y_val = session_2["speaker"].astype(str).to_numpy()

    X_final_test = session_3[feature_cols].to_numpy(
        dtype=np.float32
    )

    y_final_test = (
        session_3["speaker"].astype(str).to_numpy()
    )

    print("\n========== DATA SPLIT ==========")
    print("Session 1 - Training:", len(X_train))
    print("Session 2 - Validation:", len(X_val))
    print("Session 3 - Final Testing:", len(X_final_test))

    print("Session 2 speakers:", len(np.unique(y_val)))
    print("Session 3 speakers:", len(np.unique(y_final_test)))

    # --------------------------------------
    # C. Build Session 1 training trials
    # --------------------------------------

    print("\nBuilding Session 1 training trials...")

    X_train_pairs, y_train_pairs = build_training_trials(
        X_train,
        y_train,
    )

    print("Training trials:", len(X_train_pairs))
    print("Genuine:", np.sum(y_train_pairs == 1))
    print("Impostor:", np.sum(y_train_pairs == 0))

    # --------------------------------------
    # D. Build Session 2 validation trials
    # --------------------------------------

    print("\nBuilding Session 2 validation trials...")

    X_val_pairs, y_val_pairs = build_test_trials(
        X_train,
        y_train,
        X_val,
        y_val,
    )

    print("Validation trials:", len(X_val_pairs))
    print("Genuine:", np.sum(y_val_pairs == 1))
    print("Impostor:", np.sum(y_val_pairs == 0))

    # --------------------------------------
    # E. Build Session 3 final test trials
    # --------------------------------------

    print("\nBuilding Session 3 final test trials...")

    X_test_pairs, y_test_pairs = build_test_trials(
        X_train,
        y_train,
        X_final_test,
        y_final_test,
    )

    print("Testing trials:", len(X_test_pairs))
    print("Genuine:", np.sum(y_test_pairs == 1))
    print("Impostor:", np.sum(y_test_pairs == 0))

    # --------------------------------------
    # F. Train and evaluate models
    # --------------------------------------

    models = create_models()

    all_results = []

    for model_name, model in models.items():

        print("\n" + "=" * 60)
        print(f"TRAINING {model_name}")
        print("=" * 60)

        model.fit(
            X_train_pairs,
            y_train_pairs,
        )

        # Select threshold using Session 2
        val_scores = get_scores(
            model,
            X_val_pairs,
        )

        threshold, eer = find_eer_threshold(
            y_val_pairs,
            val_scores,
        )

        print("\nSESSION 2 VALIDATION")
        print(f"Selected Threshold: {threshold:.6f}")
        print(f"Validation EER: {eer:.4f}")

        # Evaluate on Session 3
        metrics = evaluate_model(
            model,
            X_test_pairs,
            y_test_pairs,
            threshold,
            model_name,
        )

        metrics["Validation_EER"] = eer

        all_results.append(metrics)

        # Save model and threshold
        filename = (
            "svm_voice_session_based.joblib"
            if model_name == "SVM"
            else "random_forest_voice_session_based.joblib"
        )

        model_path = MODEL_DIR / filename

        joblib.dump(
            {
                "model": model,
                "threshold": threshold,
                "feature_columns": feature_cols,
                "enrollment_speakers": np.unique(y_train),
            },
            model_path,
        )

        print("\nModel saved:", model_path)

    # --------------------------------------
    # G. Save results
    # --------------------------------------

    results_df = pd.DataFrame(all_results)

    results_path = (
        RESULTS_DIR / "voice_session_3_final_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    print("\n" + "=" * 65)
    print("VOICE VERIFICATION - FINAL COMPARISON")
    print("=" * 65)

    print(
        results_df[
            [
                "Model",
                "Threshold",
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "FAR",
                "FRR",
                "ROC-AUC",
            ]
        ].to_string(index=False)
    )

    print("\nResults saved:", results_path)
    print("\nVoice Verification completed successfully.")


if __name__ == "__main__":
    main()
