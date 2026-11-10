
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
    roc_auc_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
MODELS_DIR = PROJECT_ROOT / "saved_models" / "fusion"
RESULTS_DIR = PROJECT_ROOT / "results" / "fusion"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_IMPOSTORS_PER_TRAIN_PROBE = 5

TARGET_FAR_VALUES = [0.01, 0.05, 0.10]


# ============================================================
# 2. LOAD MULTIMODAL DATA
# ============================================================

def load_data():

    train_df = pd.read_csv(
        SPLITS_DIR / "multimodal_train.csv"
    )

    test_df = pd.read_csv(
        SPLITS_DIR / "multimodal_test.csv"
    )

    metadata_cols = ["speaker", "sequence", "session"]

    # Face features: face_feature_1 ... face_feature_3548

    face_cols = [
        col for col in train_df.columns
        if col.startswith("face_feature_")
    ]

    # Voice features:
    # 26 MFCC features + 2 ZCR features + 2 RMS features

    voice_cols = [
        col for col in train_df.columns
        if (
            col.startswith("mfcc_")
            or col.startswith("zcr_")
            or col.startswith("rms_")
        )
    ]

    feature_cols = face_cols + voice_cols

    # Validate feature counts

    if len(face_cols) != 3548:
        raise ValueError(
            f"Expected 3548 Face features, found {len(face_cols)}"
        )

    if len(voice_cols) != 30:
        raise ValueError(
            f"Expected 30 Voice features, found {len(voice_cols)}"
        )

    if len(feature_cols) != 3578:
        raise ValueError(
            f"Expected 3578 total features, found {len(feature_cols)}"
        )

    # Validate metadata

    for col in metadata_cols:

        if col not in train_df.columns or col not in test_df.columns:
            raise ValueError(
                f"Missing metadata column: {col}"
            )

    # Validate that all feature columns exist in both datasets

    missing_test_cols = [
        col for col in feature_cols
        if col not in test_df.columns
    ]

    if missing_test_cols:
        raise ValueError(
            f"Missing feature columns in test dataset: {missing_test_cols[:10]}"
        )

    # Validate missing values

    if train_df[feature_cols].isna().any().any():
        raise ValueError(
            "Missing values found in training features."
        )

    if test_df[feature_cols].isna().any().any():
        raise ValueError(
            "Missing values found in testing features."
        )

    # Validate infinite values

    if not np.isfinite(
        train_df[feature_cols].to_numpy(dtype=np.float32)
    ).all():

        raise ValueError(
            "Non-finite values found in training features."
        )

    if not np.isfinite(
        test_df[feature_cols].to_numpy(dtype=np.float32)
    ).all():

        raise ValueError(
            "Non-finite values found in testing features."
        )

    print(f"Training recordings: {len(train_df)}")
    print(f"Testing recordings: {len(test_df)}")
    print(f"Face features: {len(face_cols)}")
    print(f"Voice features: {len(voice_cols)}")
    print(f"Total fused features: {len(feature_cols)}")

    return train_df, test_df, feature_cols


# ============================================================
# 3. BUILD ENROLLMENT CENTROIDS
# ============================================================

def build_enrollment_centroids(X, y):

    centroids = {}

    for speaker in np.unique(y):

        speaker_features = X[y == speaker]

        centroids[speaker] = np.mean(
            speaker_features,
            axis=0
        )

    return centroids


# ============================================================
# 4. BUILD TRAINING TRIALS
# ============================================================

def build_training_trials(X, y):

    rng = np.random.default_rng(RANDOM_STATE)

    speakers = np.unique(y)

    X_pairs = []
    y_pairs = []

    for index in range(len(X)):

        probe = X[index]
        true_speaker = y[index]

        # Genuine trial:
        # Exclude the current probe from its enrollment reference.

        genuine_reference = X[
            (y == true_speaker)
            &
            (np.arange(len(y)) != index)
        ]

        genuine_centroid = np.mean(
            genuine_reference,
            axis=0
        )

        X_pairs.append(
            np.abs(probe - genuine_centroid)
        )

        y_pairs.append(1)

        # Impostor trials:
        # Compare against five randomly selected different speakers.

        impostor_speakers = speakers[
            speakers != true_speaker
        ]

        selected_impostors = rng.choice(
            impostor_speakers,
            size=min(
                N_IMPOSTORS_PER_TRAIN_PROBE,
                len(impostor_speakers)
            ),
            replace=False
        )

        for impostor_speaker in selected_impostors:

            impostor_reference = X[
                y == impostor_speaker
            ]

            impostor_centroid = np.mean(
                impostor_reference,
                axis=0
            )

            X_pairs.append(
                np.abs(probe - impostor_centroid)
            )

            y_pairs.append(0)

    return (
        np.asarray(X_pairs, dtype=np.float32),
        np.asarray(y_pairs, dtype=np.int32)
    )


# ============================================================
# 5. BUILD VALIDATION AND TEST TRIALS
# ============================================================

def build_test_trials(
    enrollment_X,
    enrollment_y,
    probe_X,
    probe_y
):

    centroids = build_enrollment_centroids(
        enrollment_X,
        enrollment_y
    )

    enrolled_speakers = sorted(centroids.keys())

    X_pairs = []
    y_pairs = []

    for probe, true_speaker in zip(probe_X, probe_y):

        for enrolled_speaker in enrolled_speakers:

            reference = centroids[enrolled_speaker]

            # Absolute difference between probe and enrollment.

            X_pairs.append(
                np.abs(probe - reference)
            )

            label = int(
                str(true_speaker) == str(enrolled_speaker)
            )

            y_pairs.append(label)

    return (
        np.asarray(X_pairs, dtype=np.float32),
        np.asarray(y_pairs, dtype=np.int32)
    )


# ============================================================
# 6. SELECT THRESHOLDS USING VALIDATION SCORES
# ============================================================

def select_security_thresholds(y_true, scores):

    impostor_scores = scores[y_true == 0]

    if len(impostor_scores) == 0:

        raise ValueError(
            "No impostor validation trials found."
        )

    thresholds = {}

    for target_far in TARGET_FAR_VALUES:

        # Select the impostor-score quantile corresponding
        # to the requested target FAR.

        threshold = np.quantile(
            impostor_scores,
            1 - target_far,
            method="higher"
        )

        thresholds[target_far] = float(threshold)

    return thresholds


# ============================================================
# 7. CALCULATE VERIFICATION METRICS
# ============================================================

def calculate_metrics(y_true, scores, threshold):

    # Positive class = Genuine
    # Negative class = Impostor

    y_pred = (
        scores >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    far = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    frr = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0.0
    )

    return {
        "Threshold": float(threshold),

        "Accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "F1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "FAR": far,
        "FRR": frr,

        "ROC_AUC": roc_auc_score(
            y_true,
            scores
        ),

        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }


# ============================================================
# 8. CREATE MODELS
# ============================================================

def create_models():

    # SVM with StandardScaler and PCA

    svm = Pipeline([

        ("scaler", StandardScaler()),

        ("pca", PCA(
            n_components=100,
            svd_solver="randomized",
            random_state=RANDOM_STATE
        )),

        ("classifier", SVC(
            kernel="rbf",
            C=1,
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))

    ])

    # Random Forest

    random_forest = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    return {
        "SVM": svm,
        "Random Forest": random_forest
    }


# ============================================================
# 9. MAIN PIPELINE
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("EARLY FEATURE FUSION - VERIFICATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load prepared multimodal datasets
    # --------------------------------------------------------

    train_df, test_df, feature_cols = load_data()

    # --------------------------------------------------------
    # Session 1 = Training and Enrollment
    # --------------------------------------------------------

    session_1 = train_df[
        train_df["session"].astype(str).str.lower().str.strip()
        == "session_1"
    ].copy()

    X_train = session_1[
        feature_cols
    ].to_numpy(dtype=np.float32)

    y_train = (
        session_1["speaker"].astype(str).to_numpy()
    )

    # --------------------------------------------------------
    # Session 2 = Validation
    # --------------------------------------------------------

    session_2 = test_df[
        test_df["session"].astype(str).str.lower().str.strip()
        == "session_2"
    ].copy()

    X_validation = session_2[
        feature_cols
    ].to_numpy(dtype=np.float32)

    y_validation = (
        session_2["speaker"].astype(str).to_numpy()
    )

    # --------------------------------------------------------
    # Session 3 = Final Test
    # --------------------------------------------------------

    session_3 = test_df[
        test_df["session"].astype(str).str.lower().str.strip()
        == "session_3"
    ].copy()

    X_test = session_3[
        feature_cols
    ].to_numpy(dtype=np.float32)

    y_test = (
        session_3["speaker"].astype(str).to_numpy()
    )

    print("\nSession distribution:")

    print(
        f"Session 1 - Training: {len(session_1)}"
    )

    print(
        f"Session 2 - Validation: {len(session_2)}"
    )

    print(
        f"Session 3 - Final Test: {len(session_3)}"
    )

    # --------------------------------------------------------
    # Build Training Trials
    # --------------------------------------------------------

    print("\nBuilding training trials...")

    X_train_pairs, y_train_pairs = build_training_trials(
        X_train,
        y_train
    )

    print(
        f"Training trials: {len(y_train_pairs)}"
    )

    print(
        f"Genuine: {np.sum(y_train_pairs == 1)}"
    )

    print(
        f"Impostor: {np.sum(y_train_pairs == 0)}"
    )

    # --------------------------------------------------------
    # Build Validation Trials
    # --------------------------------------------------------

    print("\nBuilding validation trials...")

    X_validation_pairs, y_validation_pairs = build_test_trials(
        X_train,
        y_train,
        X_validation,
        y_validation
    )

    print(
        f"Validation trials: {len(y_validation_pairs)}"
    )

    print(
        f"Genuine: {np.sum(y_validation_pairs == 1)}"
    )

    print(
        f"Impostor: {np.sum(y_validation_pairs == 0)}"
    )

    # --------------------------------------------------------
    # Build Final Test Trials
    # --------------------------------------------------------

    print("\nBuilding final test trials...")

    X_test_pairs, y_test_pairs = build_test_trials(
        X_train,
        y_train,
        X_test,
        y_test
    )

    print(
        f"Final test trials: {len(y_test_pairs)}"
    )

    print(
        f"Genuine: {np.sum(y_test_pairs == 1)}"
    )

    print(
        f"Impostor: {np.sum(y_test_pairs == 0)}"
    )

    # --------------------------------------------------------
    # Train and Evaluate Models
    # --------------------------------------------------------

    models = create_models()

    all_results = []

    for model_name, model in models.items():

        print("\n" + "-" * 50)
        print(f"Training {model_name}...")
        print("-" * 50)

        # Train using Session 1 only.

        model.fit(
            X_train_pairs,
            y_train_pairs
        )

        # ----------------------------------------------------
        # Validation: Session 2
        # ----------------------------------------------------

        validation_scores = model.predict_proba(
            X_validation_pairs
        )[:, 1]

        thresholds = select_security_thresholds(
            y_validation_pairs,
            validation_scores
        )

        print("\nValidation thresholds:")

        for target_far, threshold in thresholds.items():

            validation_metrics = calculate_metrics(
                y_validation_pairs,
                validation_scores,
                threshold
            )

            print(
                f"Target FAR {target_far:.0%} | "
                f"Threshold: {threshold:.6f} | "
                f"Actual FAR: {validation_metrics['FAR']:.4f} | "
                f"FRR: {validation_metrics['FRR']:.4f}"
            )

        # ----------------------------------------------------
        # Final Testing: Session 3
        # ----------------------------------------------------

        test_scores = model.predict_proba(
            X_test_pairs
        )[:, 1]

        print("\nSession 3 final test results:")

        for target_far, threshold in thresholds.items():

            metrics = calculate_metrics(
                y_test_pairs,
                test_scores,
                threshold
            )

            metrics["Model"] = model_name
            metrics["Target_FAR"] = target_far

            all_results.append(metrics)

            print(
                f"\nTarget FAR: {target_far:.0%}"
            )

            for metric_name in [
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "FAR",
                "FRR",
                "ROC_AUC"
            ]:

                print(
                    f"{metric_name}: "
                    f"{metrics[metric_name]:.4f}"
                )

        # ----------------------------------------------------
        # Save Model and Thresholds
        # ----------------------------------------------------

        model_filename = (
            "svm_early_fusion_security.joblib"
            if model_name == "SVM"
            else "random_forest_early_fusion_security.joblib"
        )

        joblib.dump(
            {
                "model": model,
                "thresholds": thresholds,
                "feature_columns": feature_cols,
                "training_protocol": "Session 1",
                "validation_protocol": "Session 2",
                "testing_protocol": "Session 3"
            },
            MODELS_DIR / model_filename
        )

        print(
            f"\nSaved model: {model_filename}"
        )

    # --------------------------------------------------------
    # Save Final Evaluation CSV
    # --------------------------------------------------------

    results_df = pd.DataFrame(all_results)

    column_order = [
        "Model",
        "Target_FAR",
        "Threshold",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "FAR",
        "FRR",
        "ROC_AUC",
        "TN",
        "FP",
        "FN",
        "TP"
    ]

    results_df = results_df[column_order]

    results_path = (
        RESULTS_DIR / "early_fusion_final_evaluation.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    print("\n" + "=" * 60)
    print("EARLY FUSION FINAL RESULTS")
    print("=" * 60)

    print(
        results_df.to_string(index=False)
    )

    print("\n========================================")
    print("EARLY FEATURE FUSION COMPLETED")
    print("========================================")

    print(
        f"Models saved in: {MODELS_DIR}"
    )

    print(
        f"Results saved in: {RESULTS_DIR}"
    )


# ============================================================
# 10. RUN
# ============================================================

if __name__ == "__main__":
    main()
