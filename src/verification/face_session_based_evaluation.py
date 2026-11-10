
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from face_verification import (
    RANDOM_STATE,
    load_data,
    build_training_trials,
    build_test_trials,
    create_models,
)


# ==========================================
# 1. PATHS AND CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "saved_models" / "face"
RESULTS_DIR = PROJECT_ROOT / "results" / "face"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. THRESHOLD SELECTION
# ==========================================

def find_eer_threshold(y_true, scores):

    fpr, tpr, thresholds = roc_curve(y_true, scores)

    fnr = 1 - tpr

    index = np.nanargmin(np.abs(fpr - fnr))

    threshold = thresholds[index]
    eer = (fpr[index] + fnr[index]) / 2

    return threshold, eer


# ==========================================
# 3. GET GENUINE SCORES
# ==========================================

def get_scores(model, X):

    positive_index = list(model.classes_).index(1)

    return model.predict_proba(X)[:, positive_index]


# ==========================================
# 4. EVALUATION METRICS
# ==========================================

def evaluate_model(y_true, scores, threshold):

    predictions = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    far = fp / (fp + tn) if (fp + tn) else 0
    frr = fn / (fn + tp) if (fn + tp) else 0

    return {
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_true, predictions, zero_division=0
        ),
        "FAR": far,
        "FRR": frr,
        "ROC-AUC": roc_auc_score(y_true, scores),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


# ==========================================
# 5. PREPARE SESSION DATA
# ==========================================

def prepare_sessions(train_df, test_df):

    if "session" not in train_df.columns:
        raise ValueError(
            "The 'session' column was not found in face_train.csv"
        )

    if "session" not in test_df.columns:
        raise ValueError(
            "The 'session' column was not found in face_test.csv"
        )

    # Normalize session names
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

    if len(session_2) == 0 or len(session_3) == 0:
        raise ValueError(
            "Session 2 or Session 3 is empty. "
            f"Available values: {test_df['session'].unique()}"
        )

    return session_2, session_3


# ==========================================
# 6. MAIN PIPELINE
# ==========================================

def main():

    print("=" * 65)
    print("FACE VERIFICATION - SESSION-BASED EVALUATION")
    print("=" * 65)

    (
        train_df,
        test_df,
        X_train,
        X_test,
        y_train,
        y_test,
    ) = load_data()

    feature_cols = [
        col for col in train_df.columns
        if col.startswith("face_feature_")
    ]

    # --------------------------------------
    # A. Separate Session 2 and Session 3
    # --------------------------------------

    session_2, session_3 = prepare_sessions(
        train_df,
        test_df,
    )

    X_val = session_2[feature_cols].to_numpy(
        dtype=np.float32
    )
    y_val = session_2["speaker"].astype(str).to_numpy()

    X_final_test = session_3[feature_cols].to_numpy(
        dtype=np.float32
    )
    y_final_test = session_3["speaker"].astype(str).to_numpy()

    print("\n========== DATA SPLIT ==========")
    print("Session 1 - Training:", len(X_train))
    print("Session 2 - Validation:", len(X_val))
    print("Session 3 - Final Testing:", len(X_final_test))

    print("\nSpeakers in Session 2:", len(np.unique(y_val)))
    print("Speakers in Session 3:", len(np.unique(y_final_test)))

    # --------------------------------------
    # B. Build training trials from Session 1
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
    # C. Build Session 2 validation trials
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
    # D. Build Session 3 final test trials
    # --------------------------------------

    print("\nBuilding Session 3 final test trials...")

    X_test_pairs, y_test_pairs = build_test_trials(
        X_train,
        y_train,
        X_final_test,
        y_final_test,
    )

    print("Final test trials:", len(X_test_pairs))
    print("Genuine:", np.sum(y_test_pairs == 1))
    print("Impostor:", np.sum(y_test_pairs == 0))

    # --------------------------------------
    # E. Train models and select thresholds
    # --------------------------------------

    models = create_models()

    selected_thresholds = {}
    validation_results = []
    final_results = []

    for model_name, model in models.items():

        print("\n" + "=" * 55)
        print(f"MODEL: {model_name}")
        print("=" * 55)

        print("\nTraining using Session 1...")
        model.fit(X_train_pairs, y_train_pairs)

        # Select threshold using Session 2
        val_scores = get_scores(model, X_val_pairs)

        threshold, eer = find_eer_threshold(
            y_val_pairs,
            val_scores,
        )

        selected_thresholds[model_name] = threshold

        val_metrics = evaluate_model(
            y_val_pairs,
            val_scores,
            threshold,
        )

        val_metrics["Model"] = model_name
        val_metrics["Validation_EER"] = eer

        validation_results.append(val_metrics)

        print("\nSESSION 2 VALIDATION")
        print(f"Selected Threshold: {threshold:.6f}")
        print(f"Validation EER: {eer:.4f}")
        print(f"Validation FAR: {val_metrics['FAR']:.4f}")
        print(f"Validation FRR: {val_metrics['FRR']:.4f}")

        # Evaluate only on Session 3
        test_scores = get_scores(model, X_test_pairs)

        test_metrics = evaluate_model(
            y_test_pairs,
            test_scores,
            threshold,
        )

        test_metrics["Model"] = model_name
        test_metrics["Validation_EER"] = eer

        final_results.append(test_metrics)

        print("\nSESSION 3 FINAL TEST")
        for metric, value in test_metrics.items():

            if metric == "Model":
                continue

            if isinstance(value, (float, np.floating)):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")

        # Save model and its validation-selected threshold
        filename = (
            "svm_face_session_based.joblib"
            if model_name == "SVM"
            else "random_forest_face_session_based.joblib"
        )

        joblib.dump(
            {
                "model": model,
                "threshold": threshold,
                "feature_columns": feature_cols,
                "enrollment_speakers": np.unique(y_train),
            },
            MODEL_DIR / filename,
        )

        print("\nSaved model:", MODEL_DIR / filename)

    # --------------------------------------
    # F. Save validation results
    # --------------------------------------

    validation_df = pd.DataFrame(validation_results)

    validation_path = (
        RESULTS_DIR / "face_session_2_validation.csv"
    )

    validation_df.to_csv(
        validation_path,
        index=False,
    )

    # --------------------------------------
    # G. Save final test results
    # --------------------------------------

    final_df = pd.DataFrame(final_results)

    final_path = (
        RESULTS_DIR / "face_session_3_final_results.csv"
    )

    final_df.to_csv(
        final_path,
        index=False,
    )

    # --------------------------------------
    # H. Final comparison
    # --------------------------------------

    print("\n" + "=" * 65)
    print("FINAL SESSION 3 MODEL COMPARISON")
    print("=" * 65)

    print(
        final_df[
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

    print("\nValidation results:", validation_path)
    print("Final test results:", final_path)

    print("\nSession-based evaluation completed successfully.")


if __name__ == "__main__":
    main()
