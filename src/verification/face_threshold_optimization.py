
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
# 1. PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "saved_models" / "face"
RESULTS_DIR = PROJECT_ROOT / "results" / "face"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. THRESHOLD SELECTION USING EER
# ==========================================

def find_eer_threshold(y_true, scores):

    fpr, tpr, thresholds = roc_curve(y_true, scores)

    fnr = 1 - tpr

    # Select threshold where FAR and FRR are closest
    index = np.nanargmin(np.abs(fpr - fnr))

    threshold = thresholds[index]
    eer = (fpr[index] + fnr[index]) / 2

    return threshold, eer


# ==========================================
# 3. EVALUATION
# ==========================================

def evaluate_at_threshold(y_true, scores, threshold):

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
# 4. GET POSITIVE CLASS SCORES
# ==========================================

def get_scores(model, X):

    positive_class_index = list(
        model.classes_
    ).index(1)

    return model.predict_proba(X)[:, positive_class_index]


# ==========================================
# 5. MAIN
# ==========================================

def main():

    print("=" * 60)
    print("FACE VERIFICATION - THRESHOLD OPTIMIZATION")
    print("=" * 60)

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
    # A. Create a validation split
    # --------------------------------------

    validation_df = train_df.groupby(
        "speaker",
        group_keys=False,
    ).sample(
        n=1,
        random_state=RANDOM_STATE,
    )

    development_df = train_df.drop(
        validation_df.index
    )

    X_dev = development_df[feature_cols].to_numpy(
        dtype=np.float32
    )
    y_dev = development_df["speaker"].astype(str).to_numpy()

    X_val = validation_df[feature_cols].to_numpy(
        dtype=np.float32
    )
    y_val = validation_df["speaker"].astype(str).to_numpy()

    print("\nValidation split:")
    print("Development recordings:", len(X_dev))
    print("Validation recordings:", len(X_val))

    # --------------------------------------
    # B. Build development and validation trials
    # --------------------------------------

    X_dev_pairs, y_dev_pairs = build_training_trials(
        X_dev,
        y_dev,
    )

    X_val_pairs, y_val_pairs = build_test_trials(
        X_dev,
        y_dev,
        X_val,
        y_val,
    )

    print("\nDevelopment trials:", len(X_dev_pairs))
    print("Validation trials:", len(X_val_pairs))
    print("Validation Genuine:", np.sum(y_val_pairs == 1))
    print("Validation Impostor:", np.sum(y_val_pairs == 0))

    # --------------------------------------
    # C. Train temporary models and find thresholds
    # --------------------------------------

    temporary_models = create_models()

    selected_thresholds = {}

    print("\n========== THRESHOLD SELECTION ==========")

    for model_name, model in temporary_models.items():

        print(f"\nTraining validation model: {model_name}")

        model.fit(X_dev_pairs, y_dev_pairs)

        val_scores = get_scores(model, X_val_pairs)

        threshold, eer = find_eer_threshold(
            y_val_pairs,
            val_scores,
        )

        selected_thresholds[model_name] = threshold

        val_metrics = evaluate_at_threshold(
            y_val_pairs,
            val_scores,
            threshold,
        )

        print(f"Selected Threshold: {threshold:.6f}")
        print(f"Validation EER: {eer:.4f}")
        print(f"Validation FAR: {val_metrics['FAR']:.4f}")
        print(f"Validation FRR: {val_metrics['FRR']:.4f}")

    # --------------------------------------
    # D. Rebuild final training trials
    #    using all six Session 1 recordings
    # --------------------------------------

    print("\nBuilding final training trials...")

    X_final_pairs, y_final_pairs = build_training_trials(
        X_train,
        y_train,
    )

    # Test trials use Session 1 enrollment references
    X_test_pairs, y_test_pairs = build_test_trials(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    print("Final training trials:", len(X_final_pairs))
    print("Final testing trials:", len(X_test_pairs))

    # --------------------------------------
    # E. Retrain and evaluate final models
    # --------------------------------------

    final_models = create_models()

    all_results = []

    print("\n========== FINAL TEST RESULTS ==========")

    for model_name, model in final_models.items():

        print(f"\nTraining final model: {model_name}")

        model.fit(X_final_pairs, y_final_pairs)

        test_scores = get_scores(model, X_test_pairs)

        threshold = selected_thresholds[model_name]

        metrics = evaluate_at_threshold(
            y_test_pairs,
            test_scores,
            threshold,
        )

        metrics["Model"] = model_name
        all_results.append(metrics)

        print(f"Threshold: {threshold:.6f}")

        for metric, value in metrics.items():
            if metric not in ["Model", "Threshold"]:
                print(f"{metric}: {value:.4f}"
                      if isinstance(value, (float, np.floating))
                      else f"{metric}: {value}")

        # Save model together with its selected threshold
        filename = (
            "svm_face_tuned.joblib"
            if model_name == "SVM"
            else "random_forest_face_tuned.joblib"
        )

        joblib.dump(
            {
                "model": model,
                "threshold": threshold,
                "feature_columns": feature_cols,
            },
            MODEL_DIR / filename,
        )

        print("Saved:", MODEL_DIR / filename)

    # --------------------------------------
    # F. Save comparison
    # --------------------------------------

    results_df = pd.DataFrame(all_results)

    results_path = (
        RESULTS_DIR / "face_verification_tuned_results.csv"
    )

    results_df.to_csv(results_path, index=False)

    print("\n========== TUNED MODEL COMPARISON ==========")

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
    print("\nThreshold optimization completed.")


if __name__ == "__main__":
    main()
