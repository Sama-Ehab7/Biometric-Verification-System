
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

from voice_verification import (
    load_data,
    build_training_trials,
    build_test_trials,
    create_models,
    get_scores,
)


# ==========================================
# 1. CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "saved_models" / "voice"
RESULTS_DIR = PROJECT_ROOT / "results" / "voice"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FAR_TARGETS = [0.01, 0.05, 0.10]


# ==========================================
# 2. SELECT THRESHOLD UNDER FAR CONSTRAINT
# ==========================================

def find_threshold_for_far(y_true, scores, far_target):

    fpr, tpr, thresholds = roc_curve(
        y_true,
        scores,
    )

    # Find thresholds satisfying the validation FAR target
    valid_indices = np.where(
        fpr <= far_target
    )[0]

    if len(valid_indices) == 0:
        raise ValueError(
            f"No threshold satisfies FAR <= {far_target}"
        )

    # Maximize genuine acceptance rate
    best_tpr = np.max(tpr[valid_indices])

    best_indices = valid_indices[
        np.isclose(tpr[valid_indices], best_tpr)
    ]

    # Choose the highest threshold when TPR is tied
    best_index = best_indices[
        np.argmax(thresholds[best_indices])
    ]

    return thresholds[best_index]


# ==========================================
# 3. EVALUATE MODEL
# ==========================================

def evaluate_model(y_true, scores, threshold):

    predictions = (
        scores >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    far = fp / (fp + tn) if (fp + tn) else 0.0
    frr = fn / (fn + tp) if (fn + tp) else 0.0

    return {
        "Threshold": float(threshold),
        "Accuracy": accuracy_score(
            y_true, predictions
        ),
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
        "ROC-AUC": roc_auc_score(
            y_true, scores
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


# ==========================================
# 4. MAIN PIPELINE
# ==========================================

def main():

    print("=" * 65)
    print("VOICE VERIFICATION - SECURITY THRESHOLD OPTIMIZATION")
    print("=" * 65)

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

    y_val = (
        session_2["speaker"].astype(str).to_numpy()
    )

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

    # --------------------------------------
    # C. Build trials
    # --------------------------------------

    print("\nBuilding Session 1 training trials...")

    X_train_pairs, y_train_pairs = build_training_trials(
        X_train,
        y_train,
    )

    print("Training trials:", len(X_train_pairs))
    print("Genuine:", np.sum(y_train_pairs == 1))
    print("Impostor:", np.sum(y_train_pairs == 0))

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

    print("\nBuilding Session 3 testing trials...")

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
    # D. Train models
    # --------------------------------------

    models = create_models()

    all_results = []

    print("\n" + "=" * 65)
    print("SECURITY THRESHOLD EXPERIMENTS")
    print("=" * 65)

    for model_name, model in models.items():

        print(f"\nTraining {model_name}...")

        model.fit(
            X_train_pairs,
            y_train_pairs,
        )

        # Scores for threshold selection
        val_scores = get_scores(
            model,
            X_val_pairs,
        )

        # Scores for final independent testing
        test_scores = get_scores(
            model,
            X_test_pairs,
        )

        selected_thresholds = {}

        for far_target in FAR_TARGETS:

            # Select threshold using Session 2 only
            threshold = find_threshold_for_far(
                y_val_pairs,
                val_scores,
                far_target,
            )

            selected_thresholds[far_target] = float(threshold)

            # Validation metrics
            val_metrics = evaluate_model(
                y_val_pairs,
                val_scores,
                threshold,
            )

            # Final Session 3 metrics
            test_metrics = evaluate_model(
                y_test_pairs,
                test_scores,
                threshold,
            )

            result = {
                "Model": model_name,
                "Target_FAR": far_target,
                "Threshold": threshold,
                "Validation_FAR": val_metrics["FAR"],
                "Validation_FRR": val_metrics["FRR"],
                "Test_Accuracy": test_metrics["Accuracy"],
                "Test_Precision": test_metrics["Precision"],
                "Test_Recall": test_metrics["Recall"],
                "Test_F1": test_metrics["F1 Score"],
                "Test_FAR": test_metrics["FAR"],
                "Test_FRR": test_metrics["FRR"],
                "Test_ROC_AUC": test_metrics["ROC-AUC"],
                "TN": test_metrics["TN"],
                "FP": test_metrics["FP"],
                "FN": test_metrics["FN"],
                "TP": test_metrics["TP"],
            }

            all_results.append(result)

            print(
                f"\n--- {model_name} | "
                f"Target FAR: {far_target:.0%} ---"
            )

            print(f"Selected Threshold: {threshold:.6f}")
            print(
                f"Validation FAR: {val_metrics['FAR']:.4f}"
            )
            print(
                f"Validation FRR: {val_metrics['FRR']:.4f}"
            )
            print(
                f"Test FAR: {test_metrics['FAR']:.4f}"
            )
            print(
                f"Test FRR: {test_metrics['FRR']:.4f}"
            )
            print(
                f"Test Recall: {test_metrics['Recall']:.4f}"
            )
            print(
                f"Test Precision: {test_metrics['Precision']:.4f}"
            )

        # Save model with all selected thresholds
        filename = (
            "svm_voice_security.joblib"
            if model_name == "SVM"
            else "random_forest_voice_security.joblib"
        )

        model_path = MODEL_DIR / filename

        joblib.dump(
            {
                "model": model,
                "thresholds": selected_thresholds,
                "feature_columns": feature_cols,
                "enrollment_speakers": np.unique(y_train),
            },
            model_path,
        )

        print("\nSaved:", model_path)

    # --------------------------------------
    # E. Save results
    # --------------------------------------

    results_df = pd.DataFrame(all_results)

    results_path = (
        RESULTS_DIR / "voice_security_threshold_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    print("\n" + "=" * 65)
    print("FINAL VOICE SECURITY COMPARISON")
    print("=" * 65)

    print(
        results_df[
            [
                "Model",
                "Target_FAR",
                "Threshold",
                "Test_Accuracy",
                "Test_Precision",
                "Test_Recall",
                "Test_F1",
                "Test_FAR",
                "Test_FRR",
            ]
        ].to_string(index=False)
    )

    print("\nResults saved:", results_path)
    print("\nVoice security threshold optimization completed.")


if __name__ == "__main__":
    main()
