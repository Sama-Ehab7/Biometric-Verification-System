
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
)

# ==========================================
# 1. PROJECT PATHS AND IMPORTS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Allow importing face_verification.py
VERIFICATION_DIR = PROJECT_ROOT / "src" / "verification"

sys.path.insert(0, str(VERIFICATION_DIR))

from face_verification import (
    load_data,
    build_test_trials,
)

MODEL_DIR = PROJECT_ROOT / "saved_models" / "face"
RESULTS_DIR = PROJECT_ROOT / "results" / "face"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD SESSION 3 TEST DATA
# ==========================================

def load_session_3_data():

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

    test_df = test_df.copy()

    test_df["session"] = (
        test_df["session"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    session_3 = test_df[
        test_df["session"] == "session_3"
    ].copy()

    if session_3.empty:
        raise ValueError("Session 3 data is empty.")

    X_session_3 = session_3[feature_cols].to_numpy(
        dtype=np.float32
    )

    y_session_3 = (
        session_3["speaker"].astype(str).to_numpy()
    )

    print("\nSession 3 recordings:", len(session_3))
    print("Session 3 speakers:", len(np.unique(y_session_3)))

    return (
        X_train,
        y_train,
        X_session_3,
        y_session_3,
    )


# ==========================================
# 3. GET GENUINE CLASS SCORES
# ==========================================

def get_scores(model, X):

    positive_index = list(model.classes_).index(1)

    return model.predict_proba(X)[:, positive_index]


# ==========================================
# 4. PLOT CONFUSION MATRIX
# ==========================================

def save_confusion_matrix(
    y_true,
    predictions,
    model_name,
    threshold,
):

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Impostor", "Genuine"],
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )

    ax.set_title(
        f"{model_name} - Face Verification\n"
        f"Threshold = {threshold:.4f}"
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    fig.tight_layout()

    filename = (
        f"{model_name.lower().replace(' ', '_')}_"
        "confusion_matrix.png"
    )

    output_path = RESULTS_DIR / filename

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Saved:", output_path)


# ==========================================
# 5. PLOT ROC CURVE
# ==========================================

def save_roc_curve(all_scores, y_true):

    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, scores in all_scores.items():

        fpr, tpr, _ = roc_curve(
            y_true,
            scores,
        )

        auc = roc_auc_score(
            y_true,
            scores,
        )

        ax.plot(
            fpr,
            tpr,
            label=f"{model_name} (AUC = {auc:.4f})",
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier",
    )

    ax.set_title("Face Verification - ROC Curve")
    ax.set_xlabel("False Acceptance Rate (FAR)")
    ax.set_ylabel("True Acceptance Rate (TAR)")

    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    output_path = RESULTS_DIR / "face_roc_curve.png"

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Saved:", output_path)


# ==========================================
# 6. PLOT FAR VS FRR CURVE
# ==========================================

def save_far_frr_curve(all_scores, y_true):

    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, scores in all_scores.items():

        far, tar, thresholds = roc_curve(
            y_true,
            scores,
        )

        frr = 1 - tar

        ax.plot(
            thresholds,
            far,
            label=f"{model_name} - FAR",
        )

        ax.plot(
            thresholds,
            frr,
            linestyle="--",
            label=f"{model_name} - FRR",
        )

    ax.set_title("Face Verification - FAR vs FRR")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Error Rate")

    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    output_path = RESULTS_DIR / "face_far_frr_curve.png"

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Saved:", output_path)


# ==========================================
# 7. MAIN EVALUATION PIPELINE
# ==========================================

def main():

    print("=" * 60)
    print("FACE VERIFICATION - EVALUATION VISUALIZATION")
    print("=" * 60)

    # Load Session 1 enrollment and Session 3 test data
    (
        X_train,
        y_train,
        X_test,
        y_test,
    ) = load_session_3_data()

    # Build exhaustive Session 3 verification trials
    print("\nBuilding Session 3 verification trials...")

    X_test_pairs, y_test_pairs = build_test_trials(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    print("Testing trials:", len(X_test_pairs))
    print("Genuine trials:", np.sum(y_test_pairs == 1))
    print("Impostor trials:", np.sum(y_test_pairs == 0))

    # Models and their selected 1% FAR thresholds
    models_to_evaluate = {
        "SVM": {
            "file": "svm_face_security.joblib",
            "far_target": 0.01,
        },
        "Random Forest": {
            "file": "random_forest_face_security.joblib",
            "far_target": 0.01,
        },
    }

    all_scores = {}
    results = []

    # --------------------------------------
    # Evaluate each model
    # --------------------------------------

    for model_name, config in models_to_evaluate.items():

        model_path = MODEL_DIR / config["file"]

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        saved_data = joblib.load(model_path)

        model = saved_data["model"]

        threshold = saved_data["thresholds"][
            config["far_target"]
        ]

        print(f"\nEvaluating {model_name}...")
        print(f"Threshold: {threshold:.6f}")

        scores = get_scores(
            model,
            X_test_pairs,
        )

        predictions = (
            scores >= threshold
        ).astype(int)

        all_scores[model_name] = scores

        # Calculate confusion matrix
        cm = confusion_matrix(
            y_test_pairs,
            predictions,
            labels=[0, 1],
        )

        tn, fp, fn, tp = cm.ravel()

        # Calculate verification metrics
        far = fp / (fp + tn) if (fp + tn) else 0
        frr = fn / (fn + tp) if (fn + tp) else 0

        auc = roc_auc_score(
            y_test_pairs,
            scores,
        )

        results.append({
            "Model": model_name,
            "Threshold": threshold,
            "FAR": far,
            "FRR": frr,
            "ROC-AUC": auc,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
        })

        print(f"FAR: {far:.4f}")
        print(f"FRR: {frr:.4f}")
        print(f"ROC-AUC: {auc:.4f}")

        # Save confusion matrix
        save_confusion_matrix(
            y_test_pairs,
            predictions,
            model_name,
            threshold,
        )

    # --------------------------------------
    # Generate comparison visualizations
    # --------------------------------------

    print("\nGenerating ROC Curve...")

    save_roc_curve(
        all_scores,
        y_test_pairs,
    )

    print("\nGenerating FAR-FRR Curve...")

    save_far_frr_curve(
        all_scores,
        y_test_pairs,
    )

    # --------------------------------------
    # Save final evaluation CSV
    # --------------------------------------

    results_df = pd.DataFrame(results)

    results_path = (
        RESULTS_DIR / "face_final_evaluation.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    print("\n" + "=" * 60)
    print("FINAL FACE EVALUATION")
    print("=" * 60)

    print(results_df.to_string(index=False))

    print("\nGenerated files:")

    for file in sorted(RESULTS_DIR.glob("*.png")):
        print(" -", file.name)

    print("\nResults saved:", results_path)

    print("\nFace evaluation visualization completed successfully.")


if __name__ == "__main__":
    main()
