from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src" / "verification")
)

from voice_verification import load_data, build_test_trials


SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
MODELS_DIR = PROJECT_ROOT / "saved_models" / "voice"
RESULTS_DIR = PROJECT_ROOT / "results" / "voice"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading voice datasets...")

(
    train_df,
    test_df,
    X_train,
    X_test,
    y_train,
    y_test,
    feature_cols,
) = load_data()

print(f"Training recordings: {len(train_df)}")
print(f"Testing recordings: {len(test_df)}")
print(f"Voice features: {len(feature_cols)}")


# ============================================================
# 3. SELECT SESSION 3 AS FINAL TEST SET
# ============================================================

session_3 = test_df[
    test_df["session"].astype(str).str.lower().str.strip()
    == "session_3"
].copy()

X_session_3 = session_3[feature_cols].to_numpy(
    dtype=np.float32
)

y_session_3 = session_3["speaker"].astype(str).to_numpy()

print(f"\nSession 3 recordings: {len(session_3)}")
print(f"Session 3 speakers: {session_3['speaker'].nunique()}")


# ============================================================
# 4. BUILD VERIFICATION TRIALS
# ============================================================

print("\nBuilding verification trials...")

X_test_pairs, y_test_pairs = build_test_trials(
    X_train,
    y_train,
    X_session_3,
    y_session_3,
)

print(f"Total test trials: {len(y_test_pairs)}")
print(f"Genuine trials: {np.sum(y_test_pairs == 1)}")
print(f"Impostor trials: {np.sum(y_test_pairs == 0)}")


# ============================================================
# 5. LOAD SAVED MODELS AND SECURITY THRESHOLDS
# ============================================================

def load_model_and_threshold(path):
    """
    Supports either:
    - A saved sklearn estimator
    - A dictionary containing model and thresholds
    """

    saved_data = joblib.load(path)

    if isinstance(saved_data, dict):
        model = saved_data.get("model")

        if model is None:
            model = saved_data.get("best_model")

        if model is None:
            raise ValueError(
                f"No model found inside: {path}"
            )

        thresholds = saved_data.get("thresholds", {})

        if 0.01 in thresholds:
            threshold = thresholds[0.01]
        elif "0.01" in thresholds:
            threshold = thresholds["0.01"]
        else:
            raise KeyError(
                f"1% FAR threshold not found in: {path}"
            )

    else:
        raise ValueError(
            f"Expected a dictionary containing model and thresholds: {path}"
        )

    return model, float(threshold)


svm_model, svm_threshold = load_model_and_threshold(
    MODELS_DIR / "svm_voice_security.joblib"
)

rf_model, rf_threshold = load_model_and_threshold(
    MODELS_DIR / "random_forest_voice_security.joblib"
)

models = {
    "SVM": {
        "model": svm_model,
        "threshold": svm_threshold,
    },
    "Random Forest": {
        "model": rf_model,
        "threshold": rf_threshold,
    },
}

print("\nLoaded models and thresholds:")
for name, data in models.items():
    print(f"{name}: threshold = {data['threshold']:.6f}")


# ============================================================
# 6. EVALUATION FUNCTION
# ============================================================

def calculate_metrics(y_true, scores, threshold):
    """
    Positive class = Genuine
    Negative class = Impostor
    """

    y_pred = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "FAR": far,
        "FRR": frr,
        "ROC_AUC": roc_auc_score(y_true, scores),
        "Threshold": threshold,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }

    return metrics, y_pred


# ============================================================
# 7. GET MODEL SCORES AND EVALUATE
# ============================================================

all_results = []
all_scores = {}

for model_name, model_data in models.items():

    print(f"\nEvaluating {model_name}...")

    model = model_data["model"]
    threshold = model_data["threshold"]

    scores = model.predict_proba(X_test_pairs)[:, 1]

    metrics, y_pred = calculate_metrics(
        y_test_pairs,
        scores,
        threshold,
    )

    all_scores[model_name] = scores

    metrics["Model"] = model_name
    all_results.append(metrics)

    print(f"Accuracy:  {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall:    {metrics['Recall']:.4f}")
    print(f"F1:        {metrics['F1']:.4f}")
    print(f"FAR:       {metrics['FAR']:.4f}")
    print(f"FRR:       {metrics['FRR']:.4f}")
    print(f"ROC-AUC:   {metrics['ROC_AUC']:.4f}")


# ============================================================
# 8. CONFUSION MATRICES
# ============================================================

for model_name, model_data in models.items():

    model = model_data["model"]
    threshold = model_data["threshold"]

    scores = all_scores[model_name]
    y_pred = (scores >= threshold).astype(int)

    cm = confusion_matrix(
        y_test_pairs,
        y_pred,
        labels=[0, 1]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Impostor", "Genuine"]
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )

    ax.set_title(
        f"Voice Verification - {model_name}\n"
        "Session 3 | Target FAR = 1%"
    )

    plt.tight_layout()

    filename = (
        "svm_confusion_matrix.png"
        if model_name == "SVM"
        else "random_forest_confusion_matrix.png"
    )

    plt.savefig(
        RESULTS_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {filename}")


# ============================================================
# 9. ROC CURVE
# ============================================================

plt.figure(figsize=(8, 6))

for model_name, scores in all_scores.items():

    fpr, tpr, _ = roc_curve(
        y_test_pairs,
        scores
    )

    auc_value = roc_auc_score(
        y_test_pairs,
        scores
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC = {auc_value:.4f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Voice Verification - ROC Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "voice_roc_curve.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("Saved: voice_roc_curve.png")


# ============================================================
# 10. FAR-FRR CURVE
# ============================================================

plt.figure(figsize=(8, 6))

for model_name, scores in all_scores.items():

    thresholds = np.unique(
        np.concatenate([
            np.linspace(0, 1, 500),
            [models[model_name]["threshold"]]
        ])
    )

    far_values = []
    frr_values = []

    for threshold in thresholds:

        y_pred = (scores >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test_pairs,
            y_pred,
            labels=[0, 1]
        ).ravel()

        far = fp / (fp + tn) if (fp + tn) else 0
        frr = fn / (fn + tp) if (fn + tp) else 0

        far_values.append(far)
        frr_values.append(frr)

    plt.plot(
        thresholds,
        far_values,
        label=f"{model_name} FAR"
    )

    plt.plot(
        thresholds,
        frr_values,
        linestyle="--",
        label=f"{model_name} FRR"
    )

plt.xlabel("Decision Threshold")
plt.ylabel("Error Rate")
plt.title("Voice Verification - FAR and FRR Curves")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "voice_far_frr_curve.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("Saved: voice_far_frr_curve.png")


# ============================================================
# 11. SAVE FINAL EVALUATION CSV
# ============================================================

results_df = pd.DataFrame(all_results)

column_order = [
    "Model",
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
    "TP",
]

results_df = results_df[column_order]

csv_path = RESULTS_DIR / "voice_final_evaluation.csv"

results_df.to_csv(
    csv_path,
    index=False,
)

print("\nFinal evaluation results:")
print(results_df.to_string(index=False))

print("\n====================================")
print("VOICE EVALUATION COMPLETED")
print("====================================")
print(f"Results saved in: {RESULTS_DIR}")