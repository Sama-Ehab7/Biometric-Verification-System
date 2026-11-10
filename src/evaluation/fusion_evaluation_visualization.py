
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
MODEL_DIR = PROJECT_ROOT / "saved_models" / "fusion"
RESULTS_DIR = PROJECT_ROOT / "results" / "fusion"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "multimodal_train.csv"
TEST_PATH = DATA_DIR / "multimodal_test.csv"

SVM_MODEL_PATH = MODEL_DIR / "svm_early_fusion_security.joblib"
RF_MODEL_PATH = MODEL_DIR / "random_forest_early_fusion_security.joblib"

TARGET_FARS = [0.01, 0.05, 0.10]


# ============================================================
# 2. LOAD DATA AND MODELS
# ============================================================

def load_data():
    print("Loading multimodal datasets...")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"Training data shape: {train_df.shape}")
    print(f"Testing data shape: {test_df.shape}")

    # Keep Session 1 for enrollment
    train_df = train_df[train_df["session"] == "session_1"].copy()

    # Keep Session 3 only for final testing
    test_df = test_df[test_df["session"] == "session_3"].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Training or Session 3 data is empty. "
            "Check the session column values."
        )

    print(f"Session 1 enrollment records: {len(train_df)}")
    print(f"Session 3 test records: {len(test_df)}")

    return train_df, test_df


def load_models():
    print("\nLoading Early Fusion models...")

    svm_artifact = joblib.load(SVM_MODEL_PATH)
    rf_artifact = joblib.load(RF_MODEL_PATH)

    return svm_artifact, rf_artifact


# ============================================================
# 3. BUILD SESSION 3 VERIFICATION TRIALS
# ============================================================

def build_test_trials(train_df, test_df, feature_columns):
    """
    Each Session 3 probe is compared against all enrolled speakers.

    Pairwise feature:
        abs(probe_features - enrollment_centroid)

    Label:
        1 = Genuine
        0 = Impostor
    """

    print("\nBuilding Session 3 verification trials...")

    enrollment_centroids = (
        train_df.groupby("speaker")[feature_columns]
        .mean()
    )

    enrolled_speakers = enrollment_centroids.index.tolist()

    X_trials = []
    y_trials = []

    for _, probe in test_df.iterrows():

        probe_features = probe[feature_columns].to_numpy(dtype=np.float64)

        for enrolled_speaker in enrolled_speakers:

            centroid = enrollment_centroids.loc[
                enrolled_speaker
            ].to_numpy(dtype=np.float64)

            difference = np.abs(probe_features - centroid)

            X_trials.append(difference)

            label = int(probe["speaker"] == enrolled_speaker)
            y_trials.append(label)

    X_trials = np.asarray(X_trials, dtype=np.float64)
    y_trials = np.asarray(y_trials, dtype=np.int64)

    print(f"Number of trials: {len(y_trials)}")
    print(f"Genuine trials: {np.sum(y_trials == 1)}")
    print(f"Impostor trials: {np.sum(y_trials == 0)}")

    return X_trials, y_trials


# ============================================================
# 4. GET GENUINE CLASS PROBABILITY
# ============================================================

def get_genuine_scores(model, X):
    """
    Return probability of class 1 (Genuine).
    """

    probabilities = model.predict_proba(X)

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError(
            f"Model does not contain Genuine class 1. Classes: {classes}"
        )

    genuine_index = classes.index(1)

    return probabilities[:, genuine_index]


# ============================================================
# 5. METRICS
# ============================================================

def calculate_metrics(y_true, scores, threshold):
    y_pred = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0

    auc = roc_auc_score(y_true, scores)

    return {
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FAR": far,
        "FRR": frr,
        "ROC-AUC": auc,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


# ============================================================
# 6. CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(y_true, scores, threshold, model_name):
    y_pred = (scores >= threshold).astype(int)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Impostor", "Genuine"],
        yticklabels=["Impostor", "Genuine"],
    )

    plt.title(
        f"{model_name} - Early Fusion Confusion Matrix\n"
        f"Target FAR = 1%, Threshold = {threshold:.6f}"
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    plt.tight_layout()

    filename = (
        "svm_confusion_matrix.png"
        if model_name == "SVM"
        else "random_forest_confusion_matrix.png"
    )

    plt.savefig(RESULTS_DIR / filename, dpi=300)
    plt.close()

    print(f"Saved: {filename}")


# ============================================================
# 7. ROC CURVE
# ============================================================

def plot_roc_curve(y_true, svm_scores, rf_scores):
    plt.figure(figsize=(8, 6))

    for scores, model_name in [
        (svm_scores, "SVM"),
        (rf_scores, "Random Forest"),
    ]:
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)

        plt.plot(
            fpr,
            tpr,
            label=f"{model_name} (AUC = {auc:.4f})"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    plt.title("Early Fusion - ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "fusion_roc_curve.png",
        dpi=300
    )

    plt.close()

    print("Saved: fusion_roc_curve.png")


# ============================================================
# 8. FAR-FRR CURVE
# ============================================================

def plot_far_frr_curve(y_true, svm_scores, rf_scores):
    thresholds = np.linspace(0, 1, 500)

    plt.figure(figsize=(9, 6))

    for scores, model_name in [
        (svm_scores, "SVM"),
        (rf_scores, "Random Forest"),
    ]:

        far_values = []
        frr_values = []

        for threshold in thresholds:
            metrics = calculate_metrics(
                y_true,
                scores,
                threshold
            )

            far_values.append(metrics["FAR"])
            frr_values.append(metrics["FRR"])

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

    plt.title("Early Fusion - FAR and FRR vs Threshold")
    plt.xlabel("Verification Threshold")
    plt.ylabel("Error Rate")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "fusion_far_frr_curve.png",
        dpi=300
    )

    plt.close()

    print("Saved: fusion_far_frr_curve.png")


# ============================================================
# 9. MAIN EVALUATION
# ============================================================

def main():
    print("=" * 65)
    print("EARLY FUSION - FINAL EVALUATION VISUALIZATION")
    print("=" * 65)

    train_df, test_df = load_data()

    svm_artifact, rf_artifact = load_models()

    # Use exact feature order saved with the trained models
    svm_features = svm_artifact["feature_columns"]
    rf_features = rf_artifact["feature_columns"]

    if svm_features != rf_features:
        raise ValueError(
            "SVM and Random Forest feature columns are not identical."
        )

    feature_columns = svm_features

    print(f"\nNumber of fusion features: {len(feature_columns)}")

    X_test, y_test = build_test_trials(
        train_df,
        test_df,
        feature_columns
    )

    # Load trained estimators
    svm_model = svm_artifact["model"]
    rf_model = rf_artifact["model"]

    # Predict Genuine probabilities
    svm_scores = get_genuine_scores(svm_model, X_test)
    rf_scores = get_genuine_scores(rf_model, X_test)

    results = []

    # Evaluate each model at all saved security thresholds
    for model_name, artifact, scores in [
        ("SVM", svm_artifact, svm_scores),
        ("Random Forest", rf_artifact, rf_scores),
    ]:

        print(f"\n{'=' * 55}")
        print(f"{model_name} RESULTS")
        print("=" * 55)

        saved_thresholds = artifact["thresholds"]

        for target_far in TARGET_FARS:

            threshold = saved_thresholds[target_far]

            metrics = calculate_metrics(
                y_test,
                scores,
                threshold
            )

            row = {
                "Model": model_name,
                "Target FAR": target_far,
                **metrics,
            }

            results.append(row)

            print(f"\nTarget FAR: {target_far * 100:.0f}%")
            print(f"Threshold: {threshold:.6f}")
            print(f"Accuracy: {metrics['Accuracy'] * 100:.2f}%")
            print(f"Precision: {metrics['Precision'] * 100:.2f}%")
            print(f"Recall: {metrics['Recall'] * 100:.2f}%")
            print(f"F1-score: {metrics['F1'] * 100:.2f}%")
            print(f"FAR: {metrics['FAR'] * 100:.2f}%")
            print(f"FRR: {metrics['FRR'] * 100:.2f}%")
            print(f"ROC-AUC: {metrics['ROC-AUC']:.4f}")

            print(
                f"TN={metrics['TN']} | FP={metrics['FP']} | "
                f"FN={metrics['FN']} | TP={metrics['TP']}"
            )

        # Confusion matrix at the 1% target FAR threshold
        threshold_1_percent = saved_thresholds[0.01]

        plot_confusion_matrix(
            y_test,
            scores,
            threshold_1_percent,
            model_name
        )

    # Save evaluation results
    results_df = pd.DataFrame(results)

    csv_path = RESULTS_DIR / "fusion_final_evaluation.csv"
    results_df.to_csv(csv_path, index=False)

    print(f"\nSaved: {csv_path.name}")

    # Generate comparison plots
    plot_roc_curve(
        y_test,
        svm_scores,
        rf_scores
    )

    plot_far_frr_curve(
        y_test,
        svm_scores,
        rf_scores
    )

    print("\n" + "=" * 65)
    print("EARLY FUSION EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 65)

    print("\nGenerated files:")
    for filename in [
        "svm_confusion_matrix.png",
        "random_forest_confusion_matrix.png",
        "fusion_roc_curve.png",
        "fusion_far_frr_curve.png",
        "fusion_final_evaluation.csv",
    ]:
        print(f"- results/fusion/{filename}")


if __name__ == "__main__":
    main()
