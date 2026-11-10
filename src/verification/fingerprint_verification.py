
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)


# =====================================================
# 1. CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fingerprint"
    / "splits"
    / "real"
)

RESULTS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "fingerprint"
)

MODELS_ROOT = (
    PROJECT_ROOT
    / "saved_models"
    / "fingerprint"
)

TARGET_FAR = 0.01
RANDOM_STATE = 42

SPLITS = {
    "enrollment": SPLIT_ROOT / "fingerprint_enrollment.csv",
    "validation": SPLIT_ROOT / "fingerprint_validation.csv",
    "test": SPLIT_ROOT / "fingerprint_test.csv"
}

# Feature columns only.
FEATURE_PREFIXES = (
    "fingerprint_hog_",
    "fingerprint_lbp_"
)


# =====================================================
# 2. LOAD DATA
# =====================================================

def load_split(path):

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = {
        "identity",
        "database",
        "subject_id",
        "image_name"
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in {path.name}: {missing}"
        )

    feature_columns = [
        column for column in df.columns
        if column.startswith(FEATURE_PREFIXES)
    ]

    if not feature_columns:
        raise ValueError("No fingerprint features found.")

    X = df[feature_columns].to_numpy(dtype=np.float32)

    if not np.isfinite(X).all():
        raise ValueError(
            f"Non-finite feature values in {path.name}"
        )

    return df, X, feature_columns


# =====================================================
# 3. BUILD ENROLLMENT TEMPLATES
# =====================================================

def build_templates(df, X):

    templates = {}
    indices = {}

    for identity in sorted(df["identity"].unique()):

        positions = np.where(
            df["identity"].to_numpy() == identity
        )[0]

        if len(positions) != 4:
            raise ValueError(
                f"{identity} has {len(positions)} "
                "enrollment images; expected 4."
            )

        templates[identity] = X[positions].mean(axis=0)
        indices[identity] = positions

    return templates, indices


# =====================================================
# 4. BUILD PAIRWISE TRAINING TRIALS
# =====================================================

def build_training_trials(df, X):

    identities = sorted(df["identity"].unique())

    templates, identity_indices = build_templates(df, X)

    trial_features = []
    trial_labels = []
    trial_metadata = []

    for identity in identities:

        query_indices = identity_indices[identity]

        for query_index in query_indices:

            query = X[query_index]

            for claimed_identity in identities:

                if claimed_identity == identity:

                    # Leave-one-out genuine template.
                    other_indices = [
                        idx for idx in query_indices
                        if idx != query_index
                    ]

                    reference = X[other_indices].mean(axis=0)

                    label = 1

                else:

                    reference = templates[claimed_identity]

                    label = 0

                difference = np.abs(query - reference)

                trial_features.append(difference)

                trial_labels.append(label)

                trial_metadata.append({
                    "probe_identity": identity,
                    "claimed_identity": claimed_identity,
                    "probe_image": df.iloc[query_index]["image_name"],
                    "trial_type": (
                        "genuine" if label == 1 else "impostor"
                    )
                })

    return (
        np.asarray(trial_features, dtype=np.float32),
        np.asarray(trial_labels, dtype=np.int32),
        pd.DataFrame(trial_metadata)
    )


# =====================================================
# 5. BUILD VALIDATION / TEST TRIALS
# =====================================================

def build_evaluation_trials(
    enrollment_df,
    enrollment_X,
    probe_df,
    probe_X
):

    templates, _ = build_templates(
        enrollment_df,
        enrollment_X
    )

    identities = sorted(templates.keys())

    trial_features = []
    trial_labels = []
    trial_metadata = []

    for probe_index, row in probe_df.iterrows():

        probe = probe_X[probe_index]

        actual_identity = row["identity"]

        for claimed_identity in identities:

            reference = templates[claimed_identity]

            difference = np.abs(probe - reference)

            label = int(
                actual_identity == claimed_identity
            )

            trial_features.append(difference)

            trial_labels.append(label)

            trial_metadata.append({
                "probe_identity": actual_identity,
                "claimed_identity": claimed_identity,
                "probe_image": row["image_name"],
                "trial_type": (
                    "genuine" if label == 1 else "impostor"
                )
            })

    return (
        np.asarray(trial_features, dtype=np.float32),
        np.asarray(trial_labels, dtype=np.int32),
        pd.DataFrame(trial_metadata)
    )


# =====================================================
# 6. MODEL DEFINITIONS
# =====================================================

def create_models():

    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(
            n_components=100,
            svd_solver="randomized",
            random_state=RANDOM_STATE
        )),
        ("classifier", SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])

    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=1
    )

    return {
        "SVM": svm,
        "RandomForest": rf
    }


# =====================================================
# 7. SELECT THRESHOLD USING VALIDATION ONLY
# =====================================================

def select_threshold(y_true, scores, target_far=0.01):

    impostor_scores = scores[y_true == 0]

    genuine_scores = scores[y_true == 1]

    if len(impostor_scores) == 0 or len(genuine_scores) == 0:
        raise ValueError("Validation requires both trial classes.")

    candidates = np.unique(
        np.concatenate(([0.0], scores, [1.0]))
    )

    valid_thresholds = []

    for threshold in candidates:

        far = np.mean(impostor_scores >= threshold)

        if far <= target_far:
            valid_thresholds.append(threshold)

    if not valid_thresholds:
        raise ValueError(
            "Could not find a threshold satisfying target FAR."
        )

    # Lowest threshold satisfying the FAR constraint
    # gives the best genuine acceptance under that constraint.
    threshold = float(min(valid_thresholds))

    validation_far = float(
        np.mean(impostor_scores >= threshold)
    )

    validation_frr = float(
        np.mean(genuine_scores < threshold)
    )

    return threshold, validation_far, validation_frr


# =====================================================
# 8. METRICS
# =====================================================

def calculate_metrics(y_true, scores, threshold):

    predictions = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    far = fp / (fp + tn) if (fp + tn) else 0.0

    frr = fn / (fn + tp) if (fn + tp) else 0.0

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "far": float(far),
        "frr": float(frr),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "genuine_trials": int(np.sum(y_true == 1)),
        "impostor_trials": int(np.sum(y_true == 0))
    }

    return metrics, predictions


# =====================================================
# 9. SAVE VISUALIZATIONS
# =====================================================

def save_visualizations(
    model_name,
    y_true,
    scores,
    predictions
):

    model_folder = RESULTS_ROOT / model_name

    model_folder.mkdir(parents=True, exist_ok=True)

    # Confusion Matrix
    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Impostor", "Genuine"]
    )

    display.plot(cmap="Blues", values_format="d")

    plt.title(f"{model_name} - Fingerprint Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        model_folder / "confusion_matrix.png",
        dpi=300
    )

    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, scores)

    auc = roc_auc_score(y_true, scores)

    plt.figure(figsize=(7, 6))

    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC = {auc:.4f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} - Fingerprint ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        model_folder / "roc_curve.png",
        dpi=300
    )

    plt.close()


# =====================================================
# 10. MAIN PIPELINE
# =====================================================

def main():

    warnings.filterwarnings(
        "ignore",
        category=UserWarning
    )

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)

    print("\nFINGERPRINT VERIFICATION PIPELINE")
    print("=" * 60)

    enrollment_df, enrollment_X, feature_columns = load_split(
        SPLITS["enrollment"]
    )

    validation_df, validation_X, validation_columns = load_split(
        SPLITS["validation"]
    )

    test_df, test_X, test_columns = load_split(
        SPLITS["test"]
    )

    if (
        feature_columns != validation_columns
        or feature_columns != test_columns
    ):
        raise ValueError("Feature columns are not aligned.")

    enrollment_ids = set(enrollment_df["identity"])
    validation_ids = set(validation_df["identity"])
    test_ids = set(test_df["identity"])

    if not (
        enrollment_ids == validation_ids == test_ids
    ):
        raise ValueError("Identity sets are not aligned.")

    print(f"Identities: {len(enrollment_ids)}")
    print(f"Features per image: {enrollment_X.shape[1]}")
    print(f"Enrollment images: {len(enrollment_df)}")
    print(f"Validation images: {len(validation_df)}")
    print(f"Test images: {len(test_df)}")

    # Training trials from enrollment using leave-one-out.
    X_train, y_train, train_metadata = build_training_trials(
        enrollment_df,
        enrollment_X
    )

    # Validation trials against full enrollment templates.
    X_validation, y_validation, validation_metadata = (
        build_evaluation_trials(
            enrollment_df,
            enrollment_X,
            validation_df,
            validation_X
        )
    )

    # Test trials against full enrollment templates.
    X_test, y_test, test_metadata = build_evaluation_trials(
        enrollment_df,
        enrollment_X,
        test_df,
        test_X
    )

    print("\nTRIAL COUNTS")
    print("-" * 60)

    print(
        f"Training: {len(y_train)} "
        f"(Genuine: {(y_train == 1).sum()}, "
        f"Impostor: {(y_train == 0).sum()})"
    )

    print(
        f"Validation: {len(y_validation)} "
        f"(Genuine: {(y_validation == 1).sum()}, "
        f"Impostor: {(y_validation == 0).sum()})"
    )

    print(
        f"Test: {len(y_test)} "
        f"(Genuine: {(y_test == 1).sum()}, "
        f"Impostor: {(y_test == 0).sum()})"
    )

    models = create_models()

    comparison = []

    for model_name, model in models.items():

        print("\n" + "=" * 60)
        print(f"TRAINING {model_name}")
        print("=" * 60)

        model.fit(X_train, y_train)

        validation_scores = model.predict_proba(
            X_validation
        )[:, 1]

        threshold, validation_far, validation_frr = (
            select_threshold(
                y_validation,
                validation_scores,
                TARGET_FAR
            )
        )

        print(f"Selected threshold: {threshold:.6f}")
        print(f"Validation FAR: {validation_far:.4%}")
        print(f"Validation FRR: {validation_frr:.4%}")

        # Final test: threshold is now frozen.
        test_scores = model.predict_proba(X_test)[:, 1]

        metrics, predictions = calculate_metrics(
            y_test,
            test_scores,
            threshold
        )

        metrics["model"] = model_name
        metrics["validation_far"] = validation_far
        metrics["validation_frr"] = validation_frr

        comparison.append(metrics)

        print("\nTEST RESULTS")

        for key, value in metrics.items():

            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

        # Save predictions and trial metadata.
        output = test_metadata.copy()

        output["true_label"] = y_test
        output["score"] = test_scores
        output["prediction"] = predictions

        model_folder = RESULTS_ROOT / model_name
        model_folder.mkdir(parents=True, exist_ok=True)

        output.to_csv(
            model_folder / "test_trial_results.csv",
            index=False
        )

        # Save model and threshold.
        joblib.dump(
            {
                "model": model,
                "threshold": threshold,
                "feature_columns": feature_columns,
                "target_far": TARGET_FAR
            },
            MODELS_ROOT / f"{model_name.lower()}_fingerprint.joblib"
        )

        save_visualizations(
            model_name,
            y_test,
            test_scores,
            predictions
        )

    # Save model comparison.
    comparison_df = pd.DataFrame(comparison)

    comparison_df.to_csv(
        RESULTS_ROOT / "fingerprint_model_comparison.csv",
        index=False
    )

    with open(
        RESULTS_ROOT / "fingerprint_run_summary.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "dataset": "FVC2002 Set B - DB1_B, DB2_B, DB3_B",
                "identities": len(enrollment_ids),
                "features": len(feature_columns),
                "target_far": TARGET_FAR,
                "training_trials": len(y_train),
                "validation_trials": len(y_validation),
                "test_trials": len(y_test),
                "models": comparison
            },
            file,
            indent=4
        )

    print("\n" + "=" * 60)
    print("FINGERPRINT VERIFICATION COMPLETED")
    print("=" * 60)

    print(
        f"Comparison saved to:\n"
        f"{RESULTS_ROOT / 'fingerprint_model_comparison.csv'}"
    )

    print(
        f"Models saved to:\n{MODELS_ROOT}"
    )


if __name__ == "__main__":
    main()
