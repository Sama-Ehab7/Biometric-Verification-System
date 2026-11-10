"""Non-overwriting controlled audit for the real FVC fingerprint experiment.

The script reuses the existing enrollment/validation/test image splits and
saved baseline models. It tests one block-balanced descriptor representation
with two traditional classifiers and writes only to a new audit directory.
It never selects thresholds from test labels and never saves/replaces models.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[2]
SPLIT_ROOT = ROOT / "data" / "processed" / "fingerprint" / "splits" / "real"
SAVED_MODELS = ROOT / "saved_models" / "fingerprint"
OUTPUT_ROOT = ROOT / "results" / "fingerprint" / "controlled_audit_v1"
TARGET_FAR = 0.01
RANDOM_STATE = 42
METADATA = {"identity", "database", "subject_id", "image_name", "image_path"}


def load_split(name: str):
    frame = pd.read_csv(SPLIT_ROOT / f"fingerprint_{name}.csv")
    frame = frame.reset_index(drop=True)
    hog_cols = [c for c in frame.columns if c.startswith("fingerprint_hog_")]
    lbp_cols = [c for c in frame.columns if c.startswith("fingerprint_lbp_")]
    if len(hog_cols) != 8100 or len(lbp_cols) != 10:
        raise ValueError(f"Unexpected descriptor dimensions in {name}: {len(hog_cols)} HOG, {len(lbp_cols)} LBP")
    values = frame[hog_cols + lbp_cols].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite values in {name} split")
    if not (frame.identity.astype(str) == frame.database.astype(str) + "_" + frame.subject_id.astype(str)).all():
        raise ValueError(f"Database-qualified identities are invalid in {name} split")
    return frame, values, len(hog_cols)


def make_templates(frame, values):
    templates = {}
    indices = {}
    identities = sorted(frame.identity.astype(str).unique())
    for identity in identities:
        idx = np.flatnonzero(frame.identity.astype(str).to_numpy() == identity)
        if len(idx) != 4:
            raise ValueError(f"Expected four enrollment impressions for {identity}; found {len(idx)}")
        templates[identity] = values[idx].mean(axis=0)
        indices[identity] = idx
    return identities, templates, indices


def block_distance(query, reference, hog_size):
    """One average HOG absolute distance and one LBP chi-square distance."""
    hog_distance = float(np.mean(np.abs(query[:hog_size] - reference[:hog_size])))
    q = np.maximum(query[hog_size:], 0.0)
    r = np.maximum(reference[hog_size:], 0.0)
    lbp_distance = float(0.5 * np.sum(((q - r) ** 2) / (q + r + 1e-8)))
    return np.asarray([hog_distance, lbp_distance], dtype=np.float32)


def training_block_trials(frame, values, hog_size):
    identities, templates, indices = make_templates(frame, values)
    features, labels = [], []
    for identity in identities:
        for query_idx in indices[identity]:
            query = values[query_idx]
            for claim in identities:
                reference = (
                    values[[i for i in indices[identity] if i != query_idx]].mean(axis=0)
                    if claim == identity else templates[claim]
                )
                features.append(block_distance(query, reference, hog_size))
                labels.append(int(claim == identity))
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int8)


def evaluation_trials(enrollment_frame, enrollment_values, probe_frame, probe_values, hog_size):
    identities, templates, _ = make_templates(enrollment_frame, enrollment_values)
    features, labels, metadata = [], [], []
    for probe_idx, row in probe_frame.iterrows():
        probe = probe_values[probe_idx]
        actual = str(row.identity)
        for claim in identities:
            features.append(block_distance(probe, templates[claim], hog_size))
            labels.append(int(actual == claim))
            metadata.append({
                "probe_identity": actual,
                "claimed_identity": claim,
                "probe_image": row.image_name,
                "probe_database": row.database,
                "claimed_database": claim.split("_", 2)[0] + "_" + claim.split("_", 2)[1],
                "trial_type": "genuine" if actual == claim else "impostor",
            })
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int8), pd.DataFrame(metadata)


def select_threshold(y_true, scores, target_far=TARGET_FAR):
    impostor = scores[y_true == 0]
    genuine = scores[y_true == 1]
    if not len(impostor) or not len(genuine):
        raise ValueError("Both validation trial classes are required")
    candidates = np.concatenate(([-np.inf], np.unique(scores), [np.inf]))
    feasible = [float(t) for t in candidates if np.mean(impostor >= t) <= target_far]
    if not feasible:
        raise ValueError("No validation threshold satisfies the target FAR")
    threshold = min(feasible)
    return threshold, float(np.mean(impostor >= threshold)), float(np.mean(genuine < threshold))


def compute_metrics(y_true, scores, threshold):
    pred = (scores >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    fpr, tpr, roc_thresholds = roc_curve(y_true, scores)
    eer_index = int(np.argmin(np.abs(fpr - (1.0 - tpr))))
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "far": float(fp / (fp + tn)) if fp + tn else 0.0,
        "frr": float(fn / (fn + tp)) if fn + tp else 0.0,
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "eer_discrete": float((fpr[eer_index] + 1.0 - tpr[eer_index]) / 2.0),
        "eer_threshold_discrete": float(roc_thresholds[eer_index]),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "genuine_trials": int(np.sum(y_true == 1)),
        "impostor_trials": int(np.sum(y_true == 0)),
        "predictions": pred,
    }


def main():
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit directory: {OUTPUT_ROOT}")

    enrollment, xe, hog_size = load_split("enrollment")
    validation, xv, _ = load_split("validation")
    test, xt, _ = load_split("test")
    ids = set(enrollment.identity.astype(str))
    if ids != set(validation.identity.astype(str)) or ids != set(test.identity.astype(str)):
        raise ValueError("Enrollment, validation, and test identities do not match")
    if not (
        set(zip(enrollment.identity, enrollment.image_name)).isdisjoint(set(zip(validation.identity, validation.image_name)))
        and set(zip(enrollment.identity, enrollment.image_name)).isdisjoint(set(zip(test.identity, test.image_name)))
        and set(zip(validation.identity, validation.image_name)).isdisjoint(set(zip(test.identity, test.image_name)))
    ):
        raise ValueError("Image-level split leakage detected")

    xb_train, y_train = training_block_trials(enrollment, xe, hog_size)
    xb_val, y_val, meta_val = evaluation_trials(enrollment, xe, validation, xv, hog_size)
    xb_test, y_test, meta_test = evaluation_trials(enrollment, xe, test, xt, hog_size)
    if (len(y_train), int(y_train.sum()), int((y_train == 0).sum())) != (3600, 120, 3480):
        raise ValueError("Unexpected enrollment training trial counts")
    if (len(y_val), len(y_test)) != (1800, 1800):
        raise ValueError("Unexpected validation/test trial counts")

    models = {
        "SVM": joblib.load(SAVED_MODELS / "svm_fingerprint.joblib")["model"],
        "RandomForest": joblib.load(SAVED_MODELS / "randomforest_fingerprint.joblib")["model"],
        "BalancedBlocksLinearSVM": Pipeline([
            ("scale", StandardScaler()),
            ("classifier", LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_STATE, max_iter=20000)),
        ]),
        "BalancedBlocksLogisticRegression": Pipeline([
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(C=1.0, class_weight="balanced", random_state=RANDOM_STATE, max_iter=2000)),
        ]),
    }

    rows, validation_rows, score_frames = [], [], {}
    for name, model in models.items():
        if name.startswith("BalancedBlocks"):
            model.fit(xb_train, y_train)
            val_scores = model.decision_function(xb_val)
            test_scores = model.decision_function(xb_test)
        else:
            # Existing baseline estimators are already fitted on the original
            # 8,110-dimensional enrollment pair trials; reuse without refitting.
            raw_train, raw_y, _ = original_training_trials(enrollment, xe)
            if not hasattr(model, "classes_"):
                model.fit(raw_train, raw_y)
            raw_val, raw_yv, _ = original_evaluation_trials(enrollment, xe, validation, xv)
            raw_test, raw_yt, _ = original_evaluation_trials(enrollment, xe, test, xt)
            val_scores = model.predict_proba(raw_val)[:, list(model.classes_).index(1)]
            test_scores = model.predict_proba(raw_test)[:, list(model.classes_).index(1)]
            if not np.array_equal(raw_yv, y_val) or not np.array_equal(raw_yt, y_test):
                raise ValueError("Baseline trial labels do not align with controlled trial labels")

        threshold, val_far, val_frr = select_threshold(y_val, val_scores)
        val_metrics = compute_metrics(y_val, val_scores, threshold)
        test_metrics = compute_metrics(y_test, test_scores, threshold)
        row = {k: v for k, v in test_metrics.items() if k != "predictions"}
        row.update({
            "model": name,
            "feature_representation": "original_8110D" if not name.startswith("BalancedBlocks") else "mean_HOG_abs_distance_plus_LBP_chi_square",
            "target_far": TARGET_FAR,
            "validation_far": val_far,
            "validation_frr": val_frr,
            "validation_roc_auc": float(roc_auc_score(y_val, val_scores)),
            "validation_genuine_trials": int(y_val.sum()),
            "validation_impostor_trials": int((y_val == 0).sum()),
        })
        rows.append(row)
        validation_rows.append({
            "model": name, "threshold": threshold, "far": val_metrics["far"],
            "frr": val_metrics["frr"], "roc_auc": val_metrics["roc_auc"],
            "tn": val_metrics["tn"], "fp": val_metrics["fp"],
            "fn": val_metrics["fn"], "tp": val_metrics["tp"],
        })
        score_frames[name] = (val_scores, test_scores, test_metrics["predictions"])

    comparison = pd.DataFrame(rows)
    best_validation = min(
        validation_rows,
        key=lambda row: (row["frr"], row["far"], -row["roc_auc"]),
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    comparison.to_csv(OUTPUT_ROOT / "fingerprint_controlled_model_comparison.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(OUTPUT_ROOT / "validation_threshold_metrics.csv", index=False)

    # Save score tables and database-conditioned score summaries for auditability.
    distribution_rows = []
    for name, (val_scores, test_scores, test_pred) in score_frames.items():
        threshold = float(comparison.loc[comparison.model == name, "threshold"].iloc[0])
        trial_file = meta_test.copy()
        trial_file["true_label"] = y_test
        trial_file["score"] = test_scores
        trial_file["prediction"] = test_pred
        trial_file.to_csv(OUTPUT_ROOT / f"{name.lower()}_test_trial_results.csv", index=False)
        for split_name, scores, labels, metadata in (
            ("validation", val_scores, y_val, meta_val),
            ("test", test_scores, y_test, meta_test),
        ):
            for group_label, mask in (("genuine", labels == 1), ("impostor", labels == 0)):
                vals = scores[mask]
                distribution_rows.append({
                    "model": name, "split": split_name, "class": group_label,
                    "count": len(vals), "mean": float(vals.mean()),
                    "std": float(vals.std()), "min": float(vals.min()),
                    "q05": float(np.quantile(vals, .05)), "median": float(np.median(vals)),
                    "q95": float(np.quantile(vals, .95)), "max": float(vals.max()),
                    "accept_rate_at_frozen_validation_threshold": float(np.mean(vals >= threshold)),
                })
        db = meta_test.probe_database.to_numpy()
        claim_db = meta_test.claimed_database.to_numpy()
        for label, mask in (
            ("same_database_impostor", (y_test == 0) & (db == claim_db)),
            ("cross_database_impostor", (y_test == 0) & (db != claim_db)),
        ):
            vals = test_scores[mask]
            distribution_rows.append({
                "model": name, "split": "test", "class": label,
                "count": len(vals), "mean": float(vals.mean()),
                "std": float(vals.std()), "min": float(vals.min()),
                "q05": float(np.quantile(vals, .05)), "median": float(np.median(vals)),
                "q95": float(np.quantile(vals, .95)), "max": float(vals.max()),
                "accept_rate_at_frozen_validation_threshold": float(np.mean(vals >= threshold)),
            })
    pd.DataFrame(distribution_rows).to_csv(OUTPUT_ROOT / "score_distribution_summary.csv", index=False)

    # ROC and FAR-FRR plots are test-set descriptions only; thresholds remain frozen.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cm_fig, cm_axes = plt.subplots(2, 2, figsize=(10, 9))
    for (name, (val_scores, test_scores, test_pred)), ax in zip(score_frames.items(), cm_axes.ravel()):
        fpr, tpr, _ = roc_curve(y_test, test_scores)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, test_scores):.3f})")
        axes[1].plot(fpr, 1.0 - tpr, label=name)
        ConfusionMatrixDisplay.from_predictions(
            y_test, test_pred, labels=[0, 1], display_labels=["Impostor", "Genuine"],
            cmap="Blues", values_format="d", ax=ax,
        )
        ax.set_title(name)
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="Fingerprint ROC — Session 3")
    axes[1].set(xlabel="FAR", ylabel="FRR", title="Fingerprint FAR–FRR — Session 3")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "fingerprint_roc_far_frr.png", dpi=180)
    plt.close(fig)
    cm_fig.tight_layout()
    cm_fig.savefig(OUTPUT_ROOT / "fingerprint_confusion_matrices.png", dpi=180)
    plt.close(cm_fig)

    # Class-conditional score density using histograms; keeps the score scale
    # explicit because probability and decision-function models use different ranges.
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (name, (_, test_scores, _)) in zip(axes.ravel(), score_frames.items()):
        ax.hist(test_scores[y_test == 0], bins=35, alpha=.65, density=True, label="Impostor")
        ax.hist(test_scores[y_test == 1], bins=25, alpha=.65, density=True, label="Genuine")
        ax.set_title(name)
        ax.set_xlabel("Score (higher = stronger match)")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "genuine_impostor_score_distributions.png", dpi=180)
    plt.close(fig)

    manifest = {
        "experiment": "Controlled real-fingerprint audit; separate from legacy outputs",
        "dataset": "DB1_B + DB2_B + DB3_B only (captured FVC2002 Set B)",
        "synthetic_DB4_B_included": False,
        "enrollment_validation_test_impressions": {"enrollment": [1, 2, 3, 4], "validation": [5, 6], "test": [7, 8]},
        "identities": len(ids), "feature_dimensions": {"HOG": 8100, "LBP": 10, "original_total": 8110},
        "training_trials": {"total": len(y_train), "genuine": int(y_train.sum()), "impostor": int((y_train == 0).sum())},
        "validation_trials": {"total": len(y_val), "genuine": int(y_val.sum()), "impostor": int((y_val == 0).sum())},
        "test_trials": {"total": len(y_test), "genuine": int(y_test.sum()), "impostor": int((y_test == 0).sum())},
        "target_far": TARGET_FAR,
        "threshold_selection": "validation scores only; threshold frozen before Session 3 evaluation",
        "block_balanced_representation": "mean absolute HOG distance plus LBP chi-square distance",
        "candidate_selection": f"lowest validation FRR at FAR <= {TARGET_FAR}; ties lower FAR then higher validation ROC-AUC",
        "selected_validation_model": best_validation["model"],
        "legacy_outputs_overwritten": False,
        "random_state": RANDOM_STATE,
    }
    (OUTPUT_ROOT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(comparison[["model", "threshold", "validation_far", "validation_frr", "accuracy", "precision", "recall", "f1", "far", "frr", "roc_auc", "eer_discrete", "tn", "fp", "fn", "tp"]].to_string(index=False))
    print("Selected by validation only:", best_validation["model"])
    print("Outputs:", OUTPUT_ROOT)


def original_training_trials(frame, values):
    identities, templates, indices = make_templates(frame, values)
    features, labels = [], []
    for identity in identities:
        for query_idx in indices[identity]:
            for claim in identities:
                reference = values[[i for i in indices[identity] if i != query_idx]].mean(axis=0) if claim == identity else templates[claim]
                features.append(np.abs(values[query_idx] - reference))
                labels.append(int(claim == identity))
    return np.asarray(features, np.float32), np.asarray(labels, np.int8), None


def original_evaluation_trials(enrollment_frame, enrollment_values, probes, probe_values):
    identities, templates, _ = make_templates(enrollment_frame, enrollment_values)
    features, labels = [], []
    for probe_idx, row in probes.iterrows():
        for claim in identities:
            features.append(np.abs(probe_values[probe_idx] - templates[claim]))
            labels.append(int(str(row.identity) == claim))
    return np.asarray(features, np.float32), np.asarray(labels, np.int8), None


if __name__ == "__main__":
    main()
