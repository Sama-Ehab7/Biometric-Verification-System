
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"

FACE_MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "face"
    / "svm_face_security.joblib"
)

VOICE_MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "voice"
    / "svm_voice_security.joblib"
)

# The validated _audit policy is the sole policy source for this sequential
# run. Legacy canonical artifacts are retained as historical outputs.
AUTHORITATIVE_POLICY_PATH = (
    PROJECT_ROOT / "saved_models" / "hierarchical"
    / "hierarchical_policy_audit.joblib"
)
AUTHORITATIVE_THRESHOLDS_PATH = (
    PROJECT_ROOT / "results" / "hierarchical"
    / "selected_thresholds_audit.json"
)
AUTHORITATIVE_COMPARISON_PATH = (
    PROJECT_ROOT / "results" / "hierarchical"
    / "hierarchical_comparison_audit.csv"
)
AUTHORITATIVE_TRIALS_PATH = (
    PROJECT_ROOT / "results" / "hierarchical"
    / "hierarchical_trial_decisions_audit.csv"
)

LEGACY_ARTIFACTS = [
    "results/hierarchical/face_coverage_experiment.csv",
    "results/hierarchical/selected_thresholds.json",
    "results/hierarchical/hierarchical_comparison.csv",
    "results/hierarchical/hierarchical_stage_summary.json",
    "results/hierarchical/hierarchical_trial_decisions.csv",
    "results/hierarchical/face_confusion_matrix.png",
    "results/hierarchical/voice_confusion_matrix.png",
    "results/hierarchical/hierarchical_confusion_matrix.png",
    "results/hierarchical/face_roc_curve.png",
    "saved_models/hierarchical/hierarchical_policy.joblib",
]

VALIDATED_AUDIT_ARTIFACTS = [
    "results/hierarchical/threshold_search_audit.csv",
    "results/hierarchical/threshold_candidates_audit.csv",
    "results/hierarchical/selected_thresholds_audit.json",
    "results/hierarchical/session2_score_distributions_audit.csv",
    "results/hierarchical/session2_trial_scores_audit.csv",
    "results/hierarchical/session2_escalation_analysis_audit.csv",
    "results/hierarchical/hierarchical_comparison_audit.csv",
    "results/hierarchical/hierarchical_stage_summary_audit.json",
    "results/hierarchical/hierarchical_trial_decisions_audit.csv",
    "results/hierarchical/face_confusion_matrix_audit.png",
    "results/hierarchical/voice_confusion_matrix_audit.png",
    "results/hierarchical/hierarchical_confusion_matrix_audit.png",
    "results/hierarchical/face_roc_curve_audit.png",
    "saved_models/hierarchical/hierarchical_policy_audit.joblib",
]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "hierarchical"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "saved_models"
    / "hierarchical"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FAR = 0.01

N_THRESHOLD_CANDIDATES = 31

FACE_REFERENCE_THRESHOLD = 0.290093
VOICE_REFERENCE_THRESHOLD = 0.816066


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_data():

    print("\nLoading Face and Voice datasets...")

    face_train = pd.read_csv(
        SPLIT_DIR / "face_train.csv"
    )

    face_test = pd.read_csv(
        SPLIT_DIR / "face_test.csv"
    )

    voice_train = pd.read_csv(
        SPLIT_DIR / "voice_train.csv"
    )

    voice_test = pd.read_csv(
        SPLIT_DIR / "voice_test.csv"
    )

    face_features = [
        col
        for col in face_train.columns
        if col.startswith("face_feature_")
    ]

    voice_features = [
        col
        for col in voice_train.columns
        if (
            col.startswith("mfcc_")
            or col.startswith("zcr_")
            or col.startswith("rms_")
        )
    ]

    if len(face_features) != 3548:

        raise ValueError(
            f"Expected 3548 Face features, "
            f"found {len(face_features)}"
        )

    if len(voice_features) != 30:

        raise ValueError(
            f"Expected 30 Voice features, "
            f"found {len(voice_features)}"
        )

    required_columns = [
        "speaker",
        "sequence",
        "session"
    ]

    for name, df in [
        ("Face train", face_train),
        ("Face test", face_test),
        ("Voice train", voice_train),
        ("Voice test", voice_test),
    ]:

        missing = [
            col
            for col in required_columns
            if col not in df.columns
        ]

        if missing:

            raise ValueError(
                f"{name} is missing columns: {missing}"
            )

    print(f"Face features: {len(face_features)}")
    print(f"Voice features: {len(voice_features)}")

    print(
        f"Session 1 Face enrollment: {len(face_train)}"
    )

    print(
        f"Session 1 Voice enrollment: {len(voice_train)}"
    )

    for session in ["session_2", "session_3"]:

        face_count = len(
            face_test[
                face_test["session"] == session
            ]
        )

        voice_count = len(
            voice_test[
                voice_test["session"] == session
            ]
        )

        print(
            f"{session}: Face={face_count}, "
            f"Voice={voice_count}"
        )

    return (
        face_train,
        face_test,
        voice_train,
        voice_test,
        face_features,
        voice_features,
    )


# ============================================================
# 3. LOAD TRAINED MODELS
# ============================================================

def load_model(path, model_name):

    print(f"\nLoading {model_name}...")

    if not path.exists():

        raise FileNotFoundError(
            f"Model file not found: {path}"
        )

    artifact = joblib.load(path)

    if isinstance(artifact, dict):

        model = artifact.get("model")

    else:

        model = artifact

    if model is None:

        raise ValueError(
            f"{model_name} was not found in artifact."
        )

    if not hasattr(model, "predict_proba"):

        raise ValueError(
            f"{model_name} does not support predict_proba."
        )

    return model


# ============================================================
# 4. BUILD ENROLLMENT CENTROIDS
# ============================================================

def build_centroids(train_df, feature_columns):

    centroids = {}

    for speaker, group in train_df.groupby("speaker"):

        features = group[
            feature_columns
        ].to_numpy(dtype=np.float32)

        centroids[str(speaker)] = features.mean(
            axis=0
        )

    return centroids


# ============================================================
# 5. GET GENUINE PROBABILITY
# ============================================================

def get_genuine_probability(model, X):

    probabilities = model.predict_proba(X)

    classes = list(model.classes_)

    if 1 not in classes:

        raise ValueError(
            f"Genuine class 1 not found. "
            f"Classes: {classes}"
        )

    genuine_index = classes.index(1)

    return probabilities[:, genuine_index]


# ============================================================
# 6. GENERATE VERIFICATION SCORES
# ============================================================

def generate_scores(
    probes,
    feature_columns,
    centroids,
    model,
):

    enrolled_speakers = sorted(
        centroids.keys()
    )

    records = []

    for _, probe in probes.iterrows():

        # Identity metadata is retained for claim pairing; this is not used
        # as a genuine/impostor label during inference.
        actual_speaker = str(probe["speaker"])

        probe_features = probe[
            feature_columns
        ].to_numpy(dtype=np.float32)

        pair_features = np.vstack([

            np.abs(
                probe_features
                - centroids[claimed_speaker]
            )

            for claimed_speaker in enrolled_speakers

        ])

        scores = get_genuine_probability(
            model,
            pair_features
        )

        for claimed_speaker, score in zip(
            enrolled_speakers,
            scores
        ):

            records.append({

                "probe_speaker": actual_speaker,

                "sequence": probe["sequence"],

                "session": probe["session"],

                "claimed_speaker": claimed_speaker,

                "face_score": float(score),

            })

    return pd.DataFrame(records)


# ============================================================
# 7. BUILD FACE + VOICE TRIALS
# ============================================================

def build_face_trials(
    face_test,
    voice_test,
    face_features,
    face_centroids,
    face_model,
    session_name,
):

    face_probes = face_test[
        face_test["session"] == session_name
    ].copy()

    voice_probes = voice_test[
        voice_test["session"] == session_name
    ].copy()

    face_keys = set(
        zip(
            face_probes["speaker"],
            face_probes["sequence"]
        )
    )

    voice_keys = set(
        zip(
            voice_probes["speaker"],
            voice_probes["sequence"]
        )
    )

    if face_keys != voice_keys:

        raise ValueError(
            f"Face/Voice alignment failed for {session_name}"
        )

    face_scores = generate_scores(
        face_probes,
        face_features,
        face_centroids,
        face_model,
    )

    expected = (
        len(face_probes)
        * len(face_centroids)
    )

    if len(face_scores) != expected:

        raise ValueError(
            f"Expected {expected} trials, "
            f"got {len(face_scores)}"
        )

    return face_scores, face_probes, voice_probes


def score_escalated_voice_trials(
    escalated_trials,
    voice_probes,
    voice_features,
    voice_centroids,
    voice_model,
):
    """Score only the trial rows selected by Face for Voice escalation."""
    if escalated_trials.empty:
        return pd.Series(dtype=np.float64, index=escalated_trials.index)

    probe_lookup = {
        (str(row.speaker), str(row.sequence), str(row.session)): row
        for row in voice_probes.itertuples(index=False)
    }

    pair_features = []
    for row in escalated_trials.itertuples(index=True):
        probe_key = (
            str(row.probe_speaker),
            str(row.sequence),
            str(row.session),
        )
        probe = probe_lookup.get(probe_key)
        claimed_speaker = str(row.claimed_speaker)
        if probe is None:
            raise ValueError(f"No aligned Voice probe for {probe_key}")
        if claimed_speaker not in voice_centroids:
            raise ValueError(f"Unenrolled claimed speaker: {claimed_speaker}")

        probe_features = np.asarray(
            [getattr(probe, name) for name in voice_features],
            dtype=np.float32,
        )
        pair_features.append(
            np.abs(probe_features - voice_centroids[claimed_speaker])
        )

    scores = get_genuine_probability(
        voice_model,
        np.asarray(pair_features, dtype=np.float32),
    )
    return pd.Series(scores, index=escalated_trials.index, dtype=np.float64)


def apply_sequential_policy(
    face_trials,
    voice_probes,
    voice_features,
    voice_centroids,
    voice_model,
    low_face_threshold,
    high_face_threshold,
    voice_threshold,
):
    """Make Face decisions first; invoke Voice only for uncertain trials."""
    result = face_trials.copy()
    face_scores = result["face_score"].to_numpy(dtype=np.float64)
    direct_accept = face_scores >= high_face_threshold
    direct_reject = face_scores <= low_face_threshold
    escalate = ~(direct_accept | direct_reject)

    result["decision"] = np.zeros(len(result), dtype=np.int8)
    result.loc[direct_accept, "decision"] = 1
    result["stage"] = np.where(
        direct_accept,
        "face_accept",
        np.where(direct_reject, "face_reject", "voice_escalation"),
    )
    # Direct Face decisions never receive a Voice score or call.
    result["voice_score"] = np.nan

    escalated_trials = result.loc[escalate]
    voice_scores = score_escalated_voice_trials(
        escalated_trials,
        voice_probes,
        voice_features,
        voice_centroids,
        voice_model,
    )
    if len(voice_scores) != int(escalate.sum()):
        raise ValueError("Voice inference count differs from Face escalation count")
    result.loc[voice_scores.index, "voice_score"] = voice_scores
    result.loc[voice_scores.index, "decision"] = (
        voice_scores.to_numpy() >= voice_threshold
    ).astype(np.int8)

    if not np.isin(result["decision"].to_numpy(), [0, 1]).all():
        raise ValueError("Not every trial received one binary decision")
    return result, int(len(voice_scores))


def attach_evaluation_labels(trials):
    """Attach genuine/impostor labels after runtime decisions are complete."""
    result = trials.copy()
    result["y_true"] = (
        result["probe_speaker"].astype(str)
        == result["claimed_speaker"].astype(str)
    ).astype(np.int8)
    return result


# ============================================================
# 8. CALCULATE METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    scores=None
):

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

    metrics = {

        "Accuracy": float(
            accuracy_score(y_true, y_pred)
        ),

        "Precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "Recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "F1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "FAR": float(far),

        "FRR": float(frr),

        "TN": int(tn),

        "FP": int(fp),

        "FN": int(fn),

        "TP": int(tp),

    }

    if (
        scores is not None
        and len(np.unique(y_true)) == 2
    ):

        metrics["ROC-AUC"] = float(
            roc_auc_score(y_true, scores)
        )

    else:

        metrics["ROC-AUC"] = np.nan

    return metrics


# ============================================================
# 9. SAVE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    y_true,
    y_pred,
    title,
    output_path,
):

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=[0, 1],
        display_labels=[
            "Impostor",
            "Genuine"
        ],
        cmap="Blues",
        values_format="d",
        ax=ax
    )

    ax.set_title(title)

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 10. SAVE ROC CURVE
# ============================================================

def save_roc_curve(
    y_true,
    scores,
    output_path,
):

    fpr, tpr, _ = roc_curve(
        y_true,
        scores
    )

    auc = roc_auc_score(
        y_true,
        scores
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    ax.plot(
        fpr,
        tpr,
        label=f"Face SVM (AUC={auc:.4f})"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random"
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "Face SVM ROC Curve - Session 3"
    )

    ax.legend()

    ax.grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 11. MAIN FUNCTION
# ============================================================

def main():
    print("=" * 70)
    print("HIERARCHICAL FACE-THEN-VOICE VERIFICATION")
    print("=" * 70)

    # Preserve all previous artifacts. This run writes only fresh _sequential files.
    output_paths = [
        RESULTS_DIR / "session2_validation_sequential.csv",
        RESULTS_DIR / "hierarchical_trial_decisions_sequential.csv",
        RESULTS_DIR / "hierarchical_comparison_sequential.csv",
        RESULTS_DIR / "hierarchical_stage_summary_sequential.json",
        RESULTS_DIR / "sequential_vs_audit_comparison.json",
        RESULTS_DIR / "hierarchical_run_manifest_sequential.json",
        RESULTS_DIR / "hierarchical_confusion_matrix_sequential.png",
        RESULTS_DIR / "face_roc_curve_sequential.png",
        MODEL_DIR / "hierarchical_policy_sequential.joblib",
    ]
    existing = [str(path) for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite sequential audit output(s): " + ", ".join(existing)
        )

    if not AUTHORITATIVE_POLICY_PATH.exists():
        raise FileNotFoundError(
            f"Authoritative policy missing: {AUTHORITATIVE_POLICY_PATH}"
        )
    if not AUTHORITATIVE_THRESHOLDS_PATH.exists():
        raise FileNotFoundError(
            f"Validated Session 2 threshold record missing: {AUTHORITATIVE_THRESHOLDS_PATH}"
        )
    if not AUTHORITATIVE_COMPARISON_PATH.exists() or not AUTHORITATIVE_TRIALS_PATH.exists():
        raise FileNotFoundError("Validated audit comparison or trial decisions are missing")

    # Load only the previously validated Session 2 policy. Do not recalibrate or retune.
    policy = joblib.load(AUTHORITATIVE_POLICY_PATH)
    validation_record = json.loads(
        AUTHORITATIVE_THRESHOLDS_PATH.read_text(encoding="utf-8")
    )
    if policy.get("threshold_selection_session") != "session_2":
        raise ValueError("Authoritative policy is not marked as Session 2 selected")
    if policy.get("optimization_status") != "FAR target satisfied":
        raise ValueError("Authoritative policy does not satisfy the stored FAR target")
    if validation_record.get("status") != "FAR target satisfied":
        raise ValueError("Stored validation record does not satisfy the FAR target")
    threshold_keys = (
        "low_face_threshold", "high_face_threshold", "voice_threshold"
    )
    for key in threshold_keys:
        if not np.isclose(
            float(policy[key]), float(validation_record[key]), rtol=0, atol=1e-12
        ):
            raise ValueError(f"Policy and validated threshold record disagree: {key}")

    print("Authoritative policy source:", AUTHORITATIVE_POLICY_PATH.relative_to(PROJECT_ROOT))
    print("Threshold selection provenance: Session 2")
    print(
        "Thresholds (Face low / Face high / Voice): "
        f"{policy['low_face_threshold']:.6f} / "
        f"{policy['high_face_threshold']:.6f} / "
        f"{policy['voice_threshold']:.6f}"
    )

    (
        face_train,
        face_test,
        voice_train,
        voice_test,
        face_features,
        voice_features,
    ) = load_data()

    if set(face_train["session"].astype(str).str.lower().str.strip()) != {"session_1"}:
        raise ValueError("Enrollment inputs must contain Session 1 only")
    if set(voice_train["session"].astype(str).str.lower().str.strip()) != {"session_1"}:
        raise ValueError("Voice enrollment inputs must contain Session 1 only")

    face_model = load_model(FACE_MODEL_PATH, "Face SVM security model")
    voice_model = load_model(VOICE_MODEL_PATH, "Voice SVM security model")
    face_centroids = build_centroids(face_train, face_features)
    voice_centroids = build_centroids(voice_train, voice_features)
    if set(face_centroids) != set(voice_centroids) or len(face_centroids) != 43:
        raise ValueError("Face and Voice Session 1 enrollment identities do not match")

    session_outputs = {}
    for session_name in ("session_2", "session_3"):
        # Generate Face scores first. The Voice model is called only after Face
        # has identified the uncertain rows for this fixed policy.
        face_trials, face_probes, voice_probes = build_face_trials(
            face_test,
            voice_test,
            face_features,
            face_centroids,
            face_model,
            session_name,
        )
        decisions, voice_pair_inferences = apply_sequential_policy(
            face_trials,
            voice_probes,
            voice_features,
            voice_centroids,
            voice_model,
            float(policy["low_face_threshold"]),
            float(policy["high_face_threshold"]),
            float(policy["voice_threshold"]),
        )
        escalation_count = int((decisions["stage"] == "voice_escalation").sum())
        direct_count = int(len(decisions) - escalation_count)
        if voice_pair_inferences != escalation_count:
            raise ValueError(
                f"{session_name}: Voice scored {voice_pair_inferences} rows, "
                f"but Face escalated {escalation_count}"
            )
        if decisions.loc[decisions["stage"] != "voice_escalation", "voice_score"].notna().any():
            raise ValueError("A direct Face decision unexpectedly received Voice inference")
        if decisions["decision"].isna().any() or not decisions["decision"].isin([0, 1]).all():
            raise ValueError("Every trial must receive exactly one binary decision")
        if len(decisions) != len(face_probes) * len(face_centroids):
            raise ValueError(f"{session_name}: unexpected trial count")

        # Attach labels only after all runtime decisions are complete.
        evaluated = attach_evaluation_labels(decisions)
        session_outputs[session_name] = {
            "trials": evaluated,
            "voice_pair_inferences": voice_pair_inferences,
            "direct_face_decisions": direct_count,
            "voice_escalations": escalation_count,
        }
        print(
            f"{session_name}: {len(evaluated)} trials; "
            f"{direct_count} direct Face decisions; "
            f"{escalation_count} Voice inferences"
        )

    validation = session_outputs["session_2"]["trials"]
    test = session_outputs["session_3"]["trials"]
    for session_name, trials in (("session_2", validation), ("session_3", test)):
        genuine = int(trials["y_true"].sum())
        impostor = int((trials["y_true"] == 0).sum())
        if (len(trials), genuine, impostor) != (3698, 86, 3612):
            raise ValueError(
                f"{session_name}: expected 3698 trials (86 genuine, 3612 impostor), "
                f"got {len(trials)} ({genuine}, {impostor})"
            )

    # Verify Session 2 decisions and metrics against the preserved policy record.
    prior_validation = pd.read_csv(
        RESULTS_DIR / "session2_trial_scores_audit.csv"
    )
    trial_keys = ["probe_speaker", "sequence", "session", "claimed_speaker"]
    validation_check = validation[trial_keys + ["decision", "y_true"]].merge(
        prior_validation[trial_keys + ["decision", "y_true"]],
        on=trial_keys,
        how="outer",
        validate="one_to_one",
        suffixes=("_sequential", "_audit"),
        indicator=True,
    )
    if not (validation_check["_merge"] == "both").all():
        raise ValueError("Session 2 trial identities differ from the validated audit")
    if not (validation_check["decision_sequential"] == validation_check["decision_audit"]).all():
        raise ValueError("Sequential Session 2 decisions differ from the validated audit policy")
    if not (validation_check["y_true_sequential"] == validation_check["y_true_audit"]).all():
        raise ValueError("Session 2 genuine/impostor labels differ from the validated audit")

    y_val = validation["y_true"].to_numpy(dtype=np.int8)
    pred_val = validation["decision"].to_numpy(dtype=np.int8)
    validation_metrics = calculate_metrics(y_val, pred_val)
    if sum(validation_metrics[k] for k in ("TN", "FP", "FN", "TP")) != len(validation):
        raise ValueError("Session 2 confusion matrix does not sum to trial count")
    for key, metric_key in (
        ("validation_far", "FAR"),
        ("validation_frr", "FRR"),
        ("validation_accuracy", "Accuracy"),
        ("validation_precision", "Precision"),
        ("validation_recall", "Recall"),
        ("validation_f1", "F1"),
    ):
        if not np.isclose(validation_metrics[metric_key], validation_record[key], atol=1e-12):
            raise ValueError(f"Session 2 metric differs from validated policy record: {metric_key}")

    # Session 3 labels are now used for final evaluation only, after inference.
    y_test = test["y_true"].to_numpy(dtype=np.int8)
    pred_test = test["decision"].to_numpy(dtype=np.int8)
    test_metrics = calculate_metrics(y_test, pred_test)
    if sum(test_metrics[k] for k in ("TN", "FP", "FN", "TP")) != len(test):
        raise ValueError("Session 3 confusion matrix does not sum to 3,698")

    # Compare the new sequential result to the preserved audit run by trial key.
    prior_test = pd.read_csv(AUTHORITATIVE_TRIALS_PATH)
    comparison_keys = trial_keys
    prior_columns = comparison_keys + ["decision", "y_true", "voice_score", "stage"]
    current_columns = comparison_keys + ["decision", "y_true", "voice_score", "stage"]
    decision_check = test[current_columns].merge(
        prior_test[prior_columns],
        on=comparison_keys,
        how="outer",
        validate="one_to_one",
        suffixes=("_sequential", "_audit"),
        indicator=True,
    )
    if not (decision_check["_merge"] == "both").all():
        raise ValueError("Session 3 trial identities differ from the validated audit")
    decisions_identical = bool(
        (decision_check["decision_sequential"] == decision_check["decision_audit"]).all()
    )
    labels_identical = bool(
        (decision_check["y_true_sequential"] == decision_check["y_true_audit"]).all()
    )
    escalation_mask = decision_check["stage_sequential"] == "voice_escalation"
    voice_scores_within_tolerance = bool(np.allclose(
        decision_check.loc[escalation_mask, "voice_score_sequential"].to_numpy(dtype=float),
        decision_check.loc[escalation_mask, "voice_score_audit"].to_numpy(dtype=float),
        rtol=0,
        atol=1e-7,
    ))
    voice_score_max_delta = float(np.max(np.abs(
        decision_check.loc[escalation_mask, "voice_score_sequential"].to_numpy(dtype=float)
        - decision_check.loc[escalation_mask, "voice_score_audit"].to_numpy(dtype=float)
    ))) if escalation_mask.any() else 0.0
    print(
        "Audit comparison checks:",
        f"decisions={decisions_identical}",
        f"labels={labels_identical}",
        f"voice_scores_within_1e-7={voice_scores_within_tolerance}",
        f"max_voice_score_delta={voice_score_max_delta:.3g}",
    )
    if not (decisions_identical and labels_identical and voice_scores_within_tolerance):
        raise ValueError(
            "Sequential Session 3 output differs from the validated audit; "
            "refusing to publish it as equivalent"
        )

    previous_comparison = pd.read_csv(AUTHORITATIVE_COMPARISON_PATH)
    previous_hierarchical = previous_comparison[
        previous_comparison["System"] == "Hierarchical Face-Voice"
    ]
    if len(previous_hierarchical) != 1:
        raise ValueError("Expected one prior authoritative Hierarchical comparison row")
    previous_hierarchical = previous_hierarchical.iloc[0]
    sequential_row = {
        "System": "Hierarchical Face-Voice",
        "Accuracy": test_metrics["Accuracy"],
        "Precision": test_metrics["Precision"],
        "Recall": test_metrics["Recall"],
        "F1": test_metrics["F1"],
        "FAR": test_metrics["FAR"],
        "FRR": test_metrics["FRR"],
        "TN": test_metrics["TN"],
        "FP": test_metrics["FP"],
        "FN": test_metrics["FN"],
        "TP": test_metrics["TP"],
        "ROC-AUC": "N/A (binary decisions)",
    }
    for key in ("TN", "FP", "FN", "TP"):
        if int(sequential_row[key]) != int(previous_hierarchical[key]):
            raise ValueError(f"Session 3 confusion count changed: {key}")
    comparison = previous_comparison[
        previous_comparison["System"] != "Hierarchical Face-Voice"
    ].copy()
    comparison = pd.concat([comparison, pd.DataFrame([sequential_row])], ignore_index=True)
    if not (comparison[["TN", "FP", "FN", "TP"]].sum(axis=1) == 3698).all():
        raise ValueError("A comparison confusion matrix does not sum to 3,698")

    stage_rows = []
    for session_name, trials in (("session_2", validation), ("session_3", test)):
        escalated = trials["stage"] == "voice_escalation"
        genuine = trials["y_true"] == 1
        impostor = ~genuine
        voice_accept = trials["decision"] == 1
        stage_rows.append({
            "session": session_name,
            "total_trials": len(trials),
            "direct_face_accept": int((trials["stage"] == "face_accept").sum()),
            "direct_face_reject": int((trials["stage"] == "face_reject").sum()),
            "voice_escalations": int(escalated.sum()),
            "voice_pair_inferences": session_outputs[session_name]["voice_pair_inferences"],
            "genuine_escalations": int((escalated & genuine).sum()),
            "impostor_escalations": int((escalated & impostor).sum()),
            "genuine_voice_accepts": int((escalated & genuine & voice_accept).sum()),
            "impostor_voice_accepts": int((escalated & impostor & voice_accept).sum()),
            "genuine_voice_rejects": int((escalated & genuine & ~voice_accept).sum()),
            "impostor_voice_rejects": int((escalated & impostor & ~voice_accept).sum()),
        })
    stage_summary = {
        "policy_source": str(AUTHORITATIVE_POLICY_PATH.relative_to(PROJECT_ROOT)),
        "threshold_selection_session": "session_2",
        "final_evaluation_session": "session_3",
        "sessions": stage_rows,
    }

    sequential_policy = dict(policy)
    sequential_policy["inference_mode"] = "strict_face_then_voice_escalations_only"
    sequential_policy["policy_source"] = str(
        AUTHORITATIVE_POLICY_PATH.relative_to(PROJECT_ROOT)
    )
    policy_output = MODEL_DIR / "hierarchical_policy_sequential.joblib"

    comparison_diff = {
        "session3_decisions_identical_to_validated_audit": decisions_identical,
        "session3_labels_identical_to_validated_audit": labels_identical,
        "escalated_voice_scores_within_1e-7_of_validated_audit": voice_scores_within_tolerance,
        "max_escalated_voice_score_delta": voice_score_max_delta,
        "session3_trial_count": len(test),
        "session3_confusion_matrix": {
            key: int(test_metrics[key]) for key in ("TN", "FP", "FN", "TP")
        },
        "metric_differences_vs_validated_audit": {
            key: float(test_metrics[key] - float(previous_hierarchical[key]))
            for key in ("Accuracy", "Precision", "Recall", "F1", "FAR", "FRR")
        },
    }
    manifest = {
        "status": "sequential run; validated policy preserved",
        "authoritative_policy_input": str(
            AUTHORITATIVE_POLICY_PATH.relative_to(PROJECT_ROOT)
        ),
        "authoritative_session2_threshold_record": str(
            AUTHORITATIVE_THRESHOLDS_PATH.relative_to(PROJECT_ROOT)
        ),
        "previous_validated_audit_artifacts_preserved": VALIDATED_AUDIT_ARTIFACTS,
        "legacy_artifacts_preserved": LEGACY_ARTIFACTS,
        "new_sequential_artifacts": [
            str(path.relative_to(PROJECT_ROOT)) for path in output_paths
        ],
        "comparison": comparison_diff,
    }

    # Write only new, suffix-specific outputs; no canonical or _audit artifact is replaced.
    validation.to_csv(output_paths[0], index=False)
    test.to_csv(output_paths[1], index=False)
    comparison.to_csv(output_paths[2], index=False)
    output_paths[3].write_text(json.dumps(stage_summary, indent=2), encoding="utf-8")
    output_paths[4].write_text(json.dumps(comparison_diff, indent=2), encoding="utf-8")
    save_confusion_matrix(
        y_test,
        pred_test,
        "Sequential Hierarchical Face-Voice - Session 3",
        output_paths[6],
    )
    save_roc_curve(
        y_test,
        test["face_score"].to_numpy(dtype=float),
        output_paths[7],
    )
    joblib.dump(sequential_policy, policy_output)
    output_paths[5].write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Session 2 validation:", validation_metrics)
    print("Session 3 final metrics:", test_metrics)
    print("Session 3 confusion matrix sums:", sum(test_metrics[k] for k in ("TN", "FP", "FN", "TP")))
    print("Session 3 decisions identical to validated audit:", decisions_identical)
    print(
        "Escalated Voice scores within 1e-7 of validated audit:",
        voice_scores_within_tolerance,
        f"(max delta {voice_score_max_delta:.3g})",
    )
    print("Comparison saved:", output_paths[2].relative_to(PROJECT_ROOT))
    print("Policy saved:", policy_output.relative_to(PROJECT_ROOT))


# ============================================================
# 12. RUN
# ============================================================

if __name__ == "__main__":

    main()
