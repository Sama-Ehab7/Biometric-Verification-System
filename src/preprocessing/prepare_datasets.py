
from pathlib import Path

import pandas as pd


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FACE_FILE = (
    PROJECT_ROOT / "data" / "processed" / "face" / "face_features.csv"
)

VOICE_FILE = (
    PROJECT_ROOT / "data" / "processed" / "voice" / "voice_features.csv"
)

MAPPING_FILE = (
    PROJECT_ROOT / "data" / "metadata" / "session_mapping.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"


# ==========================================
# 2. LOAD DATASETS
# ==========================================

def load_datasets():

    face_df = pd.read_csv(FACE_FILE)
    voice_df = pd.read_csv(VOICE_FILE)
    mapping_df = pd.read_csv(MAPPING_FILE)

    return face_df, voice_df, mapping_df


# ==========================================
# 3. PREPARE ALIGNED DATASETS
# ==========================================

def prepare_aligned_datasets(face_df, voice_df, mapping_df):

    keys = ["speaker", "sequence"]

    # Prefix modality feature names to avoid collisions
    face_features = [
        col for col in face_df.columns
        if col.startswith("face_feature_")
    ]

    voice_features = [
        col for col in voice_df.columns
        if col.startswith(("mfcc_", "zcr_", "rms_"))
    ]

    if len(face_features) != 3548:
        raise ValueError(
            f"Expected 3548 Face features, found {len(face_features)}"
        )

    if len(voice_features) != 30:
        raise ValueError(
            f"Expected 30 Voice features, found {len(voice_features)}"
        )

    face_data = face_df[keys + face_features].copy()
    voice_data = voice_df[keys + voice_features].copy()

    # Ensure one row per speaker and sequence
    for name, df in [
        ("Face", face_data),
        ("Voice", voice_data),
        ("Mapping", mapping_df),
    ]:
        if df.duplicated(keys).any():
            raise ValueError(f"Duplicate records found in {name} dataset.")

    # Merge both modalities using exact identity and sequence keys
    multimodal_data = face_data.merge(
        voice_data,
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    # Attach session information
    multimodal_data = multimodal_data.merge(
        mapping_df[keys + ["session"]],
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    # Attach session information to unimodal datasets
    face_data = face_data.merge(
        mapping_df[keys + ["session"]],
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    voice_data = voice_data.merge(
        mapping_df[keys + ["session"]],
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    if not (
        len(face_data) == len(voice_data) == len(multimodal_data) == 430
    ):
        raise ValueError("Dataset alignment failed: expected 430 records.")

    return face_data, voice_data, multimodal_data


# ==========================================
# 4. SPLIT BY SESSION
# ==========================================

def split_dataset(df):

    train_df = df[df["session"] == "session_1"].copy()

    test_df = df[
        df["session"].isin(["session_2", "session_3"])
    ].copy()

    if len(train_df) != 258:
        raise ValueError(
            f"Expected 258 training records, found {len(train_df)}"
        )

    if len(test_df) != 172:
        raise ValueError(
            f"Expected 172 testing records, found {len(test_df)}"
        )

    return train_df, test_df


# ==========================================
# 5. SAVE DATASETS
# ==========================================

def save_dataset(name, train_df, test_df):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = OUTPUT_DIR / f"{name}_train.csv"
    test_path = OUTPUT_DIR / f"{name}_test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\n{name.upper()} DATASET")
    print("Training:", train_df.shape)
    print("Testing:", test_df.shape)
    print("Saved:", train_path)
    print("Saved:", test_path)


# ==========================================
# 6. MAIN
# ==========================================

def main():

    print("=" * 55)
    print("PREPARING BIOMETRIC DATASETS")
    print("=" * 55)

    face_df, voice_df, mapping_df = load_datasets()

    face_data, voice_data, multimodal_data = (
        prepare_aligned_datasets(
            face_df,
            voice_df,
            mapping_df,
        )
    )

    datasets = {
        "face": face_data,
        "voice": voice_data,
        "multimodal": multimodal_data,
    }

    for name, df in datasets.items():

        train_df, test_df = split_dataset(df)

        save_dataset(name, train_df, test_df)

    print("\n========== FINAL SUMMARY ==========")
    print("Total recordings: 430")
    print("Training recordings: 258")
    print("Testing recordings: 172")
    print("Face features: 3548")
    print("Voice features: 30")
    print("Multimodal features: 3578")
    print("All datasets prepared successfully.")
    print("===================================")


if __name__ == "__main__":
    main()

