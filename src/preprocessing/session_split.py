
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARCHIVES_PATH = PROJECT_ROOT / "data" / "raw" / "archives"

FACE_FEATURES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "face" / "face_features.csv"
)

VOICE_FEATURES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "voice" / "voice_features.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "metadata" / "session_mapping.csv"
)


# ==========================================
# 2. SESSION ASSIGNMENT
# ==========================================

def get_session(index):
    if index < 6:
        return "session_1"
    elif index < 8:
        return "session_2"
    return "session_3"


# ==========================================
# 3. BUILD SESSION MAPPING
# ==========================================

def build_session_mapping():

    archives = sorted(ARCHIVES_PATH.glob("*.zip"))

    if len(archives) != 43:
        raise ValueError(
            f"Expected 43 ZIP archives, found {len(archives)}"
        )

    records = []

    for archive_path in archives:

        speaker = archive_path.stem

        with ZipFile(archive_path, "r") as archive:

            audio_files = [
                name for name in archive.namelist()
                if name.lower().endswith(".wav")
            ]

        if len(audio_files) != 10:
            raise ValueError(
                f"{speaker}: Expected 10 recordings, "
                f"found {len(audio_files)}"
            )

        for index, audio_path in enumerate(audio_files):

            sequence = Path(audio_path).stem

            records.append({
                "speaker": speaker,
                "sequence": sequence,
                "session": get_session(index),
                "session_recording_index": index + 1,
            })

    return pd.DataFrame(records)


# ==========================================
# 4. VALIDATE FEATURE ALIGNMENT
# ==========================================

def validate_alignment(mapping_df):

    face_df = pd.read_csv(FACE_FEATURES_PATH)
    voice_df = pd.read_csv(VOICE_FEATURES_PATH)

    def make_keys(df):
        return set(
            zip(df["speaker"].astype(str), df["sequence"].astype(str))
        )

    mapping_keys = make_keys(mapping_df)
    face_keys = make_keys(face_df)
    voice_keys = make_keys(voice_df)

    if len(mapping_keys) != 430:
        raise ValueError(
            f"Expected 430 unique mapping records, found {len(mapping_keys)}"
        )

    if mapping_keys != face_keys:
        raise ValueError("Session mapping does not match Face Features.")

    if mapping_keys != voice_keys:
        raise ValueError("Session mapping does not match Voice Features.")

    print("Mapping records:", len(mapping_keys))
    print("Face alignment:", mapping_keys == face_keys)
    print("Voice alignment:", mapping_keys == voice_keys)


# ==========================================
# 5. MAIN
# ==========================================

def main():

    print("=" * 55)
    print("VIDTIMIT SESSION MAPPING")
    print("=" * 55)

    mapping_df = build_session_mapping()

    validate_alignment(mapping_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(OUTPUT_PATH, index=False)

    print("\nSession Distribution:")
    print(mapping_df["session"].value_counts().sort_index())

    print("\nSpeakers per Session:")
    print(
        mapping_df.groupby("session")["speaker"]
        .nunique()
        .sort_index()
    )

    print("\nSaved:", OUTPUT_PATH)
    print("Dataset Shape:", mapping_df.shape)

    print("\nSession mapping completed successfully.")


if __name__ == "__main__":
    main()
