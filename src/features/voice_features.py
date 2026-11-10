
from pathlib import Path

import librosa
import numpy as np
import pandas as pd


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "vidtimit"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "voice"

OUTPUT_FILE = OUTPUT_DIR / "voice_features.csv"


# ==========================================
# 2. FEATURE EXTRACTION SETTINGS
# ==========================================

N_MFCC = 13
EXPECTED_SAMPLE_RATE = 32000


# ==========================================
# 3. EXTRACT FEATURES FROM ONE AUDIO FILE
# ==========================================

def extract_audio_features(audio_path):

    signal, sample_rate = librosa.load(
        audio_path,
        sr=None,
        mono=True
    )

    if len(signal) == 0:
        raise ValueError("Empty audio signal")

    if sample_rate != EXPECTED_SAMPLE_RATE:
        raise ValueError(
            f"Unexpected sample rate: {sample_rate}"
        )

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sample_rate,
        n_mfcc=N_MFCC
    )

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(signal)

    # Root Mean Square Energy
    rms = librosa.feature.rms(y=signal)

    features = {}

    # MFCC Mean and Standard Deviation
    for i in range(N_MFCC):

        features[f"mfcc_{i+1}_mean"] = float(
            mfcc[i].mean()
        )

        features[f"mfcc_{i+1}_std"] = float(
            mfcc[i].std()
        )

    # ZCR
    features["zcr_mean"] = float(zcr.mean())
    features["zcr_std"] = float(zcr.std())

    # RMS
    features["rms_mean"] = float(rms.mean())
    features["rms_std"] = float(rms.std())

    return features, sample_rate, len(signal)


# ==========================================
# 4. MAIN FEATURE EXTRACTION PIPELINE
# ==========================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    audio_files = sorted(
        DATASET_PATH.rglob("*.wav")
    )

    print("========== VOICE FEATURE EXTRACTION ==========")
    print(f"Total Audio Files: {len(audio_files)}")

    records = []
    failures = []

    for index, audio_path in enumerate(audio_files, start=1):

        relative_path = audio_path.relative_to(DATASET_PATH)

        # Speaker is the first folder under the dataset root
        speaker = relative_path.parts[0]

        # Audio filename identifies the utterance/sequence
        sequence = audio_path.stem

        try:

            features, sample_rate, num_samples = (
                extract_audio_features(audio_path)
            )

            record = {
                "speaker": speaker,
                "sequence": sequence,
                "audio_file": str(audio_path.relative_to(PROJECT_ROOT)),
                "sample_rate": sample_rate,
                "duration": num_samples / sample_rate,
                **features
            }

            records.append(record)

        except Exception as error:

            failures.append({
                "speaker": speaker,
                "sequence": sequence,
                "audio_file": str(audio_path),
                "error": str(error)
            })

        if index % 50 == 0:
            print(f"Processed {index}/{len(audio_files)} audio files")

    # ======================================
    # 5. SAVE FEATURES
    # ======================================

    df = pd.DataFrame(records)

    if not df.empty:

        df = df.sort_values(
            ["speaker", "sequence"]
        ).reset_index(drop=True)

        df.to_csv(
            OUTPUT_FILE,
            index=False
        )

    # Save failure report only if needed
    if failures:

        failure_file = OUTPUT_DIR / "voice_extraction_errors.csv"

        pd.DataFrame(failures).to_csv(
            failure_file,
            index=False
        )

    # ======================================
    # 6. FINAL SUMMARY
    # ======================================

    print("\n========== EXTRACTION SUMMARY ==========")

    print(f"Total Audio Files: {len(audio_files)}")
    print(f"Successfully Processed: {len(records)}")
    print(f"Failed: {len(failures)}")

    print(f"Number of Speakers: {df['speaker'].nunique() if not df.empty else 0}")
    print(f"Number of Features: {len(df.columns) - 5 if not df.empty else 0}")

    print(f"\nOutput File: {OUTPUT_FILE}")

    if failures:
        print(f"Error Report: {OUTPUT_DIR / 'voice_extraction_errors.csv'}")


if __name__ == "__main__":
    main()
