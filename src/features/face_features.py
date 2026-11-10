
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from skimage.feature import hog, local_binary_pattern


# ==========================================
# 1. PROJECT PATHS AND CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FACE_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "face"
VOICE_FEATURES_FILE = (
    PROJECT_ROOT / "data" / "processed" / "voice" / "voice_features.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "face"
OUTPUT_FILE = OUTPUT_DIR / "face_features.csv"
ERROR_FILE = PROJECT_ROOT / "data" / "metadata" / "face_feature_errors.csv"

FRAMES_PER_SEQUENCE = 20
IMAGE_SIZE = (128, 128)

# HOG configuration
HOG_ORIENTATIONS = 9
PIXELS_PER_CELL = (16, 16)
CELLS_PER_BLOCK = (2, 2)

# LBP configuration
LBP_POINTS = 8
LBP_RADIUS = 1
LBP_METHOD = "uniform"
LBP_BINS = LBP_POINTS + 2


# ==========================================
# 2. HOG FEATURE EXTRACTION
# ==========================================

def extract_hog_features(image):
    features = hog(
        image,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=PIXELS_PER_CELL,
        cells_per_block=CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        visualize=False,
        feature_vector=True,
    )

    return features


# ==========================================
# 3. LBP FEATURE EXTRACTION
# ==========================================

def extract_lbp_features(image):
    lbp_image = local_binary_pattern(
        image,
        P=LBP_POINTS,
        R=LBP_RADIUS,
        method=LBP_METHOD,
    )

    histogram, _ = np.histogram(
        lbp_image.ravel(),
        bins=np.arange(0, LBP_BINS + 1),
        range=(0, LBP_BINS),
    )

    histogram = histogram.astype(np.float32)
    histogram /= histogram.sum() + 1e-8

    return histogram


# ==========================================
# 4. SELECT REPRESENTATIVE FRAMES
# ==========================================

def select_frames(frame_paths, max_frames=FRAMES_PER_SEQUENCE):
    frame_paths = sorted(frame_paths)

    if len(frame_paths) <= max_frames:
        return frame_paths

    indices = np.linspace(
        0,
        len(frame_paths) - 1,
        num=max_frames,
        dtype=int,
    )

    return [frame_paths[i] for i in indices]


# ==========================================
# 5. EXTRACT FEATURES FROM ONE SEQUENCE
# ==========================================

def extract_sequence_features(speaker, sequence):
    sequence_path = FACE_DATA_PATH / speaker / sequence

    if not sequence_path.exists():
        raise FileNotFoundError(
            f"Sequence folder not found: {sequence_path}"
        )

    frame_paths = list(sequence_path.glob("*.jpg"))

    if not frame_paths:
        raise ValueError("No processed face frames found.")

    selected_frames = select_frames(frame_paths)

    hog_features_list = []
    lbp_features_list = []

    for frame_path in selected_frames:

        image = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            continue

        if image.shape != IMAGE_SIZE:
            image = cv2.resize(image, IMAGE_SIZE)

        hog_features_list.append(
            extract_hog_features(image)
        )

        lbp_features_list.append(
            extract_lbp_features(image)
        )

    if not hog_features_list:
        raise ValueError("All selected frames failed to load.")

    hog_array = np.asarray(hog_features_list)
    lbp_array = np.asarray(lbp_features_list)

    # Aggregate frame-level features into one sequence-level sample
    hog_mean = np.mean(hog_array, axis=0)
    hog_std = np.std(hog_array, axis=0)

    lbp_mean = np.mean(lbp_array, axis=0)
    lbp_std = np.std(lbp_array, axis=0)

    combined_features = np.concatenate([
        hog_mean,
        hog_std,
        lbp_mean,
        lbp_std,
    ])

    return {
        "features": combined_features,
        "total_frames": len(frame_paths),
        "sampled_frames": len(hog_features_list),
    }


# ==========================================
# 6. MAIN FEATURE EXTRACTION PIPELINE
# ==========================================

def main():

    print("=" * 55)
    print("FACE FEATURE EXTRACTION - HOG + LBP")
    print("=" * 55)

    if not VOICE_FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"Voice reference file not found: {VOICE_FEATURES_FILE}"
        )

    voice_df = pd.read_csv(VOICE_FEATURES_FILE)

    # Use voice recordings as the reference for exact modality alignment
    recordings = voice_df[["speaker", "sequence"]].drop_duplicates()

    print("Expected recordings:", len(recordings))
    print("Speakers:", recordings["speaker"].nunique())
    print("Frames sampled per sequence:", FRAMES_PER_SEQUENCE)

    results = []
    errors = []

    for index, row in recordings.iterrows():

        speaker = str(row["speaker"])
        sequence = str(row["sequence"])

        try:
            result = extract_sequence_features(
                speaker,
                sequence,
            )

            feature_values = result["features"]

            record = {
                "speaker": speaker,
                "sequence": sequence,
                "total_frames": result["total_frames"],
                "sampled_frames": result["sampled_frames"],
            }

            for i, value in enumerate(feature_values):
                record[f"face_feature_{i + 1}"] = float(value)

            results.append(record)

        except Exception as error:
            errors.append({
                "speaker": speaker,
                "sequence": sequence,
                "error": str(error),
            })

        completed = len(results) + len(errors)

        if completed % 25 == 0 or completed == len(recordings):
            print(
                f"Progress: {completed}/{len(recordings)} | "
                f"Successful: {len(results)} | "
                f"Failed: {len(errors)}"
            )

    # Save extracted features
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if results:
        features_df = pd.DataFrame(results)
        features_df.to_csv(OUTPUT_FILE, index=False)

        print("\nFeature dataset saved:", OUTPUT_FILE)
        print("Dataset shape:", features_df.shape)
        print(
            "Number of feature columns:",
            len([
                col for col in features_df.columns
                if col.startswith("face_feature_")
            ]),
        )

    if errors:
        error_df = pd.DataFrame(errors)
        ERROR_FILE.parent.mkdir(parents=True, exist_ok=True)
        error_df.to_csv(ERROR_FILE, index=False)

        print("\nErrors saved:", ERROR_FILE)

    print("\n========== FINAL SUMMARY ==========")
    print("Expected recordings:", len(recordings))
    print("Successfully processed:", len(results))
    print("Failed:", len(errors))

    if len(recordings) > 0:
        success_rate = len(results) / len(recordings) * 100
        print(f"Success rate: {success_rate:.2f}%")

    print("===================================")


if __name__ == "__main__":
    main()
