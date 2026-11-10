
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from skimage.feature import hog, local_binary_pattern


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "fingerprint"

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fingerprint"
    / "fingerprint_features.csv"
)

DATABASES = ["DB1_B", "DB2_B", "DB3_B", "DB4_B"]

# HOG parameters
HOG_ORIENTATIONS = 9
PIXELS_PER_CELL = (16, 16)
CELLS_PER_BLOCK = (2, 2)

# LBP parameters
LBP_POINTS = 8
LBP_RADIUS = 1
LBP_METHOD = "uniform"
LBP_BINS = LBP_POINTS + 2


# ==========================================
# 2. HOG FEATURE EXTRACTION
# ==========================================

def extract_hog_features(image):
    """
    Extract Histogram of Oriented Gradients features.
    """

    features = hog(
        image,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=PIXELS_PER_CELL,
        cells_per_block=CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True
    )

    return features


# ==========================================
# 3. LBP FEATURE EXTRACTION
# ==========================================

def extract_lbp_features(image):
    """
    Extract Local Binary Pattern histogram features.
    """

    lbp_image = local_binary_pattern(
        image,
        P=LBP_POINTS,
        R=LBP_RADIUS,
        method=LBP_METHOD
    )

    histogram, _ = np.histogram(
        lbp_image.ravel(),
        bins=np.arange(0, LBP_BINS + 1),
        range=(0, LBP_BINS)
    )

    # Normalize histogram
    histogram = histogram.astype(np.float32)

    histogram /= (histogram.sum() + 1e-7)

    return histogram


# ==========================================
# 4. COMBINED FEATURE EXTRACTION
# ==========================================

def extract_fingerprint_features(image_path):

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise ValueError("Unable to read processed image")

    if image.shape != (256, 256):
        raise ValueError(
            f"Unexpected image dimensions: {image.shape}"
        )

    hog_features = extract_hog_features(image)

    lbp_features = extract_lbp_features(image)

    combined_features = np.concatenate([
        hog_features,
        lbp_features
    ])

    return combined_features, len(hog_features), len(lbp_features)


# ==========================================
# 5. PROCESS ALL FINGERPRINT IMAGES
# ==========================================

def main():

    print("\nFINGERPRINT FEATURE EXTRACTION")
    print("=" * 55)

    records = []

    failed_images = []

    hog_length = None
    lbp_length = None

    for database in DATABASES:

        database_path = INPUT_ROOT / database

        print(f"\nProcessing {database}...")

        if not database_path.exists():
            print(f"WARNING: Missing database {database}")
            continue

        image_paths = sorted(
            path for path in database_path.rglob("*.png")
            if path.is_file()
        )

        successful = 0

        for image_path in image_paths:

            subject_id = image_path.parent.name

            identity = f"{database}_{subject_id}"

            try:

                features, current_hog_length, current_lbp_length = (
                    extract_fingerprint_features(image_path)
                )

                hog_length = current_hog_length
                lbp_length = current_lbp_length

                record = {
                    "database": database,
                    "subject_id": subject_id,
                    "identity": identity,
                    "image_name": image_path.name,
                    "image_path": str(
                        image_path.relative_to(PROJECT_ROOT)
                    )
                }

                for index, value in enumerate(
                    features[:current_hog_length]
                ):
                    record[f"fingerprint_hog_{index:04d}"] = float(value)

                for index, value in enumerate(
                    features[current_hog_length:]
                ):
                    record[f"fingerprint_lbp_{index:02d}"] = float(value)

                records.append(record)

                successful += 1

            except Exception as error:

                failed_images.append({
                    "image": str(image_path),
                    "error": str(error)
                })

                print(f"Failed: {image_path.name}: {error}")

        print(f"Images found: {len(image_paths)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(image_paths) - successful}")

    # ======================================
    # 6. SAVE FEATURES
    # ======================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if not records:
        print("\nERROR: No features were extracted.")
        return

    features_df = pd.DataFrame(records)

    features_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ======================================
    # 7. QUALITY CHECKS
    # ======================================

    feature_columns = [
        column for column in features_df.columns
        if column.startswith("fingerprint_")
    ]

    values = features_df[feature_columns].to_numpy(
        dtype=np.float64
    )

    print("\n" + "=" * 55)
    print("FEATURE EXTRACTION SUMMARY")
    print("=" * 55)

    print(f"Total successful images: {len(features_df)}")
    print(f"Failed images: {len(failed_images)}")
    print(f"Number of identities: {features_df['identity'].nunique()}")
    print(f"HOG features per image: {hog_length}")
    print(f"LBP features per image: {lbp_length}")
    print(f"Total features per image: {len(feature_columns)}")

    print(f"Feature matrix shape: {values.shape}")

    print(
        f"Missing values: "
        f"{features_df[feature_columns].isna().sum().sum()}"
    )

    print(
        f"Non-finite values: "
        f"{np.isfinite(values).sum() != values.size}"
    )

    print(
        f"Duplicate image records: "
        f"{features_df.duplicated(['identity', 'image_name']).sum()}"
    )

    print("\nImages per identity:")

    print(
        features_df.groupby("identity").size().to_string()
    )

    print(f"\nFeatures saved to:\n{OUTPUT_PATH}")

    if failed_images:
        failed_path = OUTPUT_PATH.parent / "fingerprint_failed_images.csv"

        pd.DataFrame(failed_images).to_csv(
            failed_path,
            index=False
        )

        print(f"Failed image report:\n{failed_path}")

    else:
        print("\nAll images processed successfully!")


if __name__ == "__main__":
    main()
