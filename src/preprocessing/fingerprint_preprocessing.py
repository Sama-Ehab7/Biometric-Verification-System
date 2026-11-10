
from pathlib import Path
import cv2
import pandas as pd

# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_ROOT = PROJECT_ROOT / "data" / "raw" / "fingerprint"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "fingerprint"
REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "fingerprint_preprocessing_report.csv"
)

DATABASES = ["DB1_B", "DB2_B", "DB3_B", "DB4_B"]

SUPPORTED_EXTENSIONS = {".tif", ".tiff"}

TARGET_SIZE = 256


# ==========================================
# 2. IMAGE PREPROCESSING FUNCTIONS
# ==========================================

def resize_with_padding(image, target_size=256):
    """
    Resize an image while preserving its aspect ratio.
    Add white padding to obtain a square image.
    """

    height, width = image.shape

    scale = min(target_size / width, target_size / height)

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    padded = cv2.copyMakeBorder(
        resized,
        top=(target_size - new_height) // 2,
        bottom=(target_size - new_height + 1) // 2,
        left=(target_size - new_width) // 2,
        right=(target_size - new_width + 1) // 2,
        borderType=cv2.BORDER_CONSTANT,
        value=255
    )

    return padded


def preprocess_fingerprint(image_path):
    """
    Apply preprocessing to one fingerprint image.

    Steps:
    1. Read as grayscale.
    2. Apply CLAHE for contrast enhancement.
    3. Apply median filtering to reduce noise.
    4. Resize with aspect ratio preservation and padding.
    """

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise ValueError("Unable to read image")

    original_height, original_width = image.shape

    # Contrast enhancement
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(image)

    # Noise reduction
    denoised = cv2.medianBlur(enhanced, 3)

    # Standardize image dimensions
    processed = resize_with_padding(
        denoised,
        TARGET_SIZE
    )

    return processed, original_width, original_height


# ==========================================
# 3. PROCESS ONE DATABASE
# ==========================================

def process_database(database_name):

    database_path = INPUT_ROOT / database_name

    print(f"\n{'=' * 55}")
    print(f"Processing: {database_name}")
    print(f"{'=' * 55}")

    if not database_path.exists():
        print(f"ERROR: Database folder not found: {database_path}")
        return []

    image_paths = sorted(
        path for path in database_path.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    records = []

    for image_path in image_paths:

        # Example: 101_1.tif
        # Subject ID: 101
        subject_id = image_path.stem.rsplit("_", 1)[0]

        # Database is part of the identity.
        # DB1_101 is different from DB2_101.
        identity = f"{database_name}_{subject_id}"

        output_folder = (
            OUTPUT_ROOT
            / database_name
            / subject_id
        )

        output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path = output_folder / f"{image_path.stem}.png"

        try:
            processed, original_width, original_height = (
                preprocess_fingerprint(image_path)
            )

            success = cv2.imwrite(
                str(output_path),
                processed
            )

            if not success:
                raise IOError("Failed to save processed image")

            records.append({
                "database": database_name,
                "subject_id": subject_id,
                "identity": identity,
                "image_name": image_path.name,
                "source_path": str(image_path.relative_to(PROJECT_ROOT)),
                "processed_path": str(output_path.relative_to(PROJECT_ROOT)),
                "original_width": original_width,
                "original_height": original_height,
                "processed_width": processed.shape[1],
                "processed_height": processed.shape[0],
                "status": "success",
                "error": ""
            })

        except Exception as error:

            records.append({
                "database": database_name,
                "subject_id": subject_id,
                "identity": identity,
                "image_name": image_path.name,
                "source_path": str(image_path.relative_to(PROJECT_ROOT)),
                "processed_path": "",
                "original_width": None,
                "original_height": None,
                "processed_width": None,
                "processed_height": None,
                "status": "failed",
                "error": str(error)
            })

            print(f"Failed: {image_path.name} - {error}")

    successful = sum(
        record["status"] == "success"
        for record in records
    )

    print(f"Total images: {len(image_paths)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(records) - successful}")

    return records


# ==========================================
# 4. MAIN PIPELINE
# ==========================================

def main():

    print("\nFINGERPRINT PREPROCESSING PIPELINE")
    print(f"Input:  {INPUT_ROOT}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Target size: {TARGET_SIZE} x {TARGET_SIZE}")

    all_records = []

    for database in DATABASES:
        records = process_database(database)
        all_records.extend(records)

    # Save preprocessing report
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    report = pd.DataFrame(all_records)

    report.to_csv(
        REPORT_PATH,
        index=False
    )

    # Final summary
    total = len(report)

    successful = (
        report["status"] == "success"
    ).sum() if total else 0

    failed = total - successful

    print("\n" + "=" * 55)
    print("FINAL PREPROCESSING SUMMARY")
    print("=" * 55)

    print(f"Total images: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if total:
        print(
            f"Success rate: "
            f"{successful / total * 100:.2f}%"
        )

    print(f"\nReport saved to:\n{REPORT_PATH}")
    print(f"Processed images saved to:\n{OUTPUT_ROOT}")

    if failed:
        print("\nWARNING: Some images failed preprocessing.")
    else:
        print("\nAll fingerprint images processed successfully!")


if __name__ == "__main__":
    main()
