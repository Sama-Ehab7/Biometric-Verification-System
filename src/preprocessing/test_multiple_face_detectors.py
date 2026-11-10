
from pathlib import Path
import cv2
import pandas as pd

# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "face_preprocessing_report.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "face_recovered_padded"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD OPENCV CASCADES
# ==========================================

cascade_dir = Path(cv2.data.haarcascades)

frontal = cv2.CascadeClassifier(
    str(cascade_dir / "haarcascade_frontalface_default.xml")
)

profile = cv2.CascadeClassifier(
    str(cascade_dir / "haarcascade_profileface.xml")
)

if frontal.empty() or profile.empty():
    raise RuntimeError("Could not load one or more cascades.")


# ==========================================
# 3. LOAD FAILED IMAGES
# ==========================================

df = pd.read_csv(REPORT_PATH)

failed = df[
    df["status"] == "No face detected"
].copy()

results = []


# ==========================================
# 4. PROCESS EACH FAILED IMAGE
# ==========================================

for index, row in failed.iterrows():

    image_path = Path(row["original_file"])

    image = cv2.imread(str(image_path))

    if image is None:
        results.append({
            "speaker": row["speaker"],
            "sequence": row["sequence"],
            "original_file": str(image_path),
            "recovered_file": None,
            "status": "Could not read image"
        })
        continue

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    detections = []

    # --------------------------------------
    # 1. Frontal Face Detection
    # --------------------------------------

    faces = frontal.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30)
    )

    detections.extend(list(faces))

    # --------------------------------------
    # 2. Profile Face Detection
    # --------------------------------------

    profiles = profile.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30)
    )

    detections.extend(list(profiles))

    # --------------------------------------
    # 3. Profile Detection After Flip
    # --------------------------------------

    flipped = cv2.flip(gray, 1)

    flipped_profiles = profile.detectMultiScale(
        flipped,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30)
    )

    # Convert flipped coordinates to original coordinates
    width = gray.shape[1]

    for x, y, w, h in flipped_profiles:

        original_x = width - (x + w)

        detections.append(
            (original_x, y, w, h)
        )

    # --------------------------------------
    # 4. Select Largest Detection
    # --------------------------------------

    if detections:

        x, y, w, h = max(
            detections,
            key=lambda box: box[2] * box[3]
        )

        # ----------------------------------
        # 5. Add 15% Padding
        # ----------------------------------

        pad_x = int(w * 0.15)
        pad_y = int(h * 0.15)

        img_h, img_w = image.shape[:2]

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)

        x2 = min(img_w, x + w + pad_x)
        y2 = min(img_h, y + h + pad_y)

        face = image[y1:y2, x1:x2]

        # ----------------------------------
        # 6. Resize and Save Face
        # ----------------------------------

        if face.size > 0:

            face = cv2.resize(
                face,
                (128, 128)
            )

            output_path = (
                OUTPUT_DIR
                / row["speaker"]
                / row["sequence"]
            )

            output_path.mkdir(
                parents=True,
                exist_ok=True
            )

            save_path = (
                output_path
                / f"{image_path.stem}.jpg"
            )

            success = cv2.imwrite(
                str(save_path),
                face
            )

            if success:
                status = "Recovered"
                recovered_file = str(save_path)
            else:
                status = "Save failed"
                recovered_file = None

        else:

            status = "Invalid crop"
            recovered_file = None

    else:

        status = "Still not detected"
        recovered_file = None

    # --------------------------------------
    # 7. Store Result
    # --------------------------------------

    results.append({
        "speaker": row["speaker"],
        "sequence": row["sequence"],
        "original_file": str(image_path),
        "recovered_file": recovered_file,
        "status": status
    })

    if (len(results) % 50) == 0:
        print(f"Processed {len(results)} images...")


# ==========================================
# 5. SAVE RESULTS
# ==========================================

results_df = pd.DataFrame(results)

results_csv = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "multiple_detector_results_padded.csv"
)

results_df.to_csv(
    results_csv,
    index=False
)


# ==========================================
# 6. PRINT SUMMARY
# ==========================================

print("\n===== MULTIPLE DETECTORS RESULTS =====")

print(
    results_df["status"].value_counts()
)

recovered = (
    results_df["status"] == "Recovered"
).sum()

total = len(results_df)

not_detected = (
    results_df["status"] == "Still not detected"
).sum()

print(f"\nFailed images tested: {total}")
print(f"Recovered images: {recovered}")
print(f"Still not detected: {not_detected}")

if total > 0:

    recovery_rate = recovered / total * 100

    print(
        f"Recovery rate: {recovery_rate:.2f}%"
    )

print("\nResults saved to:", results_csv)

print("\nRecovered faces saved to:", OUTPUT_DIR)
