
from pathlib import Path

import cv2
import pandas as pd


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "vidtimit"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "face"

REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "face_preprocessing_report.csv"
)


# ==========================================
# 2. LOAD HAAR CASCADES
# ==========================================

CASCADE_DIR = Path(cv2.data.haarcascades)

FRONTAL_DETECTOR = cv2.CascadeClassifier(
    str(CASCADE_DIR / "haarcascade_frontalface_default.xml")
)

PROFILE_DETECTOR = cv2.CascadeClassifier(
    str(CASCADE_DIR / "haarcascade_profileface.xml")
)

if FRONTAL_DETECTOR.empty() or PROFILE_DETECTOR.empty():
    raise RuntimeError("Could not load Haar Cascade classifiers!")


# ==========================================
# 3. FIND VIDEO FOLDER
# ==========================================

def find_video_folder(speaker_path):

    candidates = [
        folder
        for folder in speaker_path.rglob("video")
        if folder.is_dir()
    ]

    if not candidates:
        return None

    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple video folders found for {speaker_path.name}"
        )

    return candidates[0]


# ==========================================
# 4. DETECT FACE
# ==========================================

def detect_face(gray):

    detections = []

    # Frontal face
    frontal_faces = FRONTAL_DETECTOR.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30)
    )

    detections.extend(list(frontal_faces))

    # Profile face
    profile_faces = PROFILE_DETECTOR.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30)
    )

    detections.extend(list(profile_faces))

    # Profile face after horizontal flip
    flipped = cv2.flip(gray, 1)

    flipped_faces = PROFILE_DETECTOR.detectMultiScale(
        flipped,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30)
    )

    image_width = gray.shape[1]

    for x, y, w, h in flipped_faces:
        original_x = image_width - (x + w)
        detections.append((original_x, y, w, h))

    if not detections:
        return None

    # Select largest detection
    return max(
        detections,
        key=lambda box: box[2] * box[3]
    )


# ==========================================
# 5. PREPROCESS IMAGE
# ==========================================

def preprocess_image(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        return None, "Could not read image"

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    detection = detect_face(gray)

    if detection is None:
        return None, "No face detected"

    x, y, w, h = detection

    # Add 15% padding
    pad_x = int(w * 0.15)
    pad_y = int(h * 0.15)

    image_height, image_width = gray.shape

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)

    x2 = min(image_width, x + w + pad_x)
    y2 = min(image_height, y + h + pad_y)

    face = gray[y1:y2, x1:x2]

    if face.size == 0:
        return None, "Invalid crop"

    # Resize to 128x128
    face = cv2.resize(
        face,
        (128, 128),
        interpolation=cv2.INTER_AREA
    )

    return face, "Success"


# ==========================================
# 6. MAIN PREPROCESSING PIPELINE
# ==========================================

def main():

    records = []

    total = 0
    successful = 0
    failed = 0

    speakers = sorted(
        path for path in DATASET_PATH.iterdir()
        if path.is_dir()
    )

    print(f"Total Speakers: {len(speakers)}")

    for speaker_path in speakers:

        print(f"\nProcessing: {speaker_path.name}")

        video_path = find_video_folder(speaker_path)

        if video_path is None:

            records.append({
                "speaker": speaker_path.name,
                "sequence": None,
                "original_file": None,
                "processed_file": None,
                "status": "Video folder not found"
            })

            failed += 1
            continue

        sequence_folders = sorted(
            folder for folder in video_path.iterdir()
            if folder.is_dir()
        )

        for sequence_path in sequence_folders:

            image_files = sorted(
                file for file in sequence_path.iterdir()
                if file.is_file()
            )

            for image_path in image_files:

                total += 1

                face, status = preprocess_image(image_path)

                output_file = None

                if face is not None:

                    output_dir = (
                        OUTPUT_PATH
                        / speaker_path.name
                        / sequence_path.name
                    )

                    output_dir.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                    output_file = (
                        output_dir / f"{image_path.stem}.jpg"
                    )

                    saved = cv2.imwrite(
                        str(output_file),
                        face
                    )

                    if saved:
                        successful += 1
                    else:
                        failed += 1
                        status = "Save failed"
                        output_file = None

                else:
                    failed += 1

                records.append({
                    "speaker": speaker_path.name,
                    "sequence": sequence_path.name,
                    "original_file": str(image_path),
                    "processed_file": (
                        str(output_file)
                        if output_file else None
                    ),
                    "status": status
                })

                if total % 1000 == 0:
                    print(f"Processed {total} frames...")

    # Save report
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    pd.DataFrame(records).to_csv(
        REPORT_PATH,
        index=False
    )

    print("\n========== FACE PREPROCESSING SUMMARY ==========")
    print(f"Total Speakers: {len(speakers)}")
    print(f"Total Frames: {total}")
    print(f"Successfully Processed: {successful}")
    print(f"Failed: {failed}")

    if total > 0:
        print(f"Success Rate: {successful / total * 100:.2f}%")

    print(f"\nOutput: {OUTPUT_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        selected_speaker = sys.argv[1]
        original_iterdir = Path.iterdir

        def filtered_iterdir(path):
            if path == DATASET_PATH:
                return iter([DATASET_PATH / selected_speaker])
            return original_iterdir(path)

        Path.iterdir = filtered_iterdir

    main()

