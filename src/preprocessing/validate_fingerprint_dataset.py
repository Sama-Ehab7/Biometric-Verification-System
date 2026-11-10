
from pathlib import Path
from collections import defaultdict
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "fingerprint"

DATABASES = ["DB1_B", "DB2_B", "DB3_B", "DB4_B"]

def validate_database(database_name):
    folder = DATASET_ROOT / database_name

    print(f"\n{'=' * 50}")
    print(f"Database: {database_name}")
    print(f"{'=' * 50}")

    if not folder.exists():
        print("ERROR: Folder not found!")
        return

    images = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff", ".bmp", ".png", ".jpg", ".jpeg"}
    )

    identities = defaultdict(list)
    corrupted = []

    for image_path in images:
        identity = image_path.stem.rsplit("_", 1)[0]

        try:
            with Image.open(image_path) as img:
                img.verify()
            identities[identity].append(image_path.name)
        except Exception as error:
            corrupted.append((image_path.name, str(error)))

    print(f"Total images: {len(images)}")
    print(f"Total identities: {len(identities)}")

    for identity, files in sorted(identities.items()):
        print(f"{identity}: {len(files)} images")

    print(f"Corrupted images: {len(corrupted)}")

    for name, error in corrupted:
        print(f"  {name}: {error}")


def main():
    for database in DATABASES:
        validate_database(database)

if __name__ == "__main__":
    main()
