
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "face_preprocessing_report.csv"
)

df = pd.read_csv(REPORT_PATH)

failed = df[df["status"] == "No face detected"]

# Show 12 failed images
samples = failed.sample(
    n=min(12, len(failed)),
    random_state=42
)

fig, axes = plt.subplots(3, 4, figsize=(14, 10))
axes = axes.flatten()

for ax, (_, row) in zip(axes, samples.iterrows()):

    image_path = Path(row["original_file"])
    image = Image.open(image_path).convert("RGB")

    ax.imshow(image)
    ax.set_title(
        f'{row["speaker"]} / {row["sequence"]}',
        fontsize=9
    )
    ax.axis("off")

for ax in axes[len(samples):]:
    ax.axis("off")

plt.tight_layout()
plt.show()
