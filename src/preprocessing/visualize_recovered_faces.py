
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = (
    PROJECT_ROOT / "data" / "metadata"
    / "multiple_detector_results.csv"
)

df = pd.read_csv(CSV_PATH)
recovered = df[df["status"] == "Recovered"].copy()

print("Recovered images per sequence:")
print(recovered["sequence"].value_counts())

# Select 6 samples for visual inspection
samples = recovered.sample(
    n=min(6, len(recovered)),
    random_state=42
)

fig, axes = plt.subplots(
    len(samples), 2,
    figsize=(10, 3 * len(samples))
)

if len(samples) == 1:
    axes = [axes]

for ax, (_, row) in zip(axes, samples.iterrows()):

    original = Image.open(
        row["original_file"]
    ).convert("RGB")

    crop = Image.open(
        row["recovered_file"]
    ).convert("RGB")

    ax[0].imshow(original)
    ax[0].set_title(
        f'Original - {row["sequence"]}'
    )
    ax[0].axis("off")

    ax[1].imshow(crop)
    ax[1].set_title("Recovered Face Crop")
    ax[1].axis("off")

plt.tight_layout()
plt.show()
