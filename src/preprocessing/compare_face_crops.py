
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = (
    PROJECT_ROOT / "data" / "metadata"
    / "multiple_detector_results_padded.csv"
)

df = pd.read_csv(CSV_PATH)

recovered = df[df["status"] == "Recovered"]

samples = recovered.sample(
    n=min(6, len(recovered)),
    random_state=42
)

fig, axes = plt.subplots(
    len(samples), 2,
    figsize=(9, 3 * len(samples))
)

if len(samples) == 1:
    axes = [axes]

for ax, (_, row) in zip(axes, samples.iterrows()):

    original_crop_path = (
        PROJECT_ROOT / "data" / "processed"
        / "face_recovered"
        / row["speaker"]
        / row["sequence"]
        / f'{Path(row["original_file"]).stem}.jpg'
    )

    padded_crop_path = Path(row["recovered_file"])

    original_crop = Image.open(
        original_crop_path
    ).convert("RGB")

    padded_crop = Image.open(
        padded_crop_path
    ).convert("RGB")

    ax[0].imshow(original_crop)
    ax[0].set_title("Without Padding")
    ax[0].axis("off")

    ax[1].imshow(padded_crop)
    ax[1].set_title("With 15% Padding")
    ax[1].axis("off")

    fig.suptitle(
        f'{row["speaker"]} / {row["sequence"]}'
    )

plt.tight_layout()
plt.show()
