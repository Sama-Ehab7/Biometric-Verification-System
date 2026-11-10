
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "face_preprocessing_report.csv"
)

df = pd.read_csv(REPORT_PATH)

print("===== FACE PREPROCESSING FAILURE ANALYSIS =====")

# Overall statistics
print("\nOverall Status:")
print(df["status"].value_counts())

# Results per sequence
print("\nResults Per Sequence:")

summary = (
    df.groupby(["sequence", "status"])
    .size()
    .unstack(fill_value=0)
)

if "Success" not in summary.columns:
    summary["Success"] = 0

if "No face detected" not in summary.columns:
    summary["No face detected"] = 0

summary["Success Rate (%)"] = (
    summary["Success"]
    / (summary["Success"] + summary["No face detected"])
    * 100
)

print(summary.round(2).to_string())

# Failed image samples
failed = df[df["status"] != "Success"]

print("\nTotal Failed Images:", len(failed))

print("\nFirst 15 Failed Image Paths:")

for path in failed["original_file"].head(15):
    print(path)

# Save detailed analysis
output_path = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "face_failure_analysis.csv"
)

summary.to_csv(output_path)

print("\nAnalysis saved to:", output_path)
