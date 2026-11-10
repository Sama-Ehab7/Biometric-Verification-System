
from pathlib import Path

import pandas as pd


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fingerprint"
    / "fingerprint_features.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fingerprint"
    / "splits"
)

REAL_DATABASES = ["DB1_B", "DB2_B", "DB3_B"]
SYNTHETIC_DATABASES = ["DB4_B"]

EXPECTED_IMPRESSIONS = set(range(1, 9))


# ==========================================
# 2. ASSIGNMENT OF IMPRESSIONS
# ==========================================

def assign_split(impression_number):

    if impression_number in [1, 2, 3, 4]:
        return "enrollment"

    elif impression_number in [5, 6]:
        return "validation"

    elif impression_number in [7, 8]:
        return "test"

    raise ValueError(
        f"Unexpected impression number: {impression_number}"
    )


# ==========================================
# 3. LOAD AND VALIDATE FEATURES
# ==========================================

def load_features():

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Features file not found: {FEATURES_PATH}"
        )

    df = pd.read_csv(FEATURES_PATH)

    required_columns = {
        "database",
        "subject_id",
        "identity",
        "image_name"
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df.empty:
        raise ValueError("Features file is empty.")

    # Extract impression number from image name.
    # Example: 101_3.png -> 3

    df["impression_number"] = (
        df["image_name"]
        .str.extract(r"_(\d+)(?:\.[^.]+)?$")[0]
    )

    if df["impression_number"].isna().any():
        raise ValueError(
            "Could not extract impression numbers from some images."
        )

    df["impression_number"] = (
        df["impression_number"].astype(int)
    )

    # Validate identity format.
    expected_identity = (
        df["database"].astype(str)
        + "_"
        + df["subject_id"].astype(str)
    )

    if not (df["identity"] == expected_identity).all():
        raise ValueError(
            "Identity values do not match database and subject ID."
        )

    # Check duplicate image records.
    duplicate_mask = df.duplicated(
        subset=["identity", "image_name"]
    )

    if duplicate_mask.any():
        raise ValueError(
            f"Found {duplicate_mask.sum()} duplicate image records."
        )

    return df


# ==========================================
# 4. VALIDATE EACH IDENTITY
# ==========================================

def validate_identities(df):

    errors = []

    for identity, group in df.groupby("identity"):

        actual = set(group["impression_number"])

        if actual != EXPECTED_IMPRESSIONS:
            errors.append({
                "identity": identity,
                "actual_impressions": sorted(actual),
                "expected_impressions": sorted(
                    EXPECTED_IMPRESSIONS
                )
            })

    if errors:
        raise ValueError(
            f"Identity validation failed: {errors}"
        )

    print("Identity validation: PASSED")
    print("Every identity has exactly 8 impressions.")


# ==========================================
# 5. CREATE SPLITS
# ==========================================

def create_splits(df, dataset_type):

    if dataset_type == "real":
        databases = REAL_DATABASES

    else:
        databases = SYNTHETIC_DATABASES

    subset = df[df["database"].isin(databases)].copy()

    if subset.empty:
        raise ValueError(
            f"No records found for {dataset_type} databases."
        )

    subset["split"] = subset["impression_number"].apply(
        assign_split
    )

    splits = {
        "enrollment": subset[
            subset["split"] == "enrollment"
        ].copy(),

        "validation": subset[
            subset["split"] == "validation"
        ].copy(),

        "test": subset[
            subset["split"] == "test"
        ].copy()
    }

    return splits


# ==========================================
# 6. CHECK DATA LEAKAGE
# ==========================================

def check_split_leakage(splits):

    image_sets = {}

    for split_name, split_df in splits.items():

        image_sets[split_name] = set(
            zip(
                split_df["identity"],
                split_df["image_name"]
            )
        )

    split_names = list(image_sets.keys())

    for i in range(len(split_names)):

        for j in range(i + 1, len(split_names)):

            first = split_names[i]
            second = split_names[j]

            overlap = (
                image_sets[first]
                & image_sets[second]
            )

            if overlap:
                raise ValueError(
                    f"Data leakage between {first} and {second}: "
                    f"{len(overlap)} overlapping images."
                )

    print("Image-level leakage check: PASSED")

    # Each identity must be represented in all three splits.
    identity_sets = {
        name: set(split_df["identity"])
        for name, split_df in splits.items()
    }

    common_identities = set.intersection(
        *identity_sets.values()
    )

    if len(common_identities) != subset_identity_count(splits):
        raise ValueError(
            "Some identities are missing from one or more splits."
        )

    print(
        f"Identities present in all splits: "
        f"{len(common_identities)}"
    )


def subset_identity_count(splits):

    return len(
        set.union(
            *[
                set(split_df["identity"])
                for split_df in splits.values()
            ]
        )
    )


# ==========================================
# 7. SAVE SPLITS
# ==========================================

def save_splits(splits, dataset_type):

    output_folder = OUTPUT_ROOT / dataset_type

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    for split_name, split_df in splits.items():

        output_path = (
            output_folder
            / f"fingerprint_{split_name}.csv"
        )

        split_df.to_csv(
            output_path,
            index=False
        )

        print(
            f"Saved {split_name}: "
            f"{len(split_df)} records"
        )

        print(f"  {output_path}")


# ==========================================
# 8. PRINT SUMMARY
# ==========================================

def print_summary(splits, dataset_type):

    print("\n" + "=" * 55)
    print(f"{dataset_type.upper()} FINGERPRINT SPLIT SUMMARY")
    print("=" * 55)

    for split_name, split_df in splits.items():

        print(f"\n{split_name.upper()}")
        print(f"Images: {len(split_df)}")
        print(
            f"Identities: "
            f"{split_df['identity'].nunique()}"
        )

        print(
            split_df.groupby("database").size().to_string()
        )


# ==========================================
# 9. MAIN
# ==========================================

def main():

    print("\nFINGERPRINT DATASET PREPARATION")
    print("=" * 55)

    df = load_features()

    print(f"Total feature records: {len(df)}")

    validate_identities(df)

    # Check that only the expected databases exist.
    expected_databases = set(
        REAL_DATABASES + SYNTHETIC_DATABASES
    )

    actual_databases = set(df["database"])

    if actual_databases != expected_databases:
        raise ValueError(
            f"Unexpected databases. Found: {actual_databases}"
        )

    for dataset_type in ["real", "synthetic"]:

        splits = create_splits(df, dataset_type)

        check_split_leakage(splits)

        save_splits(splits, dataset_type)

        print_summary(splits, dataset_type)

    print("\n" + "=" * 55)
    print("ALL FINGERPRINT SPLITS CREATED SUCCESSFULLY")
    print("=" * 55)

    print(f"Output directory:\n{OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
