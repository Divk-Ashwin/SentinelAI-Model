"""
Debug script to trace why rows are being dropped during data loading.
Prints label distribution BEFORE and AFTER each cleaning step.
"""

import os
import sys
import glob
import logging
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_DIR = os.path.join(PROJECT_ROOT, "pipeline 1")


def print_label_distribution(df, step_name):
    """Print detailed label distribution."""
    print(f"\n{'=' * 70}")
    print(f"{step_name}")
    print(f"{'=' * 70}")
    print(f"Total rows: {len(df)}")

    if "label" in df.columns:
        print(f"\nLabel column dtype: {df['label'].dtype}")
        print(f"Unique label values: {df['label'].unique()}")
        print(f"\nLabel value_counts():")
        print(df['label'].value_counts(dropna=False))
        print(f"\nNull labels: {df['label'].isna().sum()}")
    else:
        print("WARNING: 'label' column not found in dataframe!")

    print(f"{'=' * 70}\n")


def load_data_from_pipeline():
    """Load and combine CSV files from pipeline 1 directory."""
    csv_files = glob.glob(os.path.join(PIPELINE_DIR, "*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {PIPELINE_DIR}")

    logger.info(f"Found {len(csv_files)} CSV files")

    dataframes = []
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        logger.info(f"Loading {filename}...")

        # Try different encodings
        df = None
        for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
            try:
                df = pd.read_csv(csv_file, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Error reading {filename} with {encoding}: {e}")
                break

        if df is None:
            logger.error(f"Could not read {filename} with any encoding. Skipping.")
            continue

        try:
            # Normalize column names to lowercase
            df.columns = df.columns.str.lower()

            # Handle different dataset formats
            if "kaggle" in filename.lower():
                # Kaggle: labels, text
                if "labels" in df.columns and "text" in df.columns:
                    dataframes.append(df[["labels", "text"]].rename(columns={"labels": "label"}))
                else:
                    logger.warning(f"{filename} doesn't have expected columns. Skipping.")

            elif "mendeley" in filename.lower():
                # Mendeley: LABEL, TEXT
                if "label" in df.columns and "text" in df.columns:
                    dataframes.append(df[["label", "text"]])
                else:
                    logger.warning(f"{filename} doesn't have expected columns. Skipping.")

            elif "smishtank" in filename.lower():
                # smishtank: uses Phishing label and MainText
                if "phishing" in df.columns and "maintext" in df.columns:
                    # Convert phishing (boolean/int) to binary: 1=spam, 0=ham
                    df_sub = df[["phishing", "maintext"]].copy()
                    # Drop rows with NaN values
                    df_sub = df_sub.dropna()
                    df_sub.columns = ["label", "text"]
                    df_sub["label"] = df_sub["label"].astype(int)
                    dataframes.append(df_sub)
                else:
                    logger.warning(f"{filename} doesn't have expected columns. Skipping.")

            elif "uci" in filename.lower():
                # UCI: v1 is label, v2 is text
                if "v1" in df.columns and "v2" in df.columns:
                    df_sub = df[["v1", "v2"]].copy()
                    df_sub.columns = ["label", "text"]
                    dataframes.append(df_sub)
                else:
                    logger.warning(f"{filename} doesn't have expected columns. Skipping.")

            else:
                # Generic fallback: look for label and text
                if "label" in df.columns and "text" in df.columns:
                    dataframes.append(df[["label", "text"]])
                else:
                    logger.warning(f"{filename} doesn't have expected columns. Skipping.")

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")

    if not dataframes:
        raise ValueError("No valid CSV files could be loaded")

    combined_df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Total records loaded: {len(combined_df)}")

    return combined_df


def debug_data_loading():
    """
    Debug data loading with detailed tracing of each step.
    """
    print("\n" + "=" * 70)
    print("DEBUG: DATA LOADING TRACE")
    print("=" * 70)

    # Step 0: Load raw data
    df = load_data_from_pipeline()
    total_raw = len(df)
    print_label_distribution(df, "STEP 0: RAW DATA (after combining all CSVs)")

    # Step 1: Drop rows with null labels
    print(f"\n>>> Applying Step 1: Dropping rows with null labels...")
    before_count = len(df)
    df = df[df["label"].notna()]
    after_count = len(df)
    print(f"Dropped {before_count - after_count} rows with null labels")
    print_label_distribution(df, "STEP 1: AFTER DROPPING NULL LABELS")

    # Step 2: Normalize labels
    print(f"\n>>> Applying Step 2: Normalizing labels...")

    # Define mappings
    spam_variants = {"spam", "phishing", "smishing", "1"}
    ham_variants = {"ham", "legitimate", "safe", "0", "non-phishing", "non-spam"}

    def normalize_label(label):
        """Convert label to 0 (ham) or 1 (spam), or -1 if unrecognized."""
        # Handle numeric labels directly
        if isinstance(label, (int, float)):
            if label == 1 or label == 1.0:
                return 1
            elif label == 0 or label == 0.0:
                return 0
            else:
                return -1

        # Handle string labels
        label_str = str(label).lower().strip()
        if label_str in spam_variants:
            return 1
        elif label_str in ham_variants:
            return 0
        else:
            return -1

    # Apply normalization
    df["label_original"] = df["label"].copy()  # Keep original for debugging
    df["label"] = df["label"].apply(normalize_label)

    # Check for any -1 (unrecognized) labels
    unrecognized_mask = df["label"] == -1
    unrecognized_count = unrecognized_mask.sum()

    if unrecognized_count > 0:
        print(f"\nWARNING: Found {unrecognized_count} rows with UNRECOGNIZED labels!")
        print("\nSample of unrecognized labels:")
        print(df.loc[unrecognized_mask, ["label_original", "label", "text"]].head(10))
        print("\nUnique unrecognized label values:")
        print(df.loc[unrecognized_mask, "label_original"].unique())

    print_label_distribution(df, "STEP 2: AFTER NORMALIZING LABELS (before dropping -1)")

    # Step 3: Drop rows with unrecognized labels (-1)
    print(f"\n>>> Applying Step 3: Dropping rows with unrecognized labels...")
    before_count = len(df)
    df = df[df["label"] != -1]
    after_count = len(df)
    print(f"Dropped {before_count - after_count} rows with unrecognized labels")
    print_label_distribution(df, "STEP 3: AFTER DROPPING UNRECOGNIZED LABELS")

    # Step 4: Drop rows with null or empty text
    print(f"\n>>> Applying Step 4: Dropping rows with null/empty text...")
    before_count = len(df)

    # Convert to string first
    df["text"] = df["text"].astype(str)

    # Check for null, empty, or "nan" text
    null_text_mask = df["text"].isna()
    empty_text_mask = df["text"].str.strip() == ""
    nan_string_mask = df["text"] == "nan"

    print(f"  - Null text: {null_text_mask.sum()}")
    print(f"  - Empty text (after strip): {empty_text_mask.sum()}")
    print(f"  - Text value is 'nan' string: {nan_string_mask.sum()}")

    # Drop these rows
    df = df[df["text"].notna() & (df["text"].str.strip() != "") & (df["text"] != "nan")]
    after_count = len(df)
    print(f"Dropped {before_count - after_count} rows with null/empty text")
    print_label_distribution(df, "STEP 4: AFTER DROPPING NULL/EMPTY TEXT")

    # Step 5: Check for text length filter (should be NONE)
    print(f"\n>>> Step 5: Checking for text length filter...")
    text_lengths = df["text"].str.len()
    print(f"Text length statistics:")
    print(f"  - Min length: {text_lengths.min()}")
    print(f"  - Max length: {text_lengths.max()}")
    print(f"  - Mean length: {text_lengths.mean():.1f}")
    print(f"  - Median length: {text_lengths.median():.1f}")
    print(f"\nTexts with length < 10 characters: {(text_lengths < 10).sum()}")
    print(f"Texts with length < 5 characters: {(text_lengths < 5).sum()}")
    print(f"Texts with length < 3 characters: {(text_lengths < 3).sum()}")

    print("\nNO TEXT LENGTH FILTER APPLIED (as requested)")

    # Step 6: Remove exact duplicates
    print(f"\n>>> Applying Step 6: Removing exact duplicates...")
    before_count = len(df)
    df = df.drop_duplicates(subset=["text", "label"])
    after_count = len(df)
    print(f"Dropped {before_count - after_count} duplicate rows")
    print_label_distribution(df, "STEP 6: AFTER REMOVING DUPLICATES")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Raw data: {total_raw} rows")
    print(f"Final data: {len(df)} rows")
    print(f"Retention rate: {len(df)/total_raw*100:.1f}%")
    print(f"Data loss: {total_raw - len(df)} rows ({(total_raw - len(df))/total_raw*100:.1f}%)")
    print("\n" + "=" * 70)
    print("FINAL LABEL DISTRIBUTION")
    print("=" * 70)
    print(f"Label 0 (Ham/Legitimate): {(df['label'] == 0).sum()} samples")
    print(f"Label 1 (Spam/Phishing): {(df['label'] == 1).sum()} samples")
    print(f"Total: {len(df)} samples")
    print(f"Class balance: {(df['label'] == 1).sum() / len(df) * 100:.1f}% spam")
    print("=" * 70)

    return df


if __name__ == "__main__":
    try:
        debug_data_loading()
        print("\nOK Debug data loading completed successfully!")
    except Exception as e:
        logger.error(f"ERROR Debug failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
