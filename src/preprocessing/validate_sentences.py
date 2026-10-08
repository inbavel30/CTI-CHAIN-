import pandas as pd
from pathlib import Path

CSV_FILE = Path("data/processed/sentences.csv")

df = pd.read_csv(CSV_FILE)

print("\n===================================")
print("CTI SENTENCE DATASET VALIDATION")
print("===================================")

# 1. Basic statistics
print(f"\nTotal rows       : {len(df):,}")
print(f"Unique reports   : {df['report_id'].nunique()}")
print(f"Unique sentences : {df['sentence_id'].nunique()}")

# 2. Expected report IDs
expected_ids = {f"R{i:03d}" for i in range(1, 51)}
actual_ids = set(df["report_id"].dropna().astype(str))

missing = sorted(expected_ids - actual_ids)
extra = sorted(actual_ids - expected_ids)

print("\n--- REPORT CHECK ---")

if not missing:
    print("Missing reports  : NONE")
else:
    print("Missing reports  :", missing)

if not extra:
    print("Unexpected IDs   : NONE")
else:
    print("Unexpected IDs   :", extra)

# 3. Missing values
print("\n--- MISSING VALUES ---")

for column in df.columns:
    missing_count = df[column].isna().sum()
    print(f"{column:15} : {missing_count}")

# 4. Empty text
empty_text = df["text"].fillna("").astype(str).str.strip().eq("").sum()

print("\n--- TEXT CHECK ---")
print(f"Empty sentences  : {empty_text}")

# 5. Duplicate sentence IDs
duplicate_ids = df["sentence_id"].duplicated().sum()

print(f"Duplicate IDs    : {duplicate_ids}")

# 6. Duplicate text within same report
duplicate_text = df.duplicated(
    subset=["report_id", "text"]
).sum()

print(f"Duplicate text   : {duplicate_text}")

# 7. Sentences per report
report_counts = (
    df.groupby("report_id")
      .size()
      .sort_values()
)

print("\n--- SENTENCES PER REPORT ---")
print(report_counts.to_string())

# 8. Very small reports
print("\n--- REPORTS WITH < 50 SENTENCES ---")

small_reports = report_counts[report_counts < 50]

if len(small_reports) == 0:
    print("NONE")
else:
    print(small_reports.to_string())

# 9. Page validation
invalid_pages = (
    pd.to_numeric(df["page"], errors="coerce").isna().sum()
)

print("\n--- PAGE CHECK ---")
print(f"Invalid page values : {invalid_pages}")

# 10. Final result
print("\n===================================")

if (
    df["report_id"].nunique() == 50
    and empty_text == 0
    and duplicate_ids == 0
    and invalid_pages == 0
    and not missing
):
    print("DATASET BASIC VALIDATION : PASSED")
else:
    print("DATASET BASIC VALIDATION : NEEDS REVIEW")

print("===================================")