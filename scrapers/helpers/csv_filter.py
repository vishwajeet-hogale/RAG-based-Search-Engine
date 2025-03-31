import pandas as pd

# Load the CSV
df = pd.read_csv("../modules/data/summarized_publications_raw.csv")

# Define filter phrases
filters = [
    # "Verify you are human",
    # "Failed after retries",
    # "CAPTCHA triggered",
    "No summary",
    # "Summarization failed",
    # "ArXiv is committed to these values",
    # "ACL materials are Copyright",
    # "Skipped (PDF)",
    # "We value your privacy By clicking"
]

failure_mask = df["Publication Summary"].astype(str).apply(
    lambda x: any(phrase.lower() in x.lower() for phrase in filters)
)

blank_mask = df["Publication Summary"].isna() | (df["Publication Summary"].astype(str).str.strip() == "")

# Combine both
combined_mask = failure_mask | blank_mask

# Get rows with bad/missing summaries
bad_rows = df[combined_mask]
clean_rows = df[~combined_mask]

# 1. Create a copy to avoid SettingWithCopyWarning
bad_rows_updated = bad_rows.copy()
clean_rows_updated = clean_rows.copy()

# 2. Loop through bad rows
for idx, bad_row in bad_rows.iterrows():
    link = bad_row["Publication Link"]

    # Check if link exists in clean_rows
    matching_clean = clean_rows[clean_rows["Publication Link"] == link]

    if not matching_clean.empty:
        # Copy the row and update its summary
        new_row = bad_row.copy()
        new_row["Publication Summary"] = matching_clean.iloc[0]["Publication Summary"]

        # Append to clean_rows
        clean_rows_updated = pd.concat([clean_rows_updated, pd.DataFrame([new_row])], ignore_index=True)

        # Drop from bad_rows
        bad_rows_updated = bad_rows_updated[bad_rows_updated["Publication Link"] != link]

print(f"Transferred {len(bad_rows) - len(bad_rows_updated)} duplicates from bad to clean.")

# Optional: save updated versions
bad_rows_updated.to_csv("no_summary.csv", index=False)
# clean_rows_updated.to_csv("valid_paper_summaries_with_duplicates.csv", index=False)


print(f"Saved {len(bad_rows_updated)} bad or blank rows.")
# print(f"Saved {len(clean_rows_updated)} clean rows.")
