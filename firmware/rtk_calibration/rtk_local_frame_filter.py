#!/usr/bin/env python3
"""
RTK Local-Frame Data Filter
---------------------------

Filters RTK data already converted into the local frame based on
RTK fix quality.

Input:
- CSV file with columns:
  timestamp_utc, local_x_m, local_y_m, local_z_m, fix_quality

Output:
- Filtered CSV with the same columns
- Only rows matching the allowed fix qualities are kept

No coordinate transformation is performed.
"""

# --------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------
import os
import csv

# --------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------

# Path to input CSV (local-frame RTK data)
INPUT_CSV = (
    "firmware/rtk_calibration/rtk_data_in_local_frame/day4/"
    "test_S3_rtk_log_in_local_frame_20251212_114031.csv"
)

# Output base directory
OUTPUT_BASE_DIR = (
    "firmware/rtk_calibration/rtk_data_in_local_frame/filtered"
)

# Allowed RTK fix qualities
# Examples:
#   {4}        → RTK Fixed only
#   {4, 5}     → RTK Fixed + RTK Float
#   {5}        → RTK Float only
ALLOWED_FIX_QUALITIES = {4, 5}

# Prefix added to output filename
OUTPUT_PREFIX = "filtered_"

# --------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------
def extract_test_day_from_path(path: str) -> int:
    """
    Extract test day number from a path containing 'dayX'.

    Example:
        ".../rtk_data_in_local_frame/day4/file.csv" → 4
    """
    import re

    match = re.search(r"/day(\d+)", path.replace("\\", "/"))
    if not match:
        raise ValueError(
            f"Could not extract test day from path: {path}"
        )

    return int(match.group(1))

# --------------------------------------------------------------------
# SET UP OUTPUT PATH
# --------------------------------------------------------------------
# Test day (used only for output folder structure)
TEST_DAY = extract_test_day_from_path(INPUT_CSV)

output_dir = os.path.join(OUTPUT_BASE_DIR, f"day{TEST_DAY}")
os.makedirs(output_dir, exist_ok=True)

input_filename = os.path.basename(INPUT_CSV)
output_filename = OUTPUT_PREFIX + input_filename
output_path = os.path.join(output_dir, output_filename)

# --------------------------------------------------------------------
# FILTERING PROCESS
# --------------------------------------------------------------------
total_rows = 0
kept_rows = 0

with open(INPUT_CSV, "r", encoding="utf-8") as infile, \
     open(output_path, "w", newline="", encoding="utf-8") as outfile:

    reader = csv.DictReader(infile)
    writer = csv.DictWriter(
        outfile,
        fieldnames=reader.fieldnames
    )

    # Write header
    writer.writeheader()

    for row in reader:
        total_rows += 1

        try:
            fix_quality = int(row["fix_quality"])
        except (KeyError, ValueError):
            continue

        if fix_quality in ALLOWED_FIX_QUALITIES:
            writer.writerow(row)
            kept_rows += 1

# --------------------------------------------------------------------
# SUMMARY
# --------------------------------------------------------------------
print("RTK Local-Frame Filtering Complete")
print("----------------------------------")
print(f"Input file        : {INPUT_CSV}")
print(f"Output file       : {output_path}")
print(f"Allowed qualities : {sorted(ALLOWED_FIX_QUALITIES)}")
print(f"Total rows        : {total_rows}")
print(f"Rows kept         : {kept_rows}")
print(f"Rows removed      : {total_rows - kept_rows}")
print(f"Kept percentage   : {100.0 * kept_rows / total_rows:.2f}%")