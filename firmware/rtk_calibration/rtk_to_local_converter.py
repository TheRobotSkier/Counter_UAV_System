#!/usr/bin/env python3
"""
RTK → Local Frame Conversion Script
==================================

This script converts RTK CSV logs containing NMEA GGA messages into
3D positions expressed in the system's LOCAL coordinate frame.

It uses a precomputed rigid calibration (rotation + translation)
that maps the LOCAL frame to the RTK ENU frame.

For each valid RTK sample, the output CSV contains:
    - timestamp_utc   : ISO-8601 UTC timestamp from the log
    - local_x_m       : Local-frame X position [meters]
    - local_y_m       : Local-frame Y position [meters]
    - local_z_m       : Local-frame Z position [meters]
    - fix_quality     : RTK fix quality from GGA (0–5)

Important:
- No filtering is performed
- No averaging is performed
- No calibration is computed here
- This script is a *pure coordinate conversion*

Intended use:
- Convert RTK ground truth into the same local frame as other sensors
- Enable direct accuracy/precision comparisons
"""

# ====================================================================
# Imports
# ====================================================================
import os
import json
import csv
import re
import numpy as np
from pyproj import Transformer


# ====================================================================
# USER CONFIGURATION
# ====================================================================

# Which calibration / test day to use
TEST_DAY = 4  # Options: 1, 2, 3, 4

# Output directory for converted CSV files
OUTPUT_DIR = f"firmware/rtk_calibration/rtk_data_in_local_frame/day{TEST_DAY}"

# If True, prepend batch key (e.g. "test_L1_") to output filenames
ADD_BATCH_NAME_PREFIX = True


# --------------------------------------------------------------------
# CSV batch definitions (grouped by test day)
# --------------------------------------------------------------------

CSV_FILES_TEST_DAY_1 = {
    "test_L1": "src/cuav_system_logs/09_12_RTK_Logs_Large_array_test/RTK_Logs/rtk_log_20251209_134423.csv",
    "test_L2": "src/cuav_system_logs/09_12_RTK_Logs_Large_array_test/RTK_Logs/rtk_log_20251209_134910.csv",
    "test_L3": "src/cuav_system_logs/09_12_RTK_Logs_Large_array_test/RTK_Logs/rtk_log_20251209_135447.csv",
    "test_L4": "src/cuav_system_logs/09_12_RTK_Logs_Large_array_test/RTK_Logs/rtk_log_20251209_144723.csv",
    "test_L5": "src/cuav_system_logs/09_12_RTK_Logs_Large_array_test/RTK_Logs/rtk_log_20251209_145318.csv",
    "test_S1": "src/cuav_system_logs/09_12_RTK_Logs_Small_array_test/rtk_log_20251209_141622.csv",
    "test_S2": "src/cuav_system_logs/09_12_RTK_Logs_Small_array_test/rtk_log_20251209_142034.csv",
    "test_S3": "src/cuav_system_logs/09_12_RTK_Logs_Small_array_test/rtk_log_20251209_142531.csv",
    "test_S4": "src/cuav_system_logs/09_12_RTK_Logs_Small_array_test/rtk_log_20251209_142943.csv",
}

CSV_FILES_TEST_DAY_2 = {}
CSV_FILES_TEST_DAY_3 = {
    "test_L1": "src/cuav_system_logs/11_12_recordings/rtk_log_20251211_154334.csv",
    "test_pp1": "src/cuav_system_logs/11_12_recordings/rtk_log_20251211_134935.csv",
    "test_pp4": "src/cuav_system_logs/11_12_recordings/rtk_log_20251211_143325.csv",
    "test_pp5": "src/cuav_system_logs/11_12_recordings/rtk_log_20251211_143928.csv",
    "test_pp6": "src/cuav_system_logs/11_12_recordings/rtk_log_20251211_144439.csv",
    "test_pp7": "src/cuav_system_logs/11_12_recordings/rtk_log_20251211_144941.csv",
    "test_pp8": "src/cuav_system_logs/11_12_recordings/rtk_log_20251211_145341.csv",
}

CSV_FILES_TEST_DAY_4 = {
    "test_L1": "src/cuav_system_logs/12_12_large_array_test/Large Array test/rtk_log_20251212_102125.csv",
    "test_L2": "src/cuav_system_logs/12_12_large_array_test/Large Array test/rtk_log_20251212_102810.csv",
    "test_L3": "src/cuav_system_logs/12_12_large_array_test/Large Array test/rtk_log_20251212_103224.csv",
    "test_L4": "src/cuav_system_logs/12_12_large_array_test/Large Array test/rtk_log_20251212_105303.csv",
    "test_L5": "src/cuav_system_logs/12_12_large_array_test/Large Array test/rtk_log_20251212_105725.csv",
    "test_S1": "src/cuav_system_logs/12_12_small_array_test/Small Array test/rtk_log_20251212_112651.csv",
    "test_S2": "src/cuav_system_logs/12_12_small_array_test/Small Array test/rtk_log_20251212_113250.csv",
    "test_S3": "src/cuav_system_logs/12_12_small_array_test/Small Array test/rtk_log_20251212_114031.csv",
    "test_S4": "src/cuav_system_logs/12_12_small_array_test/Small Array test/rtk_log_20251212_115046.csv",
    "test_S5": "src/cuav_system_logs/12_12_small_array_test/Small Array test/rtk_log_20251212_115545.csv",
    "test_S6": "src/cuav_system_logs/12_12_small_array_test/Small Array test/rtk_log_20251212_120015.csv",
    "test_fs2": "src/cuav_system_logs/12_12-System_RTK_Logs/rtk_log_20251212_143459.csv",
    "test_fs3": "src/cuav_system_logs/12_12-System_RTK_Logs/rtk_log_20251212_144440.csv",
    "test_ppp1": "src/cuav_system_logs/12_12-System_RTK_Logs/rtk_log_20251212_150354.csv",
    "test_ppp2": "src/cuav_system_logs/12_12-System_RTK_Logs/rtk_log_20251212_151200.csv",
    "test_ppp3": "src/cuav_system_logs/12_12-System_RTK_Logs/rtk_log_20251212_151903.csv",
    "test_fs4": "src/cuav_system_logs/12_12-System_RTK_Logs/rtk_log_20251212_152958.csv",
}


# --------------------------------------------------------------------
# Select batch + calibration based on TEST_DAY
# --------------------------------------------------------------------
if TEST_DAY == 1:
    CSV_BATCH = CSV_FILES_TEST_DAY_1
    CALIBRATION_FILE = "firmware/rtk_calibration/test_calibrations/calibration_data_test_day1_(09-12-2025).json"
elif TEST_DAY == 3:
    CSV_BATCH = CSV_FILES_TEST_DAY_3
    CALIBRATION_FILE = "firmware/rtk_calibration/test_calibrations/calibration_data_test_day3_(11-12-2025).json"
elif TEST_DAY == 4:
    CSV_BATCH = CSV_FILES_TEST_DAY_4
    CALIBRATION_FILE = "firmware/rtk_calibration/test_calibrations/calibration_data_test_day4_(12-12-2025).json"
else:
    raise ValueError(f"Unsupported TEST_DAY: {TEST_DAY}")


# ====================================================================
# LOAD CALIBRATION (Local → ENU)
# ====================================================================
with open(CALIBRATION_FILE, "r") as f:
    calib = json.load(f)

R = np.array(calib["rotation_matrix"])     # Local → ENU rotation
t = np.array(calib["translation_vector"])  # Local → ENU translation

BASE_LAT = calib["base_station"]["lat"]
BASE_LON = calib["base_station"]["lon"]
BASE_ALT = calib["base_station"]["alt"]


# ====================================================================
# SET UP WGS84 → ENU TRANSFORMER
# ====================================================================
transformer = Transformer.from_pipeline(
    f"""
    +proj=pipeline
    +step +proj=longlat +ellps=WGS84
    +step +proj=geocent +ellps=WGS84
    +step +proj=topocentric +ellps=WGS84
         +lat_0={BASE_LAT}
         +lon_0={BASE_LON}
         +h_0={BASE_ALT}
    """
)

def lla_to_enu(lat, lon, alt):
    """Convert latitude/longitude/altitude to ENU coordinates."""
    return np.array(transformer.transform(lon, lat, alt))

def enu_to_local(p_enu):
    """Apply inverse rigid transform: ENU → Local."""
    return R.T @ (p_enu - t)


# ====================================================================
# NMEA HELPERS
# ====================================================================
def parse_gga(gga: str):
    """
    Parse a $GNGGA or $GPGGA sentence.

    Returns:
        (lat_deg, lon_deg, alt_m, fix_quality)
        or None if parsing fails
    """
    if not isinstance(gga, str):
        return None

    gga = gga.strip()
    if gga.startswith("$GPGGA"):
        gga = "$GNGGA" + gga[len("$GPGGA"):]

    if not gga.startswith("$GNGGA"):
        return None

    parts = gga.split(",")
    if len(parts) < 10:
        return None

    try:
        lat = float(parts[2][:2]) + float(parts[2][2:]) / 60.0
        if parts[3] == "S":
            lat *= -1

        lon = float(parts[4][:3]) + float(parts[4][3:]) / 60.0
        if parts[5] == "W":
            lon *= -1

        alt = float(parts[9])
        fix_quality = int(parts[6])

        return lat, lon, alt, fix_quality
    except Exception:
        return None


def extract_timestamp_from_filename(path):
    """Extract YYYYMMDD_HHMMSS timestamp from filename."""
    m = re.search(r"\d{8}_\d{6}", path)
    return m.group(0) if m else "unknown"


def extract_gga_from_rest(rest: str):
    """
    Extract a complete GGA sentence from a CSV line fragment.
    Uses checksum (*XX) to detect sentence end.
    """
    start = rest.find("$GNGGA,")
    if start == -1:
        start = rest.find("$GPGGA,")
    if start == -1:
        return None

    s = rest[start:]
    star = s.find("*")
    if star == -1 or star + 3 > len(s):
        return None

    return s[: star + 3]


# ====================================================================
# MAIN CONVERSION LOOP
# ====================================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

for name, csv_path in CSV_BATCH.items():
    print(f"Processing: {name}")

    timestamp_tag = extract_timestamp_from_filename(csv_path)
    output_name = f"rtk_log_in_local_frame_{timestamp_tag}.csv"
    if ADD_BATCH_NAME_PREFIX:
        output_name = f"{name}_{output_name}"

    output_path = os.path.join(OUTPUT_DIR, output_name)

    rows_out = []

    with open(csv_path, "r", encoding="utf-8") as f:
        _ = f.readline()  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                timestamp, rest = line.split(",", 1)
            except ValueError:
                continue

            gga = extract_gga_from_rest(rest)
            if gga is None:
                continue

            parsed = parse_gga(gga)
            if parsed is None:
                continue

            lat, lon, alt, fix_q = parsed
            p_local = enu_to_local(lla_to_enu(lat, lon, alt))

            rows_out.append([
                timestamp,
                float(p_local[0]),
                float(p_local[1]),
                float(p_local[2]),
                fix_q,
            ])

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp_utc",
            "local_x_m",
            "local_y_m",
            "local_z_m",
            "fix_quality",
        ])
        writer.writerows(rows_out)

    print(f"  Saved {len(rows_out)} samples → {output_path}")

print("\nConversion complete.")
