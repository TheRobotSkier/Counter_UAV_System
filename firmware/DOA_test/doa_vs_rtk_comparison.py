#!/usr/bin/env python3
"""
DOA vs RTK Ground Truth Comparison Script
----------------------------------------

This script compares Direction-of-Arrival (DOA) estimates against
ground-truth DOA computed from RTK positions in the local frame.

For each RTK sample (~1 Hz):
- Compute ground-truth azimuth & elevation
- Find the closest DOA estimate in time (~10–20 Hz)
- Compute DOA errors
- Save aligned data to a new CSV file

No filtering, smoothing, or interpolation is applied.
"""

# ============================================================
# Imports
# ============================================================
import os
import csv
import numpy as np
from datetime import datetime
import sys

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ============================================================
# CONFIGURATION
# ============================================================

TEST = "test_S3"
FILTERED = False   # Use filtered RTK data if True

MAX_TIME_OFFSET_S = 0.05  # Reject matches with |Δt| larger than this


# ------------------------------------------------------------
# DOA input files
# ------------------------------------------------------------
DOA_CSV_FILES_TEST_DAY_4 = {
    "test_L1": "src/cuav_system_logs/doa_session_2025-12-12_10-21-54-555092/doa_log.csv",
    "test_L2": "src/cuav_system_logs/doa_session_2025-12-12_10-28-33-665349/doa_log.csv",
    "test_L3": "src/cuav_system_logs/doa_session_2025-12-12_10-32-51-041192/doa_log.csv",
    "test_L4": "src/cuav_system_logs/doa_session_2025-12-12_10-53-29-403814/doa_log.csv",
    "test_L5": "src/cuav_system_logs/doa_session_2025-12-12_10-57-41-968739/doa_log.csv",
    "test_S1": "src/cuav_system_logs/doa_session_2025-12-12_11-27-30-182769/doa_log.csv",
    "test_S2": "src/cuav_system_logs/doa_session_2025-12-12_11-33-17-130679/doa_log.csv",
    "test_S3": "src/cuav_system_logs/doa_session_2025-12-12_11-41-24-062098/doa_log.csv",
    "test_S4": "src/cuav_system_logs/doa_session_2025-12-12_11-51-15-377311/doa_log.csv",
    "test_S5": "src/cuav_system_logs/doa_session_2025-12-12_11-56-21-277689/doa_log.csv",
    "test_S6": "src/cuav_system_logs/doa_session_2025-12-12_12-00-41-044424/doa_log.csv",
}

RTK_BASE_FOLDER = (
    "firmware/rtk_calibration/rtk_data_in_local_frame/filtered/day4"
    if FILTERED else
    "firmware/rtk_calibration/rtk_data_in_local_frame/day4"
)

OUTPUT_DIR = "firmware/DOA_test/doa_vs_rtk_comparison/day4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Helper functions
# ============================================================

def parse_timestamp(ts: str) -> float:
    """Convert ISO UTC timestamp to Unix seconds."""
    return datetime.fromisoformat(ts.replace("Z", "")).timestamp()


def cart2sph(cart):
    """
    Convert Cartesian coordinates to spherical DOA.

    Input:
        [x, y, z] (meters, local frame)

    Output:
        distance [m], azimuth [deg], elevation [deg]
    """
    x, y, z = cart
    dist = np.sqrt(x**2 + y**2 + z**2)
    az = np.degrees(np.arctan2(y, x)) % 360
    el = np.degrees(np.arctan2(z, np.hypot(x, y)))
    return dist, az, el


def angular_error_deg(a, b):
    """Smallest signed difference between two angles (degrees)."""
    return (a - b + 180) % 360 - 180


# ============================================================
# Load RTK data (local frame)
# ============================================================

rtk_file = next(
    f for f in os.listdir(RTK_BASE_FOLDER)
    if f.startswith(TEST) and f.endswith(".csv")
)

rtk_path = os.path.join(RTK_BASE_FOLDER, rtk_file)

rtk_times = []
rtk_positions = []

with open(rtk_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rtk_times.append(parse_timestamp(row["timestamp_utc"]))
        rtk_positions.append([
            float(row["local_x_m"]),
            float(row["local_y_m"]),
            float(row["local_z_m"]),
        ])

rtk_times = np.array(rtk_times)
rtk_positions = np.array(rtk_positions)

# ============================================================
# Load DOA data (Option A: skip comment lines)
# ============================================================

doa_path = DOA_CSV_FILES_TEST_DAY_4[TEST]

doa_times = []
doa_az = []
doa_el = []

with open(doa_path, "r", encoding="utf-8") as f:
    # Remove comment lines (starting with '#')
    lines = [line for line in f if not line.lstrip().startswith("#")]

reader = csv.DictReader(lines)

for row in reader:
    try:
        doa_times.append(float(row["timestamp"]))   # Unix seconds
        doa_az.append(float(row["azimuth"]))
        doa_el.append(float(row["elevation"]))
    except (KeyError, ValueError):
        continue

doa_times = np.array(doa_times)
doa_az = np.array(doa_az)
doa_el = np.array(doa_el)

# ============================================================
# Alignment & comparison
# ============================================================

aligned_rows = []

rejected_time_offset = 0
total_rtk_samples = len(rtk_times)

for t_rtk, pos in zip(rtk_times, rtk_positions):

    # Ground-truth DOA from RTK position
    _, gt_az, gt_el = cart2sph(pos)

    # Find closest DOA timestamp
    idx = np.argmin(np.abs(doa_times - t_rtk))

    dt = doa_times[idx] - t_rtk

    # ------------------------------------------------------------
    # Time-offset gating
    # ------------------------------------------------------------
    if abs(dt) > MAX_TIME_OFFSET_S:
        rejected_time_offset += 1
        continue  # Reject this RTK–DOA pair

    az_err = angular_error_deg(doa_az[idx], gt_az)
    el_err = doa_el[idx] - gt_el

    aligned_rows.append([
        datetime.utcfromtimestamp(t_rtk).isoformat() + "Z",
        pos[0], pos[1], pos[2],
        gt_az, gt_el,
        datetime.utcfromtimestamp(doa_times[idx]).isoformat() + "Z",
        doa_az[idx], doa_el[idx],
        az_err, el_err,
        dt
    ])

print("\nAlignment summary")
print("-----------------")
print(f"Total RTK samples           : {total_rtk_samples}")
print(f"Rejected (|Δt| > {MAX_TIME_OFFSET_S:.3f}s) : {rejected_time_offset}")
print(f"Accepted aligned samples    : {len(aligned_rows)}")

if total_rtk_samples > 0:
    rejection_pct = 100.0 * rejected_time_offset / total_rtk_samples
    print(f"Rejection rate              : {rejection_pct:.1f}%")

# ============================================================
# Save aligned comparison CSV
# ============================================================

output_path = os.path.join(OUTPUT_DIR, f"doa_vs_rtk_{TEST}.csv")

with open(output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "rtk_timestamp",
        "rtk_x_m", "rtk_y_m", "rtk_z_m",
        "gt_az_deg", "gt_el_deg",
        "doa_timestamp",
        "doa_az_deg", "doa_el_deg",
        "az_error_deg", "el_error_deg",
        "time_offset_s"
    ])
    writer.writerows(aligned_rows)

print(f"\nSaved aligned DOA comparison to:\n{output_path}")
