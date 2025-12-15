#!/usr/bin/env python3
"""
RTK–Local-Frame Calibration Script
----------------------------------

This script calibrates the rigid transformation between a local coordinate
frame (defined by known GPS antenna mount positions on a calibration
tetrahedron) and RTK-derived global positions.

Static RTK recordings are taken with the GPS receiver placed at known,
fixed positions (one per tetrahedron corner, optionally with extra points).
These measurements are used to estimate the rotation and translation that
map the local frame to the RTK ENU frame centered at the base station.

The resulting calibration enables conversion of arbitrary RTK measurements
into the local frame, providing ground-truth positions for evaluating other
sensors (e.g. LiDAR-based UAV tracking).

Main steps:
1. Load CSV files containing GNGGA messages
2. Extract latitude, longitude, and altitude
3. Convert RTK samples to ENU coordinates (WGS84 → ENU at base station)
4. Visualize RTK measurement spread for each corner
5. Compute rigid transform (rotation + translation) using SVD / Kabsch
6. Evaluate alignment errors in the local frame
7. Save calibration parameters to JSON for later reuse
"""

# --------------------------------------------------------------------
# Standard imports
# --------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Transformer
import os
import sys
import json
import re
from datetime import datetime

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),"..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --------------------------------------------------------------------
# OUTPUT DIRECTORY
# --------------------------------------------------------------------
CALIBRATION_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "test_calibrations"
)

os.makedirs(CALIBRATION_OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------------------------
# USER CONFIGURATION
# --------------------------------------------------------------------
# Select which calibration dataset to use
# 1 = Drone club (09/12)
# 2 = UNI (10/12)
# 3 = Fårup (11/12)
# 4 = Drone club (12/12)
TEST_DAY = 3

# Include additional calibration points (corner1A, corner2A) if available
EXTRA_CORNERS = True



if TEST_DAY==1:
    # Path to recordings
    CSV_FILES = {
        "corner1": "src/cuav_system_logs/Calibration_files/rtk_log_20251209_114932.csv",
        "corner2": "src/cuav_system_logs/Calibration_files/rtk_log_20251209_114454.csv",
        "corner3": "src/cuav_system_logs/Calibration_files/rtk_log_20251209_115408.csv",
        "corner4": "src/cuav_system_logs/Calibration_files/rtk_log_20251209_115234.csv",
    }
    # RTK base station coordinates (WGS84)
    BASE_LAT = 57.063907500
    BASE_LON = 10.031707600
    BASE_ALT = 40.4000

elif TEST_DAY==2:
    CSV_FILES = {}

elif TEST_DAY==3:
    # Path to recordings 
    CSV_FILES = {
        "corner1": "src/cuav_system_logs/11_12_recordings/rtk_log_20251211_111526.csv",
        "corner2": "src/cuav_system_logs/11_12_recordings/rtk_log_20251211_111745.csv",
        "corner3": "src/cuav_system_logs/11_12_recordings/rtk_log_20251211_111942.csv",
        "corner4": "src/cuav_system_logs/11_12_recordings/rtk_log_20251211_112203.csv",
        "corner1A": "src/cuav_system_logs/11_12_recordings/rtk_log_20251211_112504.csv",
        "corner2A": "src/cuav_system_logs/11_12_recordings/rtk_log_20251211_112709.csv",
    }
    # RTK base station coordinates (WGS84)
    BASE_LAT = 57.266970000
    BASE_LON = 9.6449271
    BASE_ALT = 50.9000
    

elif TEST_DAY==4:
    CSV_FILES = {
        "corner1": "src/cuav_system_logs/Calibration_files/12_12_RTK_Calibration/RTK_Calibration/rtk_log_20251212_093054.csv",
        "corner2": "src/cuav_system_logs/Calibration_files/12_12_RTK_Calibration/RTK_Calibration/rtk_log_20251212_093307.csv",
        "corner3": "src/cuav_system_logs/Calibration_files/12_12_RTK_Calibration/RTK_Calibration/rtk_log_20251212_093600.csv",
        "corner4": "src/cuav_system_logs/Calibration_files/12_12_RTK_Calibration/RTK_Calibration/rtk_log_20251212_093807.csv",
        "corner1A": "src/cuav_system_logs/Calibration_files/12_12_RTK_Calibration/RTK_Calibration/rtk_log_20251212_094024.csv",
        "corner2A": "src/cuav_system_logs/Calibration_files/12_12_RTK_Calibration/RTK_Calibration/rtk_log_20251212_094203.csv",
    }
    # RTK base station coordinates (WGS84)
    BASE_LAT = 57.06390633
    BASE_LON = 10.03170450
    BASE_ALT = 40.200

else:
    raise RuntimeError("Invalid TEST_DAY value")

if EXTRA_CORNERS:
    CORNER_NAMES = [
        "corner1",
        "corner2",
        "corner3",
        "corner4",
        "corner1A",
        "corner2A",
    ]
else:
    CORNER_NAMES = [
        "corner1",
        "corner2",
        "corner3",
        "corner4",
    ]





# Import local-frame geometry of GPS antenna positions
# (mm → meters)
import Transformations
P_LOCAL = 0.001 * Transformations.BASE_2_GPS_MIC_I   # shape (4,3)
if EXTRA_CORNERS:
    # Append extra corners 1A and 2A
    P_LOCAL = np.vstack([                             # shape (6,3)
        P_LOCAL,
        0.001 * Transformations.BASE_2_CORNER_1A_GPS, # shape (1,3)
        0.001 * Transformations.BASE_2_CORNER_2A_GPS  # shape (1,3)
    ])
print("Theoretical tetrahedron positions (local frame):\n", P_LOCAL)

# Sanity check
assert P_LOCAL.shape[0] == len(CORNER_NAMES)


# --------------------------------------------------------------------
# STEP 1 — NMEA GGA PARSER
# --------------------------------------------------------------------
def parse_gga(line):
    """
    Extract (lat, lon, alt) from a GNGGA NMEA sentence.
    Returns None if parsing fails.

    Note:
    - Latitude/longitude are converted from NMEA DDMM.MMMMM format
    - Altitude is height above mean sea level (MSL)
    """
    if not line.startswith("$GNGGA"):
        return None

    parts = line.split(",")
    if len(parts) < 10:
        return None

    # ----- Parse latitude -----
    raw_lat = parts[2]
    lat_dir = parts[3]

    # NMEA: DDMM.MMMMM → DD + MM/60
    try:
        lat_deg = float(raw_lat[:2])
        lat_min = float(raw_lat[2:])
        lat = lat_deg + lat_min/60.0
        if lat_dir == "S":
            lat = -lat
    except:
        return None

    # ----- Parse longitude -----
    raw_lon = parts[4]
    lon_dir = parts[5]

    try:
        lon_deg = float(raw_lon[:3])
        lon_min = float(raw_lon[3:])
        lon = lon_deg + lon_min/60.0
        if lon_dir == "W":
            lon = -lon
    except:
        return None

    # ----- Altitude (meters above MSL) -----
    try:
        alt = float(parts[9])
    except:
        alt = None

    return lat, lon, alt


# --------------------------------------------------------------------
# STEP 2 — LOAD RTK CSV AND EXTRACT ALL LAT/LON/ALT
# --------------------------------------------------------------------
def load_rtk_csv(path):
    """
    Load an RTK CSV where each line contains:
    timestamp, gga sentence, rmc sentence

    Returns arrays: lat[], lon[], alt[]
    """
    lats, lons, alts = [], [], []

    with open(path, "r") as f:
        for line in f:
            if "$GNGGA" not in line:
                continue
            # Extract GGA substring
            parts = line.strip().split("$GNGGA")
            if len(parts) < 2:
                continue
            gga = "$GNGGA" + parts[1].split("$")[0]   # isolate the GGA part

            parsed = parse_gga(gga)
            if parsed is None:
                continue

            lat, lon, alt = parsed
            lats.append(lat)
            lons.append(lon)
            alts.append(alt)

    return np.array(lats), np.array(lons), np.array(alts)


# --------------------------------------------------------------------
# STEP 3 — SET UP PROJ TRANSFORMER (WGS84 → ENU)
# --------------------------------------------------------------------
# PROJ pipeline for converting WGS84 (lat, lon, alt) to local ENU coordinates
# centered at the RTK base station

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
    """Convert latitude, longitude, altitude → ENU meters using pyproj."""
    e, n, u = transformer.transform(lon, lat, alt)
    return np.array([e, n, u])


# --------------------------------------------------------------------
# STEP 4 — LOAD DATA FOR ALL FOUR CORNERS AND COMPUTE MEANS
# --------------------------------------------------------------------
corner_data = {}
corner_means = {}

for name, path in CSV_FILES.items():
    print(f"Loading: {name} from {path}")
    lats, lons, alts = load_rtk_csv(path)

    if len(lats) == 0:
        raise RuntimeError(f"No valid GNGGA data found in: {path}")

    enu_points = np.array([lla_to_enu(lat, lon, alt)
                           for lat, lon, alt in zip(lats, lons, alts)])

    corner_data[name] = enu_points
    corner_means[name] = np.mean(enu_points, axis=0)

# Convert dict to stacked matrix 4×3
P_ENU = np.vstack([corner_means[name] for name in CORNER_NAMES])
print("\nMeasured tetrahedron positions (ENU):\n", P_ENU)


# --------------------------------------------------------------------
# STEP 5 — PLOT SPREAD FOR EACH CORNER
# --------------------------------------------------------------------
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111, projection="3d")

colors = plt.cm.tab10.colors  # supports many corners

for name, color in zip(CORNER_NAMES, colors):
    pts = corner_data[name]
    ax.scatter(pts[:,0], pts[:,1], pts[:,2],
               s=4, label=name, color=color)

ax.set_title("RTK Measurement Spread for Each Tetrahedron Corner")
ax.set_xlabel("East [m]")
ax.set_ylabel("North [m]")
ax.set_zlabel("Up [m]")
ax.legend()
plt.tight_layout()
plt.show()


# --------------------------------------------------------------------
# STEP 6 — COMPUTE RIGID TRANSFORM USING KABSCH/SVD
# --------------------------------------------------------------------
def rigid_transform(P, Q):
    """Compute rigid transform using the Kabsch algorithm:
    solves Q ≈ R·P + t in a least-squares sense."""
    Pc = P - np.mean(P, axis=0)
    Qc = Q - np.mean(Q, axis=0)

    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure right-handed rotation
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = np.mean(Q, axis=0) - R @ np.mean(P, axis=0)
    return R, t

# Solve for transform that maps LOCAL frame → RTK ENU frame
R, t = rigid_transform(P_LOCAL, P_ENU)

print("\n------------------------------------")
print("LOCAL → ENU TRANSFORM SOLUTION")
print("------------------------------------")
print("Rotation matrix R:\n", R)
print("\nTranslation vector t:\n", t)


# --------------------------------------------------------------------
# STEP 7 — COMPUTE ALIGNMENT ERRORS
# --------------------------------------------------------------------
P_pred = (R @ P_LOCAL.T).T + t
errors = P_ENU - P_pred
rms = np.sqrt(np.mean(np.sum(errors**2, axis=1)))

print("\nAlignment RMS error (m):", rms)
print("Individual errors (m):\n", errors)

# --------------------------------------------------------------------
# STEP 7B — LOCAL-FRAME ERROR ANALYSIS (KEY METRIC)
# --------------------------------------------------------------------

print("\n------------------------------------")
print("LOCAL-FRAME CORNER ERRORS")
print("------------------------------------")

local_errors = []
local_error_norms = []

for i, name in enumerate(CORNER_NAMES):
    # Measured corner in ENU (mean)
    p_enu = P_ENU[i]

    # Transform to local frame
    p_local_est = R.T @ (p_enu - t)

    # Theoretical local position
    p_local_ref = P_LOCAL[i]

    # Error vector in LOCAL frame
    err_vec = p_local_est - p_local_ref
    err_norm = np.linalg.norm(err_vec)

    local_errors.append(err_vec)
    local_error_norms.append(err_norm)

    print(f"{name}:")
    print(f"  Estimated local : {p_local_est}")
    print(f"  Theoretical     : {p_local_ref}")
    print(f"  Error vector    : {err_vec}")
    print(f"  Error norm [m]  : {err_norm:.4f}")

local_errors = np.array(local_errors)
local_error_norms = np.array(local_error_norms)

rms_local = np.sqrt(np.mean(local_error_norms**2))

# RMS error here represents how well RTK-measured points align
# with the theoretical local geometry after calibration
print("\n------------------------------------")
print(f"LOCAL-FRAME RMS ERROR: {rms_local:.4f} m")
print("------------------------------------")


# --------------------------------------------------------------------
# STEP 8 — VISUALIZE MODEL VS MEASURED TETRAHEDRON
# --------------------------------------------------------------------
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(P_LOCAL[:,0], P_LOCAL[:,1], P_LOCAL[:,2],
           color="blue", s=80, label="Model (Local Frame)")

ax.scatter(P_ENU[:,0], P_ENU[:,1], P_ENU[:,2],
           color="red", s=80, label="Measured (RTK ENU)")

ax.set_title("Tetrahedron Calibration: Model vs RTK Measurements")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()

plt.tight_layout()
plt.show()


# --------------------------------------------------------------------
# STEP 9 — PRINT FINAL TRANSFORM AS 4×4 MATRIX
# --------------------------------------------------------------------
T = np.eye(4)
T[:3,:3] = R
T[:3, 3] = t

print("\nFinal 4×4 transform (Local → ENU):\n", T)
print("\nDone.")

# --------------------------------------------------------------------
# STEP 10 — SAVE CALIBRATION TO FILE (LOCAL → RTK ENU)
# --------------------------------------------------------------------

import json
import os
import re
from datetime import datetime

def extract_timestamp_from_filename(path):
    """
    Extract timestamp like YYYYMMDD_HHMMSS from RTK filename.
    """
    filename = os.path.basename(path)
    match = re.search(r"\d{8}_\d{6}", filename)
    if not match:
        raise ValueError(f"Could not extract timestamp from filename: {filename}")
    return match.group(0)

# Use corner1 as reference for calibration recording time
calibration_recording_timestamp = extract_timestamp_from_filename(
    CSV_FILES["corner1"]
)

# Convert to readable date
dt = datetime.strptime(calibration_recording_timestamp, "%Y%m%d_%H%M%S")
calibration_date_str = dt.strftime("%d-%m-%Y")

calibration_data = {
    "test_day": TEST_DAY,
    "calibration_recording_timestamp": calibration_recording_timestamp,
    "calibration_date": calibration_date_str,
    "base_station": {
        "lat": BASE_LAT,
        "lon": BASE_LON,
        "alt": BASE_ALT
    },
    "extra_corners": EXTRA_CORNERS,
    "corner_names": CORNER_NAMES,
    "rotation_matrix": R.tolist(),
    "translation_vector": t.tolist(),
    "rms_local_error_m": float(rms_local),
    "description": "Rigid transform (rotation + translation) mapping Local frame to RTK ENU frame"
}

output_filename = os.path.join(
    CALIBRATION_OUTPUT_DIR,
    f"calibration_data_test_day{TEST_DAY}_({calibration_date_str}).json"
)

with open(output_filename, "w") as f:
    json.dump(calibration_data, f, indent=4)

print("\n------------------------------------")
print(f"Calibration saved to: {output_filename}")
print("------------------------------------")



# --------------------------------------------------------------------
# EXTRA STEP — PLOT MEASUREMENT SPREAD IN LOCAL FRAME
# --------------------------------------------------------------------

def enu_to_local(p_enu):
    """Apply inverse transform: local = R^T * (enu - t)."""
    return R.T @ (p_enu - t)

fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111, projection="3d")

colors = plt.cm.tab10.colors

for name, color in zip(CORNER_NAMES, colors):
    pts_enu = corner_data[name]
    pts_local = np.array([enu_to_local(p) for p in pts_enu])
    ax.scatter(
        pts_local[:,0], pts_local[:,1], pts_local[:,2],
        s=4, label=name, color=color
    )

# Plot theoretical tetrahedron geometry
ax.scatter(P_LOCAL[:,0], P_LOCAL[:,1], P_LOCAL[:,2],
           color="cyan", s=80, label="Theoretical Positions", marker="^")

ax.set_title("RTK Measurement Spread in LOCAL Frame")
ax.set_xlabel("Local X [m]")
ax.set_ylabel("Local Y [m]")
ax.set_zlabel("Local Z [m]")
ax.legend()
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------
# EXTRA STEP — PLOT MEASUREMENT SPREAD IN LOCAL FRAME (WITH MATCHING COLORS)
# --------------------------------------------------------------------

fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111, projection="3d")

corner_names = CORNER_NAMES
colors = plt.cm.tab10.colors

for i, (corner, color) in enumerate(zip(corner_names, colors)):
    # Transform all ENU samples to local frame
    pts_enu = corner_data[corner]
    pts_local = np.array([enu_to_local(p) for p in pts_enu])
    
    # Plot measured points
    ax.scatter(
        pts_local[:,0], pts_local[:,1], pts_local[:,2],
        s=6, color=color, alpha=0.5,
        label=f"{corner} (measured)"
    )
    
    # Plot theoretical tetrahedron corner for this index
    ax.scatter(
        P_LOCAL[i,0], P_LOCAL[i,1], P_LOCAL[i,2],
        s=120, color=color, marker="^", edgecolors="k",
        label=f"{corner} (theoretical)"
    )

ax.set_title("RTK Measurement Spread in LOCAL Frame (Measured vs Theoretical)")
ax.set_xlabel("Local X [m]")
ax.set_ylabel("Local Y [m]")
ax.set_zlabel("Local Z [m]")
ax.legend(
    loc="center left",
    bbox_to_anchor=(1.1, 0.5),
    borderaxespad=0.0
)
plt.tight_layout()
plt.show()
