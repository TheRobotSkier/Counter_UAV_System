#!/usr/bin/env python3
"""
RTK–Local-Frame Calibration Script
----------------------------------

This script performs a full calibration between your local coordinate frame
(tetrahedron GPS mount positions) and global RTK measurements using 4 static
recordings (one per tetrahedron corner).

It does:

1. Load CSV files containing GNGGA messages
2. Extract lat/lon/alt
3. Convert RTK samples to ENU using pyproj (WGS84 → ENU)
4. Visualize RTK measurement spread for each corner
5. Compute rigid transform (rotation + translation) Local → ENU via SVD/Kabsch
6. Plot theoretical vs measured tetrahedron
7. Output transformation matrix
"""

# --------------------------------------------------------------------
# Standard imports
# --------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from pyproj import CRS, Transformer
from mpl_toolkits.mplot3d import Axes3D
import csv

# --------------------------------------------------------------------
# USER CONFIGURATION
# --------------------------------------------------------------------

# Path to your 4 recordings (change to your actual paths)
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

# Import your tetrahedron geometry (mm → meters)
import Transformations
P_LOCAL = 0.001 * Transformations.BASE_2_GPS_MIC_I   # shape (4,3)
print("Theoretical tetrahedron positions (local frame):\n", P_LOCAL)


# --------------------------------------------------------------------
# STEP 1 — NMEA GGA PARSER
# --------------------------------------------------------------------
def parse_gga(line):
    """
    Extract (lat, lon, alt) from a GNGGA NMEA sentence.
    Returns None if parsing fails.
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
# A fully PROJ-compliant pipeline that works everywhere
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
P_ENU = np.vstack([corner_means[f"corner{i}"] for i in range(1, 5)])
print("\nMeasured tetrahedron positions (ENU):\n", P_ENU)


# --------------------------------------------------------------------
# STEP 5 — PLOT SPREAD FOR EACH CORNER
# --------------------------------------------------------------------
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111, projection="3d")

colors = ["r","g","b","k"]

for (name, pts), c in zip(corner_data.items(), colors):
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=4, label=name, color=c)

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
    """Compute R, t such that:  Q ≈ R P + t   """
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
# EXTRA STEP — PLOT MEASUREMENT SPREAD IN LOCAL FRAME
# --------------------------------------------------------------------

def enu_to_local(p_enu):
    """Apply inverse transform: local = R^T * (enu - t)."""
    return R.T @ (p_enu - t)

fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111, projection="3d")

colors = ["r", "g", "b", "k"]

for (name, pts_enu), color in zip(corner_data.items(), colors):
    pts_local = np.array([enu_to_local(p) for p in pts_enu])
    ax.scatter(pts_local[:,0], pts_local[:,1], pts_local[:,2],
               s=4, label=name, color=color)

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

def enu_to_local(p_enu):
    """Apply inverse transform: local = R^T * (enu - t)."""
    return R.T @ (p_enu - t)

fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111, projection="3d")

corner_names = ["corner1", "corner2", "corner3", "corner4"]
colors = ["r", "g", "b", "k"]

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
ax.legend()
plt.tight_layout()
plt.show()
