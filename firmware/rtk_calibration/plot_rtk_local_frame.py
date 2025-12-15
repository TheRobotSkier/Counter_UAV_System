#!/usr/bin/env python3
"""
RTK Local-Frame Visualization Script
====================================

This script loads a converted RTK CSV file containing positions in the
LOCAL frame and visualizes the 3D trajectory.

Each point is colored based on the RTK fix quality extracted from the
original GGA messages.

The script also prints a human-readable explanation of RTK fix quality
levels and their expected precision.
"""

# ====================================================================
# Imports
# ====================================================================
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import os


# ====================================================================
# USER CONFIGURATION
# ====================================================================

# Path to a converted RTK-local CSV file
INPUT_CSV = (
    "firmware/rtk_calibration/rtk_data_in_local_frame/day4/"
    "test_S3_rtk_log_in_local_frame_20251212_114031.csv"
)

# Marker size for scatter plot
POINT_SIZE = 8 # e.g. 1 = small, 10 = medium, 20 = large

# --------------------------------------------------------------------
# VISUALIZATION OPTIONS
# --------------------------------------------------------------------

SAME_ASPECT_RATIO = True         # Set equal scaling for all axes

PLOT_LOCAL_ORIGIN = True            # Plot (0,0,0)
PLOT_LOCAL_AXES = True              # Plot X/Y/Z arrows


ORIGIN_MARKER_SIZE = 120              # Size of origin marker
ORIGIN_MARKER_COLOR = "black"


AXIS_LENGTH = 5.0                   # Length of axes arrows [meters]
AXIS_LINEWIDTH = 2.0                 # Line width of axes arrows

# ====================================================================
# RTK FIX QUALITY DEFINITIONS
# ====================================================================

FIX_QUALITY_INFO = {
    0: {
        "name": "Invalid",
        "color": "gray",
        "precision": "No valid position"
    },
    1: {
        "name": "GPS Fix (Standalone)",
        "color": "red",
        "precision": "≈ 1–3 m"
    },
    2: {
        "name": "DGPS",
        "color": "orange",
        "precision": "≈ 0.3–1 m"
    },
    4: {
        "name": "RTK Fixed",
        "color": "green",
        "precision": "≈ 1–2 cm"
    },
    5: {
        "name": "RTK Float",
        "color": "blue",
        "precision": "≈ 5–30 cm"
    },
}


# ====================================================================
# LOAD CSV DATA
# ====================================================================
timestamps = []
positions = []
fix_qualities = []

with open(INPUT_CSV, "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        timestamps.append(row["timestamp_utc"])
        positions.append([
            float(row["local_x_m"]),
            float(row["local_y_m"]),
            float(row["local_z_m"]),
        ])
        fix_qualities.append(int(row["fix_quality"]))

positions = np.array(positions)
fix_qualities = np.array(fix_qualities)

if positions.size == 0:
    raise RuntimeError("No data points found in CSV file.")

print(f"\nLoaded {len(positions)} samples from:")
print(f"  {os.path.abspath(INPUT_CSV)}")


# ====================================================================
# PRINT FIX QUALITY EXPLANATION
# ====================================================================
print("\nRTK Fix Quality Meaning:")
print("------------------------")
for q, info in FIX_QUALITY_INFO.items():
    print(
        f"  {q}: {info['name']:<20} | "
        f"Color: {info['color']:<6} | "
        f"Expected precision: {info['precision']}"
    )


# ====================================================================
# 3D PLOT
# ====================================================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

for q, info in FIX_QUALITY_INFO.items():
    mask = fix_qualities == q
    if not np.any(mask):
        continue

    ax.scatter(
        positions[mask, 0],
        positions[mask, 1],
        positions[mask, 2],
        s=POINT_SIZE,
        color=info["color"],
        label=f"{q}: {info['name']}",
        alpha=0.7,
    )

ax.set_title("RTK Positions in Local Frame (Colored by Fix Quality)")
ax.set_xlabel("Local X [m]")
ax.set_ylabel("Local Y [m]")
ax.set_zlabel("Local Z [m]")

# Place legend outside plot for clarity
ax.legend(
    loc="center left",
    bbox_to_anchor=(1.05, 0.5),
    borderaxespad=0.0
)

# Set equal aspect ratio
def set_axes_equal_grounded(ax, z_min=None):
    """
    Set 3D plot axes to equal scale.
    X and Y are centered.
    Z is anchored so the minimum is at the bottom (ground-referenced).
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]

    max_range = max(x_range, y_range, z_range)

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)

    # Determine bottom of Z axis
    if z_min is None:
        z_min = z_limits[0]

    # Apply limits
    ax.set_xlim3d(x_middle - max_range / 2, x_middle + max_range / 2)
    ax.set_ylim3d(y_middle - max_range / 2, y_middle + max_range / 2)
    ax.set_zlim3d(z_min, z_min + max_range)


if SAME_ASPECT_RATIO:
    set_axes_equal_grounded(ax, z_min=0.0)

# --------------------------------------------------------------------
# Plot local-frame origin
# --------------------------------------------------------------------
if PLOT_LOCAL_ORIGIN:
    ax.scatter(
        0.0, 0.0, 0.0,
        s=ORIGIN_MARKER_SIZE,
        color=ORIGIN_MARKER_COLOR,
        marker="X",
        label="Local frame origin (0,0,0)"
    )
# --------------------------------------------------------------------
# Plot local-frame XYZ axes
# --------------------------------------------------------------------
if PLOT_LOCAL_AXES:
    ax.quiver(
        0, 0, 0,                 # origin
        AXIS_LENGTH, 0, 0,        # X axis
        color="red",
        linewidth=AXIS_LINEWIDTH,
        label="Local X"
    )

    ax.quiver(
        0, 0, 0,
        0, AXIS_LENGTH, 0,        # Y axis
        color="green",
        linewidth=AXIS_LINEWIDTH,
        label="Local Y"
    )

    ax.quiver(
        0, 0, 0,
        0, 0, AXIS_LENGTH,        # Z axis
        color="blue",
        linewidth=AXIS_LINEWIDTH,
        label="Local Z"
    )

plt.tight_layout()
plt.show()
