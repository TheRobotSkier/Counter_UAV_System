#!/usr/bin/env python3
"""
DOA Analysis Script
==================

Analyzes aligned DOA vs RTK ground-truth data produced by
doa_vs_rtk_comparison.py.

Produces:
1) 3D RTK positions colored by DOA error
2) DOA error vs distance
3) Error distributions (histograms + CDF)
4) Azimuth error vs elevation
5) DOA error time series

Also prints a minimal "gold standard" performance summary.

Designed to be easily extendable to Tier-3 plots.
"""

# ============================================================
# Imports
# ============================================================
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_CSV = (
    "firmware/DOA_test/doa_vs_rtk_comparison/day4/"
    "doa_vs_rtk_test_S3.csv"
)


OUTPUT_DIR = "firmware/DOA_test/doa_analysis/day4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# DOA estimator search grid (IMPORTANT FOR INTERPRETATION)
AZ_MIN, AZ_MAX = 0.0, 360.0
EL_MIN, EL_MAX = 0.0, 90.0

# ============================================================
# Helper functions
# ============================================================

def angular_error_deg(a, b):
    """Smallest signed angular difference (degrees)."""
    return (a - b + 180) % 360 - 180


def load_csv(path):
    """Load aligned DOA vs RTK CSV into structured arrays."""
    data = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def parse_iso(ts):
    return datetime.fromisoformat(ts.replace("Z", "")).timestamp()

# ============================================================
# Load data
# ============================================================

rows = load_csv(INPUT_CSV)

rtk_xyz = np.array([
    [float(r["rtk_x_m"]), float(r["rtk_y_m"]), float(r["rtk_z_m"])]
    for r in rows
])

rtk_time = np.array([parse_iso(r["rtk_timestamp"]) for r in rows])
doa_time = np.array([parse_iso(r["doa_timestamp"]) for r in rows])

gt_az = np.array([float(r["gt_az_deg"]) for r in rows])
gt_el = np.array([float(r["gt_el_deg"]) for r in rows])

est_az = np.array([float(r["doa_az_deg"]) for r in rows])
est_el = np.array([float(r["doa_el_deg"]) for r in rows])

az_err = np.array([float(r["az_error_deg"]) for r in rows])
el_err = np.array([float(r["el_error_deg"]) for r in rows])

time_offset = np.array([float(r["time_offset_s"]) for r in rows])

# Derived metrics
distance = np.linalg.norm(rtk_xyz, axis=1)
total_ang_err = np.sqrt(az_err**2 + el_err**2)

# ============================================================
# Boundary diagnostics (DOA grid saturation)
# ============================================================

on_el_boundary = (est_el <= EL_MIN + 1e-6) | (est_el >= EL_MAX - 1e-6)
boundary_ratio = np.mean(on_el_boundary)

# ============================================================
# GOLD-STANDARD METRICS
# ============================================================

def pct(x, p):
    return np.percentile(x, p)

print("\n================ DOA PERFORMANCE SUMMARY ================\n")
print(f"Total samples            : {len(total_ang_err)}")
print(f"Median angular error     : {pct(total_ang_err, 50):.2f} deg")
print(f"90th percentile error    : {pct(total_ang_err, 90):.2f} deg")
print(f"95th percentile error    : {pct(total_ang_err, 95):.2f} deg")
print(f"Max angular error        : {np.max(total_ang_err):.2f} deg")
print(f"Median distance          : {pct(distance, 50):.2f} m")
print(f"Boundary elevation hits  : {boundary_ratio*100:.1f} %")

if boundary_ratio > 0.1:
    print("⚠️  WARNING: Significant fraction of DOA estimates hit elevation grid boundary")

print("\n=========================================================\n")

# ============================================================
# PLOT 1 — 3D RTK positions colored by DOA error
# ============================================================

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

sc = ax.scatter(
    rtk_xyz[:,0], rtk_xyz[:,1], rtk_xyz[:,2],
    c=total_ang_err, cmap="viridis", s=15
)

ax.set_title("RTK Ground Truth Positions Colored by DOA Error")
ax.set_xlabel("Local X [m]")
ax.set_ylabel("Local Y [m]")
ax.set_zlabel("Local Z [m]")
plt.colorbar(sc, label="Total DOA Error [deg]")
plt.tight_layout()
plt.show()

# ============================================================
# PLOT 2 — DOA error vs distance
# ============================================================

plt.figure(figsize=(7, 4))
plt.scatter(distance, total_ang_err, s=12, alpha=0.6)
plt.xlabel("Distance from Origin [m]")
plt.ylabel("Total DOA Error [deg]")
plt.title("DOA Error vs Distance")
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# PLOT 3 — Error distributions (hist + CDF)
# ============================================================

plt.figure(figsize=(7, 4))
plt.hist(total_ang_err, bins=50, density=True, alpha=0.7)
plt.xlabel("Total DOA Error [deg]")
plt.ylabel("Probability Density")
plt.title("DOA Error Distribution")
plt.tight_layout()
plt.show()

sorted_err = np.sort(total_ang_err)
cdf = np.arange(1, len(sorted_err)+1) / len(sorted_err)

plt.figure(figsize=(7, 4))
plt.plot(sorted_err, cdf)
plt.xlabel("Total DOA Error [deg]")
plt.ylabel("CDF")
plt.title("DOA Error CDF")
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# PLOT 4 — Azimuth error vs elevation
# ============================================================

plt.figure(figsize=(7, 4))
plt.scatter(gt_el, az_err, s=12, alpha=0.6)
plt.xlabel("Ground Truth Elevation [deg]")
plt.ylabel("Azimuth Error [deg]")
plt.title("Azimuth Error vs Elevation")
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# PLOT 5 — DOA error time series
# ============================================================

t_rel = rtk_time - rtk_time[0]

plt.figure(figsize=(8, 4))
plt.plot(t_rel, total_ang_err, linewidth=1)
plt.xlabel("Time [s]")
plt.ylabel("Total DOA Error [deg]")
plt.title("DOA Error Over Time")
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# EXTENSION POINT — Tier-3 plots
# ============================================================

# Examples to add here later:
# - Error cones in 3D
# - Angular residual vectors
# - Covariance ellipses
# - RMSE vs SNR proxy
# - Error heatmaps in az/el space
