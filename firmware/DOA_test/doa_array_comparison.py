#!/usr/bin/env python3
"""
DOA Array-Level Performance Comparison
======================================

Aggregates DOA vs RTK ground-truth comparison results across
multiple tests to evaluate overall array performance.

Compares:
- Large microphone array
- Small microphone array

This script operates on the output of:
    doa_vs_rtk_comparison.py

It produces:
- Aggregate statistics
- Distance-dependent metrics (0–40 m, >40 m)
- Optional plots for system-level evaluation
"""

# ============================================================
# Imports
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION (EDIT THIS SECTION)
# ============================================================

# ------------------------------------------------------------
# Input CSVs
# ------------------------------------------------------------
INPUT_CSV_L = {
    "L1": "firmware/DOA_test/doa_vs_rtk_comparison/day4/doa_vs_rtk_test_L1.csv",
    "L2": "firmware/DOA_test/doa_vs_rtk_comparison/day4/doa_vs_rtk_test_L2.csv",
    "L3": "firmware/DOA_test/doa_vs_rtk_comparison/day4/doa_vs_rtk_test_L3.csv",
    "L4": "firmware/DOA_test/doa_vs_rtk_comparison/day4/doa_vs_rtk_test_L4.csv",
    "L5": "firmware/DOA_test/doa_vs_rtk_comparison/day4/doa_vs_rtk_test_L5.csv",
}

INPUT_CSV_S = {
    "S1": "firmware/DOA_test/doa_vs_rtk_comparison/day4/doa_vs_rtk_test_S1.csv",
    "S3": "firmware/DOA_test/doa_vs_rtk_comparison/day4/doa_vs_rtk_test_S3.csv",
    "S5": "firmware/DOA_test/doa_vs_rtk_comparison/day4/doa_vs_rtk_test_S5.csv",
    "S6": "firmware/DOA_test/doa_vs_rtk_comparison/day4/doa_vs_rtk_test_S6.csv",
}

# ------------------------------------------------------------
# Distance binning (meters)
# ------------------------------------------------------------
DISTANCE_THRESHOLD_M = 50.0
DISTANCE_BINS = [0, DISTANCE_THRESHOLD_M, np.inf]
DISTANCE_BIN_LABELS = [
    f"0–{DISTANCE_THRESHOLD_M:.0f} m",
    f">{DISTANCE_THRESHOLD_M:.0f} m",
]



# Minimum Z height (meters); set to None to disable
MIN_Z_M = 3.0

# ------------------------------------------------------------
# Output options
# ------------------------------------------------------------
PRINT_SUMMARY_TABLES = True
PRINT_DISTANCE_TABLES = True
PRINT_PER_TEST_STATS = True
PRINT_PARTICLE_FILTER_NOISE_SUMMARY = True

PLOT_ERROR_VS_DISTANCE = True
PLOT_ERROR_HISTOGRAMS = True
PLOT_DISTANCE_BOXPLOTS = True

# Plot appearance
FIGSIZE = (10, 6)
SCATTER_ALPHA = 0.25
HIST_BINS = 40

# ============================================================
# Helper functions
# ============================================================

def load_csvs(csv_dict, array_label):
    """
    Load multiple doa_vs_rtk CSV files and tag them with array/test info.
    """
    rows = []

    for test_name, path in csv_dict.items():
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        df = pd.read_csv(path)

        for _, r in df.iterrows():
            rows.append({
                "array": array_label,
                "test": test_name,
                "distance": np.sqrt(
                    r["rtk_x_m"]**2 +
                    r["rtk_y_m"]**2 +
                    r["rtk_z_m"]**2
                ),
                "rtk_z_m": r["rtk_z_m"],
                "az_error": r["az_error_deg"],
                "el_error": r["el_error_deg"],
                "total_error": np.sqrt(
                    r["az_error_deg"]**2 +
                    r["el_error_deg"]**2
                ),
            })

    return pd.DataFrame(rows)


def summary_stats(series):
    """
    Compute simple summary statistics.
    """
    return {
        "count": len(series),
        "median": np.median(series),
        "mean": np.mean(series),
        "std": np.std(series),
        "p90": np.percentile(series, 90),
    }

# ============================================================
# Load all data
# ============================================================

df_large = load_csvs(INPUT_CSV_L, "Large")
df_small = load_csvs(INPUT_CSV_S, "Small")
df_all = pd.concat([df_large, df_small], ignore_index=True)

# ------------------------------------------------------------
# Apply minimum Z-height filtering (if enabled)
# ------------------------------------------------------------
if MIN_Z_M is not None:
    df_all = df_all[df_all["distance"].notna()]  # safety
    df_all = df_all[df_all["rtk_z_m"] >= MIN_Z_M]


# ============================================================
# GOLD-STANDARD SUMMARY TABLE
# ============================================================

if PRINT_SUMMARY_TABLES:
    print("\n================ OVERALL PERFORMANCE =================")

    for label, df in [("Large Array", df_large), ("Small Array", df_small)]:
        print(f"\n{label}")

        for err_label, col in [
            ("Total", "total_error"),
            ("Azimuth", "az_error"),
            ("Elevation", "el_error"),
        ]:
            s = summary_stats(df[col])
            print(
                f"  {err_label:<9}: "
                f"median={s['median']:.2f}°, "
                f"mean={s['mean']:.2f}°, "
                f"std={s['std']:.2f}°, "
                f"n={s['count']}"
            )

# ============================================================
# DISTANCE-BINNED STATISTICS
# ============================================================

df_all["distance_bin"] = pd.cut(
    df_all["distance"],
    DISTANCE_BINS,
    labels=DISTANCE_BIN_LABELS,
    right=False
)

if PRINT_DISTANCE_TABLES:
    print("\n================ DISTANCE-BINNED PERFORMANCE =================")

    for array in ["Large", "Small"]:
        print(f"\n{array} Array")
        df_arr = df_all[df_all["array"] == array]

        for bin_label in DISTANCE_BIN_LABELS:
            d = df_arr[df_arr["distance_bin"] == bin_label]["total_error"]
            if len(d) == 0:
                continue
            s = summary_stats(d)
            print(
                f"  {bin_label}: "
                f"median={s['median']:.2f}°, "
                f"mean={s['mean']:.2f}°, "
                f"std={s['std']:.2f}°, "
                f"n={s['count']}"
            )

# ============================================================
# PER-TEST CONSISTENCY
# ============================================================

if PRINT_PER_TEST_STATS:
    print("\n================ PER-TEST ERRORS =================")

    for array in ["Large", "Small"]:
        print(f"\n{array} Array")
        df_arr = df_all[df_all["array"] == array]

        for test in sorted(df_arr["test"].unique()):
            d = df_arr[df_arr["test"] == test]["total_error"]
            print(
                f"  {test}: "
                f"median={np.median(d):.2f}°, "
                f"mean={np.mean(d):.2f}°, "
                f"std={np.std(d):.2f}°"
            )

# ============================================================
# PARTICLE FILTER NOISE SUMMARY
# ============================================================
def print_pf_stats(label, df):
    print(f"\n{label}")
    for name, subset in [
        ("All distances", df),
        (f"≤{DISTANCE_THRESHOLD_M:.0f} m", df[df["distance"] <= DISTANCE_THRESHOLD_M]),
        (f">{DISTANCE_THRESHOLD_M:.0f} m", df[df["distance"] > DISTANCE_THRESHOLD_M]),
    ]:
        if len(subset) == 0:
            continue

        print(f"  {name}")
        for err_name, col in [
            ("Total", "total_error"),
            ("Azimuth", "az_error"),
            ("Elevation", "el_error"),
        ]:
            print(
                f"    {err_name:<9}: "
                f"mean={np.mean(subset[col]):.3f}°, "
                f"std={np.std(subset[col]):.3f}°"
            )

if PRINT_PARTICLE_FILTER_NOISE_SUMMARY:
    print("\n================ PARTICLE FILTER NOISE SUMMARY =================")

    print_pf_stats("Large Array", df_large)
    print_pf_stats("Small Array", df_small)


# ============================================================
# PLOTS
# ============================================================

# ------------------------------------------------------------
# Plot 1 — Error vs distance
# ------------------------------------------------------------
if PLOT_ERROR_VS_DISTANCE:
    plt.figure(figsize=FIGSIZE)

    for label, df, color in [
        ("Large Array", df_large, "tab:blue"),
        ("Small Array", df_small, "tab:orange"),
    ]:
        plt.scatter(
            df["distance"],
            df["total_error"],
            s=12,
            alpha=SCATTER_ALPHA,
            label=label,
            color=color,
        )

    plt.xlabel("Distance [m]")
    plt.ylabel("Total DOA Error [deg]")
    plt.title("DOA Error vs Distance")
    plt.legend()
    plt.grid(True)

    plt.ylim(bottom=0)

    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# Plot 2 — Error histograms
# ------------------------------------------------------------
if PLOT_ERROR_HISTOGRAMS:
    plt.figure(figsize=FIGSIZE)

    plt.hist(
        df_large["total_error"],
        bins=HIST_BINS,
        density=True,
        alpha=0.6,
        label="Large Array",
    )
    plt.hist(
        df_small["total_error"],
        bins=HIST_BINS,
        density=True,
        alpha=0.6,
        label="Small Array",
    )

    plt.xlabel("Total DOA Error [deg]")
    plt.ylabel("Probability Density")
    plt.title("DOA Error Distribution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# Plot 3 — Distance-binned boxplots (0–50 m, >50 m)
# ------------------------------------------------------------
if PLOT_DISTANCE_BOXPLOTS:
    fig, ax = plt.subplots(figsize=FIGSIZE)

    data = []
    labels = []

    for bin_label in DISTANCE_BIN_LABELS:
        for array in ["Large", "Small"]:
            vals = df_all[
                (df_all["array"] == array) &
                (df_all["distance_bin"] == bin_label)
            ]["total_error"]

            data.append(vals)
            labels.append(f"{array}\n{bin_label}")

    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel("Total DOA Error [deg]")
    ax.set_title("DOA Error by Distance Bin")
    ax.grid(True)
    ax.set_ylim(bottom=-0.5)
    plt.tight_layout()
    plt.show()
