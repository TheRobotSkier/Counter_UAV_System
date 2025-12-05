#!/usr/bin/env python3
"""
doa_log_viewer.py

Utility script to visualize DOA logs produced by doa_logging_node.py.
Plots azimuth and elevation as a function of time.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def load_doa_log(csv_path: Path):
    times = []
    az = []
    el = []
    detected = []

    with csv_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) != 4:
                continue
            t, a, e, flag = parts
            times.append(float(t))
            az.append(float(a))
            el.append(float(e))
            detected.append(flag.lower() == "true")

    if not times:
        raise RuntimeError(f"No data rows found in {csv_path}")

    # Make times relative to first sample
    t0 = times[0]
    times = [t - t0 for t in times]
    return times, az, el, detected


def main():
    parser = argparse.ArgumentParser(
        description="Plot DOA azimuth/elevation from a doa_log.csv file."
    )
    parser.add_argument(
        "session_dir",
        help="Path to a doa_session_* directory (containing doa_log.csv)",
    )
    args = parser.parse_args()

    session_dir = Path(args.session_dir).expanduser().resolve()
    csv_path = session_dir / "doa_log.csv"

    if not csv_path.exists():
        raise SystemExit(f"Error: {csv_path} does not exist.")

    times, az, el, detected = load_doa_log(csv_path)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

    ax1.plot(times, az, label="Azimuth (deg)")
    ax1.set_ylabel("Azimuth [deg]")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(times, el, label="Elevation (deg)")
    ax2.set_ylabel("Elevation [deg]")
    ax2.set_xlabel("Time [s]")
    ax2.grid(True)
    ax2.legend()

    fig.suptitle(f"DOA Timeline\n{session_dir}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
