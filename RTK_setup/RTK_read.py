import pandas as pd
from pyproj import Proj, transform
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

# ---------------------------------------------------
# STEP 1: LOAD CSV
# ---------------------------------------------------
rows = []
path = "/home/gustav/Documents/PP_Viz/recordings/rtk_log_20251209_142531.csv"

with open(path, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith("local_time_utc"):
            continue  # skip header

        # Split off the timestamp from the rest
        try:
            local_time, rest = line.split(",", 1)
        except ValueError:
            continue

        # Find where $GNRMC starts
        idx = rest.find("$GNRMC")
        if idx == -1:
            # no RMC in this line, skip
            continue

        gga = rest[:idx].rstrip(",")   # everything before $GNRMC
        rmc = rest[idx:]               # from $GNRMC to end

        rows.append((local_time, gga, rmc))

# ---------------------------------------------------
# Helper: extract lat, lon, alt from GNGGA
# ---------------------------------------------------
def parse_gga(gga):
    """Extract decimal degrees lat, lon and altitude from GNGGA sentence."""
    if not isinstance(gga, str) or not gga.startswith("$GNGGA"):
        return None, None, None

    parts = gga.split(",")
    if len(parts) < 10:
        return None, None, None

    # latitude: DDMM.MMMMM
    raw_lat = parts[2]
    lat_dir = parts[3]
    # longitude: DDDMM.MMMMM
    raw_lon = parts[4]
    lon_dir = parts[5]

    try:
        # Convert NMEA to decimal degrees
        lat_deg = float(raw_lat[:2])
        lat_min = float(raw_lat[2:])
        lat = lat_deg + lat_min / 60.0
        if lat_dir == "S":
            lat = -lat

        lon_deg = float(raw_lon[:3])
        lon_min = float(raw_lon[3:])
        lon = lon_deg + lon_min / 60.0
        if lon_dir == "W":
            lon = -lon

        alt = float(parts[9])  # altitude (MSL)
    except:
        return None, None, None

    return lat, lon, alt

# ---------------------------------------------------
# STEP 2: Parse all rows
# ---------------------------------------------------
lats = []
lons = []
alts = []
times = []

for local_time, gga, rmc in rows:
    lat, lon, alt = parse_gga(gga)
    if lat is None:
        continue

    lats.append(lat)
    lons.append(lon)
    alts.append(alt)

    times.append(datetime.strptime(local_time, "%Y-%m-%dT%H:%M:%S.%fZ"))

print(f"Parsed {len(lats)} valid RTK points.")

# ---------------------------------------------------
# STEP 3: Use first point as reference
# ---------------------------------------------------
lat0 = lats[0]
lon0 = lons[0]
alt0 = alts[0]
t0   = times[0]

# pyproj ENU projection based on starting point
proj_enu = Proj(proj='aeqd', lat_0=lat0, lon_0=lon0, ellps='WGS84')

# ---------------------------------------------------
# STEP 4: Convert every lat/lon to ENU meters
# ---------------------------------------------------
east = []
north = []
up = []
dt_sec = []

for lat, lon, alt, t in zip(lats, lons, alts, times):
    e, n = proj_enu(lon, lat)  # pyproj expects lon, lat order
    east.append(e)
    north.append(n)
    up.append(alt - alt0)

    dt_sec.append((t - t0).total_seconds())

# ---------------------------------------------------
# STEP 5: 3D PLOT OF TRAJECTORY
# ---------------------------------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
print(f"Plotting {len(east)} trajectory points...")
ax.plot(east, north, up, marker='o')

ax.set_xlabel("x: East(m)")
ax.set_ylabel("y: North (m)")
ax.set_zlabel("z: Up (m)")

ax.set_title("Rover Trajectory (ENU Coordinates)")

# Equal scale for 3D axes
max_range = max(
    max(east) - min(east),
    max(north) - min(north),
    max(up) - min(up)
) / 2.0

mid_x = (max(east) + min(east)) / 2.0
mid_y = (max(north) + min(north)) / 2.0
mid_z = (max(up) + min(up)) / 2.0

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.show()

# ---------------------------------------------------
# OPTIONAL: save processed data
# ---------------------------------------------------
out = pd.DataFrame({
    "time_s": dt_sec,
    "east_m": east,
    "north_m": north,
    "up_m": up
})

out.to_csv("rtk_processed.csv", index=False)
print("Saved rtk_processed.csv")
