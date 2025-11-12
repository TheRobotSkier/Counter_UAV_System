import numpy as np
import itertools
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
D_M = 1.0  # distance between microphones (meters)
SPEED_OF_SOUND = 343.0  # m/s

# Microphone coordinates (regular tetrahedron) centered at origin
MIC_1 = np.array([D_M*np.sqrt(3)/3, 0.0, 0.0])
MIC_2 = np.array([-D_M*np.sqrt(3)/6, 0.5*D_M, 0.0])
MIC_3 = np.array([-D_M*np.sqrt(3)/6, -0.5*D_M, 0.0])
MIC_4 = np.array([0.0, 0.0, D_M*np.sqrt(6)/3])

mics = [MIC_1, MIC_2, MIC_3, MIC_4]

# -----------------------------
# Input: sound source position
# -----------------------------
# Example position (you can change this)
source_pos = np.array([2.0, 1.0, 1.5])  # meters

# -----------------------------
# Compute distances and time delays
# -----------------------------
distances = np.array([np.linalg.norm(source_pos - mic) for mic in mics])
times = distances / SPEED_OF_SOUND

# -----------------------------
# Compute time differences between each microphone pair
# -----------------------------
# Generate all unique microphone pairs
    # Using itertools.combinations to avoid duplicate pairs
        # range(4) → creates the list [0, 1, 2, 3], representing the 4 microphones.
        # itertools.combinations(range(4), 2) → generates all unique 2-element combinations of those indices (without repetition, and order doesn’t matter).
pairs = list(itertools.combinations(range(4), 2)) # [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
time_differences = {}

for i, j in pairs:
    diff = times[i] - times[j]
    time_differences[f"MIC_{i+1}-MIC_{j+1}"] = diff

# -----------------------------
# Print results
# -----------------------------
# Print microphone distances
print("Microphone distances (m):")
for idx, d in enumerate(distances, 1):
    print(f"  MIC_{idx}: {d:.4f} m")

# Print absolute arrival times
print("\nArrival times (s):")
for idx, t in enumerate(times, 1):
    print(f"  MIC_{idx}: {t*1e6:.3f} µs")

# Print relative times (time delays w.r.t. first detection)
min_time = np.min(times)
relative_times = times - min_time
print("\nRelative arrival times (µs) (0 = first detection):")
for idx, t in enumerate(relative_times, 1):
    print(f"  MIC_{idx}: {t*1e6:.3f} µs")

# Print time differences between microphone pairs
print("\nTime differences between microphones:")
for k, v in time_differences.items():
    print(f"  {k}: {v*1e6:.3f} µs")

# -----------------------------
# Plot the array and source
# -----------------------------
def set_axes_equal(ax):
    """Set 3D plot axes to equal scale."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot origin
ax.scatter(0, 0, 0, c='k', marker='x', s=80, label='Origin')

# Plot microphones
mic_coords = np.array(mics)
ax.scatter(mic_coords[:,0], mic_coords[:,1], mic_coords[:,2], c='b', marker='o', s=80, label='Microphones')

# Plot source
ax.scatter(source_pos[0], source_pos[1], source_pos[2], c='r', marker='*', s=150, label='Sound Source')

# Connect microphones for visualizing the tetrahedron
edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
for i, j in edges:
    ax.plot([mic_coords[i,0], mic_coords[j,0]],
            [mic_coords[i,1], mic_coords[j,1]],
            [mic_coords[i,2], mic_coords[j,2]], 'k--', linewidth=0.8)

ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.legend()
ax.set_title('Tetrahedral Microphone Array Simulation')
ax.grid(True)
set_axes_equal(ax)
plt.show()


