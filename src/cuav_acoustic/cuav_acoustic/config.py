"""
config.py
Shared parameters for real-time and recording DOA processing.
Used by:
    - doa_node.py
    - doa_node_recording.py
    - doa_node_from_recording.py
"""

import numpy as np

# ==============================================
# Microphone Array Geometry
# ==============================================
D_M = 1.0 #0.213          # Edge length of the tetrahedral array [meters]
SPEED_OF_SOUND = 343.0  # Speed of sound in air [m/s]

def regular_tetrahedron_array(d_m: float) -> np.ndarray:
    """
    Return 4x3 array of mic positions (regular tetrahedron) centered at origin.
    d_m: edge length [m]
    """
    mic_1 = np.array([d_m * np.sqrt(3) / 3, 0.0, 0.0])
    mic_2 = np.array([-d_m * np.sqrt(3) / 6, 0.5 * d_m, 0.0])
    mic_3 = np.array([-d_m * np.sqrt(3) / 6, -0.5 * d_m, 0.0])
    mic_4 = np.array([0.0, 0.0, d_m * np.sqrt(6) / 3])

    mic_positions = np.array([mic_1, mic_2, mic_3, mic_4])
    return mic_positions

# Rotate the microphone array 180 degrees around the Z-axis?
ROTATE_180 = True   # Set True if array is mounted backwards

R_Z_180 = np.array([
    [-1,  0,  0],
    [  0, -1,  0],
    [  0,  0,  1]
])

# # Precomputed actual microphone positions (4×3) used by the DOA system:
if ROTATE_180:
    MIC_POSITIONS = (R_Z_180 @ regular_tetrahedron_array(D_M).T).T
else:
    MIC_POSITIONS = regular_tetrahedron_array(D_M)


# ==============================================
# Frame Parameters
# ==============================================
FRAME_DUR_SEC = 0.100     # 100 ms frame
OVERLAP_50 = True         # True → 50% overlap (hop = 50 ms)


# ==============================================
# SRP-PHAT Grid Parameters
# ==============================================
AZ_DEG_STEP = 1.0
EL_DEG_STEP = 1.0

# Search grid
AZIMUTHS = np.arange(0.0, 360.0, AZ_DEG_STEP)       # 0..359 deg
ELEVATIONS = np.arange(0.0, 90.0 + 1e-6, EL_DEG_STEP)  # 0..90 deg

# GCC-PHAT interpolation factor
INTERP_GCC = 16


# ==============================================
# Speed of sound configuration
# ==============================================
# Mode options:
#   "fixed"     → use FIXED_SPEED_OF_SOUND
#   "service"   → call ROS2 serial_interface_node.py
SPEED_OF_SOUND_MODE = "service"     # OR: "fixed"

# Fixed fallback value (used when SPEED_OF_SOUND_MODE = "fixed")
FIXED_SPEED_OF_SOUND = 343.0        # m/s (20 °C, 50% RH)

# Serial-based service name
SPEED_OF_SOUND_SERVICE_NAME = "/get_speed_of_sound"


# ==============================================
# Other Useful Settings
# ==============================================
# Channel configuration for real-time processing
NUM_CHANNELS = 4

# Timestamps / debug printing options (optional)
PRINT_DSP_TIMING = True
