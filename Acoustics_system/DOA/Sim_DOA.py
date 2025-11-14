
import numpy as np
from numpy.fft import rfft, irfft

# ---------------------------
# Parameters
# ---------------------------
D_M = 1.0  # distance between microphones (meters)
SPEED_OF_SOUND = 343.0  # m/s

# Microphone coordinates (regular tetrahedron) centered at origin
MIC_1 = np.array([D_M*np.sqrt(3)/3, 0.0, 0.0])
MIC_2 = np.array([-D_M*np.sqrt(3)/6, 0.5*D_M, 0.0])
MIC_3 = np.array([-D_M*np.sqrt(3)/6, -0.5*D_M, 0.0])
MIC_4 = np.array([0.0, 0.0, D_M*np.sqrt(6)/3])

# Search grid
AZ_DEG_STEP = 1.0     # azimuth step [deg]
EL_DEG_STEP = 1.0     # elevation step [deg]
AZIMUTHS = np.arange(0.0, 360.0, AZ_DEG_STEP)       # 0..360 exclusive
ELEVATIONS = np.arange(0.0, 90.0 + 1e-6, EL_DEG_STEP)  # 0..90 inclusive

# GCC-PHAT interpolation factor (higher -> finer delay resolution, more CPU)
INTERP_GCC = 16 

# -----------------------------
# Inputs:
# -----------------------------

SPEED_OF_SOUND = 343.0  # m/s

# Choose coordinate type:
CARTISIAN=False # True: cartisian (x, y, z); False: spherical (distance, azimuth, elevation)

# sound source position: cartisian coordinates (x, y, z) in meters:
source_pos_caert = np.array([2.0, 1.0, 1.5])  # meters

# sound source position: spherical coordinates (distance, azimuth, elevation)
source_pos_sph = np.array([2.5, 30.0, 20.0])  # meters, degrees, degrees


# -----------------------------
# Functions:
# -----------------------------

def sph2cart(sph_cord):
    """
    Input:
    np.array([distance, azimuth_deg, elevation_deg])

    Output:
    np.array([x, y, z])
    """
    distance, azimuth_deg, elevation_deg=sph_cord
    az_rad = np.radians(azimuth_deg)
    el_rad = np.radians(elevation_deg)
    x = distance * np.cos(el_rad) * np.cos(az_rad)
    y = distance * np.cos(el_rad) * np.sin(az_rad)
    z = distance * np.sin(el_rad)
    return np.array([x, y, z])

def cart2sph(caert_cord):
    """Input:
    np.array([x, y, z])
    
    Output:
    np.array([distance, azimuth_deg, elevation_deg])"""
    x, y, z = caert_cord
    dist = np.sqrt(x**2 + y**2 + z**2) # distance from origin in meters   
    az_deg = np.degrees(np.arctan2(y, x)) % 360  # azimuth in degrees
    
    el_deg = np.degrees(np.arctan2(z, np.hypot(x, y))) # elevation in degrees, note: np.hypot(x, y)=sqrt(x^2 + y^2)
    return np.array([dist, az_deg, el_deg])



# Select source position based on coordinate type
if CARTISIAN:
    source_pos = source_pos_caert
    source_spherical_pos = cart2sph(source_pos)
else:
    source_spherical_pos = source_pos_sph
    source_pos = sph2cart(source_pos_sph)

print(f"Source position (x, y, z): {source_pos}")
print(f"Source sphetai position (distance, azimuth, elevation): {source_spherical_pos}")


# ---------------------------
# GCC-PHAT cross-correlation (full cc + lags) for a pair
# ---------------------------
def gcc_phat_cc(sig, refsig, fs, interp=16):
    """
    Returns:
      cc: PHAT-weighted cross-correlation array (interpolated, centered)
      lags: time lags [s] corresponding to cc indices
    """
    n = sig.shape[0] + refsig.shape[0]
    nfft = 1 << (n - 1).bit_length()  # next pow2

    SIG = rfft(sig, n=nfft)
    REFSIG = rfft(refsig, n=nfft)
    R = SIG * np.conj(REFSIG)
    R /= np.abs(R) + np.finfo(float).eps

    cc = irfft(R, n=interp * nfft)
    max_shift = int(interp * nfft / 2)
    # center it so zero lag is in the middle
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
    # time axis for lags
    lags = np.arange(-max_shift, max_shift + 1, dtype=float) / (interp * fs)
    return cc, lags

