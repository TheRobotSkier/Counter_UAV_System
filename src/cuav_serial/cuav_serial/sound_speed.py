"""
sound_speed.py

Utility functions for computing speed of sound in air using
Cramer (1993) formula, given temperature (°C) and relative humidity (%).
"""

import math

# -------------------------------------------------------------------
# Speed of sound formula constants (Cramer 1993)
# -------------------------------------------------------------------

CO2 = 0.0004228      # mole fraction of CO2 in air (422.8 ppm)
P = 101350.0         # mean atmospheric pressure in Pa


# Coefficients for Eq. 15
A = [
    331.5024,    # a0
    0.603055,    # a1
    -0.000528,   # a2
    51.471935,   # a3
    0.1495874,   # a4
    -0.000782,   # a5
    -1.82e-7,    # a6
    3.73e-8,     # a7
    -2.93e-10,   # a8
    -85.20931,   # a9
    -0.228525,   # a10
    5.91e-5,     # a11
    -2.835149,   # a12
    -2.15e-13,   # a13
    29.179762,   # a14
    0.000486     # a15
]

def compute_speed_of_sound(t_c: float, rh_percent: float) -> float:
    """
    Compute speed of sound in air [m/s] using Cramer (1993).

    Parameters
    ----------
    t_c : float
        Air temperature in degrees Celsius.
    rh_percent : float
        Relative humidity in percent (0–100).

    Returns
    -------
    c : float
        Speed of sound in m/s.
    """
    T_k = t_c + 273.15
    RH_frac = rh_percent / 100.0

    # Enhancement factor
    f = 1.00062 + 3.14e-8 * P + 5.6e-7 * t_c**2

    # Saturation vapor pressure (Pa)
    psv = math.exp(1.2811805e-5 * T_k * T_k -
                   1.9509874e-2 * T_k +
                   34.04926034 -
                   6.3536311e3 / T_k)

    # Mole fraction of water vapor
    xw = RH_frac * f * (psv / P)
    Xc = CO2

    # Eq. 15
    c = (A[0] + A[1]*t_c + A[2]*t_c**2 +
         (A[3] + A[4]*t_c + A[5]*t_c**2) * xw +
         (A[6] + A[7]*t_c + A[8]*t_c**2) * P +
         (A[9] + A[10]*t_c + A[11]*t_c**2) * Xc +
         A[12]*xw**2 + A[13]*P*P + A[14]*Xc*Xc +
         A[15]*xw * P * Xc)

    return c