import serial, math, time

# Speed of sound in air using Cramer (1993)
# Inputs: temperature (°C), relative humidity (%)
# Constants: CO2 = 422.8 ppm, mean pressure = 101349 Pa
CO2 = 0.0004228 # mole fraction of CO2 in air (422.8 ppm)
P = 101350.0 # mean atmospheric pressure in Pa

# Coefficients for Cramer's equation (1993) a_0 to a_15:
A = [331.5024, 0.603055, -0.000528, 51.471935, 0.1495874, -0.000782,
     -1.82e-7, 3.73e-8, -2.93e-10, -85.20931, -0.228525, 5.91e-5,
     -2.835149, -2.15e-13, 29.179762, 0.000486]

def speed_of_sound(t_c, RH_percent):
    T_k = t_c + 273.15 # Convert to Kelvin
    RH_frac = RH_percent/100.0 # Convert to fraction

    # Eq. A2: enhancement factor:
    f = 1.00062 + 3.14e-8*P + 5.6e-7*t_c*t_c

    # Eq. A3: saturation vapor pressure (Pa):
    psv = math.exp(1.2811805e-5*T_k*T_k - 1.9509874e-2*T_k + 34.04926034 - 6.3536311e3/T_k)

    # Eq. A1: mole fraction of water vapor:
    xw = RH_frac*f*(psv/P)

    # CO2 mole fraction:
    Xc = CO2

    # Eq. 15: Speed of sound (c) in m/s:
    c = (A[0] + A[1]*t_c + A[2]*t_c*t_c +
            (A[3] + A[4]*t_c + A[5]*t_c*t_c)*xw +
            (A[6] + A[7]*t_c + A[8]*t_c*t_c)*P +
            (A[9] + A[10]*t_c + A[11]*t_c*t_c)*Xc +
            A[12]*xw*xw + A[13]*P*P + A[14]*Xc*Xc +
            A[15]*xw*P*Xc)

    return c

# --- Setup serial connection to Arduino ---
ser = serial.Serial('COM3', 115200, timeout=2) # Update 'COM3' to your Arduino port
                                    # timeout=2 is important to prevent blocking reads

# --- Wait for Arduino to say "Ready" ---
print("Waiting for Arduino...")
while True:
    line = ser.readline().decode(errors='ignore').strip()
    if "Ready" in line:
        print("Arduino is ready.")
        break

# --- Main data loop ---
print("Reading sensor data...\n")
while True:
    line = ser.readline().decode().strip()
    try:
        t, h = map(float, line.split('|'))
        c = speed_of_sound(t, h)
        print(f"T={t:.2f} °C, RH={h:.2f} %, c={c:.4f} m/s")
    except:
        pass
