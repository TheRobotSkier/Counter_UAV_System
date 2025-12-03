#!/usr/bin/env python3
"""
speed_of_sound_service.py

ROS2 Service:
    /acoustics/speed_of_sound
Type:
    acoustics_msgs/srv/SpeedOfSound

Behavior:
    - Sends "MEASURE" to Arduino
    - Arduino responds:
            READY
            START
            th_<temp>_<humidity>
            th_<temp>_<humidity>
            ...
            DONE
    - Collects readings
    - Computes average temp & humidity
    - Computes speed of sound
    - Returns result
"""

import rclpy
from rclpy.node import Node
from acoustics_msgs.srv import SpeedOfSound

import serial
import math
import time

# -------------------------------------------------------------------
# Speed of sound formula constants (Cramer 1993)
# -------------------------------------------------------------------
CO2 = 0.0004228
P = 101350.0

A = [331.5024, 0.603055, -0.000528, 51.471935, 0.1495874, -0.000782,
     -1.82e-7, 3.73e-8, -2.93e-10, -85.20931, -0.228525, 5.91e-5,
     -2.835149, -2.15e-13, 29.179762, 0.000486]

def compute_speed_of_sound(t_c, RH_percent):
    """Cramer (1993) formula."""
    T_k = t_c + 273.15
    RH_frac = RH_percent / 100.0

    f = 1.00062 + 3.14e-8 * P + 5.6e-7 * t_c**2
    psv = math.exp(1.2811805e-5 * T_k * T_k -
                   1.9509874e-2 * T_k +
                   34.04926034 -
                   6.3536311e3 / T_k)

    xw = RH_frac * f * (psv / P)
    Xc = CO2

    c = (A[0] + A[1]*t_c + A[2]*t_c**2 +
         (A[3] + A[4]*t_c + A[5]*t_c**2) * xw +
         (A[6] + A[7]*t_c + A[8]*t_c**2) * P +
         (A[9] + A[10]*t_c + A[11]*t_c**2) * Xc +
         A[12]*xw**2 + A[13]*P*P + A[14]*Xc*Xc +
         A[15]*xw * P * Xc)

    return c


# -------------------------------------------------------------------
# ROS2 Service Node
# -------------------------------------------------------------------
class SpeedOfSoundServer(Node):

    def __init__(self):
        super().__init__('speed_of_sound_server')

        # Open serial port
        self.ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
        self.get_logger().info("Waiting for Arduino...")

        # Wait for READY
        while True:
            line = self.ser.readline().decode().strip()
            if line.upper() == "READY":
                self.get_logger().info("Arduino ready.")
                break

        # Create service
        self.srv = self.create_service(
            SpeedOfSound,
            '/acoustics/speed_of_sound',
            self.handle_request
        )
        self.get_logger().info("SpeedOfSound service ready.")

    # -------------------------------------------------------------------
    # Service callback
    # -------------------------------------------------------------------
    def handle_request(self, request, response):

        self.get_logger().info("Service request received → MEASURE")

        # Tell Arduino to begin measurement
        self.ser.write(b"MEASURE\n")

        temps = []
        hums = []

        started = False

        # Read until DONE
        while True:
            line = self.ser.readline().decode().strip()

            if not line:
                continue

            # Look for START
            if line.upper() == "START":
                started = True
                continue

            # Look for DONE
            if line.upper() == "DONE":
                break

            # Parse measurement lines
            if started and line.startswith("th_"):
                try:
                    _, t_str, h_str = line.split("_")
                    t = float(t_str)
                    h = float(h_str)
                    temps.append(t)
                    hums.append(h)
                except Exception as e:
                    self.get_logger().warn(f"Bad line: {line}")

        # Compute averages
        if len(temps) == 0:
            self.get_logger().error("No valid measurements received!")
            response.speed_of_sound = float('nan')
            return response

        t_avg = sum(temps) / len(temps)
        h_avg = sum(hums) / len(hums)

        # Compute speed of sound
        c = compute_speed_of_sound(t_avg, h_avg)

        self.get_logger().info(
            f"T_avg={t_avg:.2f}°C, H_avg={h_avg:.2f}%, c={c:.3f} m/s"
        )

        response.speed_of_sound = float(c)
        return response


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = SpeedOfSoundServer()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
