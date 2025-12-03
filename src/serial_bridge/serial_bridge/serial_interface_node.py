#!/usr/bin/env python3
"""
serial_interface_node.py

ROS2 Service:
    /get_speed_of_sound

Type:
    counter_uav_msgs/srv/SpeedOfSound

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
    - Computes speed of sound using Cramer (1993)
    - Returns result in the service response
"""

import rclpy
from rclpy.node import Node
from counter_uav_msgs.srv import SpeedOfSound

import serial
from serial_bridge.sound_speed import compute_speed_of_sound


class SpeedOfSoundServer(Node):
    """
    ROS2 node providing the /get_speed_of_sound service.

    Handles:
        - Serial communication with Arduino
        - Measurement trigger ("MEASURE")
        - Parsing temperature/humidity readings
        - Computing speed of sound (Cramer 1993)
    """

    def __init__(self) -> None:
        super().__init__('speed_of_sound_server')

        # Declare parameters for serial port and baud rate
        self.declare_parameter('port', '/dev/ttyACM0')
        self.declare_parameter('baud', 115200)

        port = self.get_parameter('port').value
        baud = self.get_parameter('baud').value

        # Open serial port to Arduino
        self.ser = serial.Serial(port, baud, timeout=1.0)
        self.get_logger().info("Waiting for Arduino...")

        # Wait for a "READY" line from Arduino before accepting requests
        while True:
            line = self.ser.readline().decode().strip()
            if line.upper() == "READY":
                self.get_logger().info("Arduino ready.")
                break

        # Create ROS2 service
        self.srv = self.create_service(
            SpeedOfSound,
            '/get_speed_of_sound',
            self.handle_request
        )
        self.get_logger().info("SpeedOfSound service /get_speed_of_sound is ready.")

    # -------------------------------------------------------------------
    # Service callback
    # -------------------------------------------------------------------
    def handle_request(self, request, response):
        """
        Handle a /get_speed_of_sound request.

        This function:
          - Sends "MEASURE" to Arduino
          - Collects 'th_<temp>_<humidity>' lines between START and DONE
          - Computes average T and RH
          - Computes speed of sound and fills response.speed_of_sound
        """

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
            self.get_logger().error("No valid measurements received from Arduino!")
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
    # Proper cleanup
    # -------------------------------------------------------------------
    def destroy_node(self):
        """Ensure the serial port is cleanly closed on shutdown."""
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
                self.get_logger().info("Serial port closed.")
        except Exception as e:
            self.get_logger().warn(f"Error closing serial port: {e}")

        # Call the parent class destroy_node()
        super().destroy_node()


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
def main(args=None) -> None:
    rclpy.init(args=args)
    node = SpeedOfSoundServer()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
