```markdown
# cuav_arduino_sensors
Temperature/Humidity measurement firmware used by Counter_UAV_System for speed-of-sound calibration.

This is a **PlatformIO Arduino project** that communicates with a ROS2 node via USB serial.  
It measures **temperature (°C)** and **humidity (%)** using a **DHT22** sensor and streams data for 5 seconds when requested.

---

## 🔧 Hardware Requirements

- Arduino Uno (or compatible board)
- DHT22 temperature/humidity sensor
- USB cable to PC

### Wiring Table

| DHT22 Pin | Arduino Pin |
|-----------|-------------|
| VCC       | 5V          |
| GND       | GND         |
| DATA      | D7          |

---

## 📡 Serial Protocol

The ROS2 service `/get_speed_of_sound` sends the command:
´´´
MEASURE
´´´

Arduino responds:
´´´
READY
START
th_<temperature_C>_<humidity_percent>
th_<temperature_C>_<humidity_percent>
...
DONE
´´´

Example:
´´´
th_21.55_47.80
th_21.60_47.50
th_21.58_47.62
´´´

# 📁 Project Structure
´´´
firmware/
│
├── README.md
├── Temperatur_and_humidity
│   ├── include
│   │   └── README
│   ├── platformio.ini        # Build configuration
│   ├── src
│   │   └── main.cpp          # Firmware
│   └── test
│       ├── old_main.cpp
│       └── README
└── test_tools
    └── split_wav.py
´´´

