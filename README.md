# Counter_UAV_System
A multi-sensor, close-range UAV interception platform integrating acoustic detection, LiDAR-based localization, predictive tracking, and automated targeting.

---

## 🛡️ Overview

**Counter_UAV_System** is a modular ROS2-based system designed to detect, track, and intercept small UAVs at close range.  
The platform integrates:

- **Acoustic direction-of-arrival (DOA)** using a tetrahedral 4-microphone array  
- **LiDAR-based 3D localization and classification** (PointPillars, future integration)  
- **Predictive motion modeling** for estimating UAV trajectory  
- **Pneumatic gun targeting & firing** when a UAV enters the interception window  

This repository contains the **full acoustic subsystem**, **Arduino firmware for environmental sensing**, and tooling for **real-time**, **logging**, and **offline DOA processing**.

---

## 🚀 Key Features

### 🎧 Acoustic DOA Subsystem
- Real-time DOA estimation (SRP-PHAT / GCC-PHAT)
- JACK low-latency multichannel audio pipeline
- ROS2 topic publisher `/acoustic_doa`
- Logging mode (CSV + raw WAV recording)
- Offline playback node to recompute DOA on recorded data
- DOA log viewer tool for quick visualization

### 🌫️ LiDAR Localization (future)
- 3D conical LiDAR
- UAV classification using PointPillars

### 🎯 Targeting System (future)
- Motion prediction using particle filters
- Pneumatic gun control logic

### 🔌 Speed-of-Sound Calibration Firmware
- Arduino Uno + DHT22
- ROS2 service `/get_speed_of_sound`
- Provides temperature/humidity for accurate DOA geometry

---

## 📦 Repository Structure
Counter_UAV_System/
│
├── src/
│ ├── cuav_acoustic/ # DOA node, logging node, viewer, offline node
│ ├── cuav_serial/ # Speed-of-sound ROS2 service node
│ ├── cuav_interfaces/ # Custom ROS2 msgs + srvs
│ └── cuav_bringup/ # Launch files
│
├── firmware/
│ └── cuav_arduino_sensors/ # Arduino PlatformIO project for DHT22
│
├── scripts/ # JACK start/stop + system launch helpers
│
├── cuav_system_logs/ # DOA logs + WAV recordings (ignored by Git)
│
└── README.md

---
## 🔧 System Configuration (`config.py`)

The file:
´´´
src/cuav_acoustic/cuav_acoustic/config.py
´´´
controls all key acoustic processing parameters.

Below are the **most important settings**:

### 🎤 **Microphone Geometry**
#### `D_M`
D_M = 0.213 # (example) distance between microphones in meters
This is **the physical spacing between microphones** in your tetrahedral array.

👉 **If your array spacing is wrong, DOA will be wrong**, because TDOA → angle conversion depends directly on this value.

Measure from **mic center to mic center** using calipers.  
This is the **single most important parameter** to set correctly!


### 🎧 **Audio + Processing Settings**

| Parameter | Meaning |
|----------|---------|
| `NUM_CHANNELS` | Number of microphones (4 for tetrahedral) |
| `FRAME_DUR_SEC` | Frame size used for FFT (e.g., 20 ms) |
| `OVERLAP_50` | Whether to use 50% frame overlap |
| `MIC_POS` | 3D coordinates of the microphones in meters |

### 🌡️ Speed of Sound Settings
The DOA node can get speed-of-sound two ways:

1. Fixed value

2. From ROS2 service `/get_speed_of_sound`

Relevant config parameters:
| Parameter                | Description                             |
| ------------------------ | --------------------------------------- |
| `DEFAULT_SPEED_OF_SOUND` | Used when mode is `"fixed"`             |
| `USE_SERVICE`            | If True → call Arduino calibration node |

### 📊 DOA Estimation Settings
| Parameter             | Meaning                                                  |
| --------------------- | -------------------------------------------------------- |
| `SRP_GRID_RES_DEG`    | Search grid resolution (smaller = more accurate, slower) |
| `GCC_PHAT_ENABLED`    | Toggle GCC-PHAT calculation                              |
| `DETECTION_THRESHOLD` | Min signal level to classify UAV as "detected"           |


---

# ⚙️ Installation

## 1. Install ROS2 Humble (Ubuntu 22.04)
Follow official instructions:  
https://docs.ros.org/en/humble/Installation.html

---

## 2. Install system dependencies

### JACK Audio (real-time)

```bash
sudo apt update
sudo apt install jackd2 qjackctl pulseaudio-module-jack

```
### Python JACK client:
```
pip3 install jack-client
```

### Allow real-time audio:
```
sudo usermod -a -G audio $USER
```
Log out and back in.

---

## 3. Install required Python packages
```
pip3 install numpy soundfile matplotlib pyserial setuptools
```
---

## 4. Install PlatformIO (for Arduino firmware)
install PlatformIO VS Code extension (recommended)
or
```
pip3 install platformio
```
---


# 🛠️ Building the ROS2 Workspace
```
cd Counter_UAV_System
colcon build --symlink-install
source install/setup.bash
```
---

# ▶️ Running the Acoustic DOA System
## Option A — Use the startup scripts
Full real-time system:
```
./start_cuav_acoustic_system.sh
```

Logging mode (records .csv + .wav):
```
./start_cuav_acoustic_logging.sh
```

## Option B — Manual Startup
## 1. Start JACK
(Replace hw:USB with your audio device)
```
jackd -R -d alsa -d hw:USB -r 96000 -p 256 -n 3
```

## 2. Start speed-of-sound service
```
ros2 run cuav_serial speed_of_sound_server
```

## 3. Start DOA node
```
ros2 run cuav_acoustic doa_node
```

## 4. Echo the topic
```
ros2 topic echo /acoustic_doa
```

# 🧪 Offline DOA Processing Tools (!Not fully functional yet!)
## Recompute DOA from recorded WAV files:
```
ros2 run cuav_acoustic doa_offline_node --ros-args \
    -p session_dir:=/absolute/path/to/session \
    -p realtime:=false
```

## Plot DOA timeline from CSV log:
```
ros2 run cuav_acoustic doa_log_viewer /absolute/path/to/session
```

# 📡 Flashing the Arduino Firmware
Upload Arduino code via PlatformIO GUI,
or navigate to the firmware folder:
```
cd firmware/cuav_arduino_sensors
pio run --target upload
```

Start serial monitor via latformIO GUI, or:
```
pio device monitor
```

# 💡 Future Extensions
- Sensor fusion (LiDAR + acoustic)

- UAV classifier training

- Targeting + firing control node

- System dashboard / GUI


# You now have a full acoustic subsystem for UAV detection and logging.