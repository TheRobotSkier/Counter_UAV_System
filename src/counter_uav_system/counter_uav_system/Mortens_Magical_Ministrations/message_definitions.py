# message_definitions.py
"""
Custom message definitions using standard ROS2 messages.
All message structures are defined here for easy modification.
"""

from dataclasses import dataclass
from typing import List
import numpy as np

# Standard ROS2 message imports for type hints
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import Header
import json

# ============== Parameter Classes ==============

@dataclass
class ParticleFilterParams:
    """Parameters for the particle filter"""
    num_particles: int = 1000
    doa_std_rad: float = 0.1  # 0.1 rad = ~5.73 degrees
    pp_std_m: float = 0.5     # meters
    resample_threshold: float = 0.5  # ratio of particles for resampling
    prediction_dt: float = 0.1  # seconds
    position_noise_std: float = 0.5
    velocity_noise_std: float = 0.2
    max_horizontal_speed: float = 22.0
    max_vertical_speed: float = 5.0
    velocity_decay: float = 0.98

@dataclass
class SensorNoiseParams:
    """Parameters for sensor noise simulation"""
    doa_noise_degrees: float = 5.0
    pp_noise_meters: float = 1.0
    system_position: List[float] = None
    
    def __post_init__(self):
        if self.system_position is None:
            self.system_position = [0.0, 0.0, 0.0]

@dataclass
class DroneSimParams:
    """Parameters for drone simulation"""
    update_rate: float = 10.0  # Hz
    despawn_distance: float = 0.5
    respawn_delay: float = 10.0
    spawn_distance: float = 100.0
    max_speed: float = 500.0
    max_acceleration: float = 100.0
    max_vertical_speed: float = 200.0
    max_vertical_acceleration: float = 50.0

# ============== Message Serialization Functions ==============

def serialize_drone_state(position: np.ndarray, velocity: np.ndarray, header: Header) -> str:
    """Serialize drone state to JSON string"""
    data = {
        'header': {
            'stamp': {
                'sec': header.stamp.sec,
                'nanosec': header.stamp.nanosec
            },
            'frame_id': header.frame_id
        },
        'true_position': {
            'x': float(position[0]),
            'y': float(position[1]),
            'z': float(position[2])
        },
        'true_velocity': {
            'x': float(velocity[0]),
            'y': float(velocity[1]),
            'z': float(velocity[2])
        }
    }
    return json.dumps(data)

def deserialize_drone_state(json_str: str) -> tuple:
    """Deserialize drone state from JSON string"""
    data = json.loads(json_str)
    position = np.array([
        data['true_position']['x'],
        data['true_position']['y'],
        data['true_position']['z']
    ])
    velocity = np.array([
        data['true_velocity']['x'],
        data['true_velocity']['y'],
        data['true_velocity']['z']
    ])
    return position, velocity

def serialize_doa_data(azimuth: float, elevation: float, header: Header) -> str:
    """Serialize DOA data to JSON string"""
    data = {
        'header': {
            'stamp': {
                'sec': header.stamp.sec,
                'nanosec': header.stamp.nanosec
            },
            'frame_id': header.frame_id
        },
        'azimuth': azimuth,
        'elevation': elevation
    }
    return json.dumps(data)

def serialize_pp_data(position: np.ndarray, confidence: float, header: Header) -> str:
    """Serialize PointPillars data to JSON string"""
    data = {
        'header': {
            'stamp': {
                'sec': header.stamp.sec,
                'nanosec': header.stamp.nanosec
            },
            'frame_id': header.frame_id
        },
        'position': {
            'x': float(position[0]),
            'y': float(position[1]),
            'z': float(position[2])
        },
        'confidence': confidence
    }
    return json.dumps(data)

def serialize_filter_state(position: np.ndarray, velocity: np.ndarray, header: Header) -> str:
    """Serialize particle filter state to JSON string"""
    data = {
        'header': {
            'stamp': {
                'sec': header.stamp.sec,
                'nanosec': header.stamp.nanosec
            },
            'frame_id': header.frame_id
        },
        'estimated_position': {
            'x': float(position[0]),
            'y': float(position[1]),
            'z': float(position[2])
        },
        'estimated_velocity': {
            'x': float(velocity[0]),
            'y': float(velocity[1]),
            'z': float(velocity[2])
        }
    }
    return json.dumps(data)

def serialize_particles(particles: np.ndarray, header: Header) -> str:
    """Serialize particles for visualization"""
    # Convert particles to list of lists for JSON serialization
    particles_list = particles.tolist() if particles is not None else []
    
    data = {
        'header': {
            'stamp': {
                'sec': header.stamp.sec,
                'nanosec': header.stamp.nanosec
            },
            'frame_id': header.frame_id
        },
        'particles': particles_list,
        'num_particles': len(particles_list)
    }
    return json.dumps(data)

def deserialize_particles(json_str: str) -> np.ndarray:
    """Deserialize particles from JSON string"""
    data = json.loads(json_str)
    return np.array(data['particles'])