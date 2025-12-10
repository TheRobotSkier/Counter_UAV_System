# config.py
"""
Configuration file for easy parameter changes.
All parameters can be modified here without touching the main code.
"""

# Drone Simulation Parameters
DRONE_PARAMS = {
    'update_rate': 10.0,  # Hz
    'despawn_distance': 0.5,  # meters
    'respawn_delay': 10.0,  # seconds
    'spawn_distance': 100.0,  # meters
    'log_level': 1,  # 0=debug, 1=info, 2=warn
}

# UAV Dynamic Model Parameters
UAV_PARAMS = {
    'max_speed': 500.0,
    'max_acceleration': 100.0,
    'max_vertical_speed': 200.0,
    'max_vertical_acceleration': 50.0,
    'velocity_decay': 1.0,
    'position_noise_std': 0.0,
    'random_accel_std': 0.5,
    'enable_noise': True,
    'enable_constraints': True,
}

# Sensor Simulation Parameters
SENSOR_PARAMS = {
    'doa_noise_level': 10.0,  # degrees
    'pp_noise_std': 1.0,  # meters
    'system_position': [0.0, 0.0, 0.0],  # x, y, z
    'log_level': 1,
}

# Launch Configuration
LAUNCH_PARAMS = {
    'start_drone_node': True,
    'start_sensor_node': True,
    'publish_rate': 10.0,  # Hz
    'simulation_timeout': 0,  # 0 = run forever
}