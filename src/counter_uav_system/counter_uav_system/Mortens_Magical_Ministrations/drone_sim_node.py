#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import Header
import time
from dataclasses import dataclass
from typing import Dict, List
import json

@dataclass
class DroneState:
    """Drone state using standard ROS2 messages"""
    header: Header
    true_position: Point
    true_velocity: Vector3
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for easy serialization"""
        return {
            'header': {
                'stamp': {
                    'sec': self.header.stamp.sec,
                    'nanosec': self.header.stamp.nanosec
                },
                'frame_id': self.header.frame_id
            },
            'true_position': {
                'x': self.true_position.x,
                'y': self.true_position.y,
                'z': self.true_position.z
            },
            'true_velocity': {
                'x': self.true_velocity.x,
                'y': self.true_velocity.y,
                'z': self.true_velocity.z
            }
        }
    
    @classmethod
    def create(cls, position: np.ndarray, velocity: np.ndarray, node: Node, frame_id: str = "world"):
        """Create a DroneState message from numpy arrays"""
        header = Header()
        header.stamp = node.get_clock().now().to_msg()
        header.frame_id = frame_id
        
        pos = Point()
        pos.x = float(position[0])
        pos.y = float(position[1])
        pos.z = float(position[2])
        
        vel = Vector3()
        vel.x = float(velocity[0])
        vel.y = float(velocity[1])
        vel.z = float(velocity[2])
        
        return cls(header=header, true_position=pos, true_velocity=vel)

class UAVDynamicModel:
    """Simple dynamic model for DJI Mavic-like UAV"""
    def __init__(self):
        # Parameters that can be easily modified
        self.max_speed = 500.0 
        self.max_acceleration = 100.0 
        self.max_vertical_speed = 200.0 
        self.max_vertical_acceleration = 50.0 
        self.velocity_decay = 1.0
        self.position_noise_std = 0.0
        self.random_accel_std = 0.5  # Added parameter
        
        # You can add more parameters here and adjust them directly
        self.enable_noise = True
        self.enable_constraints = True

    def set_parameters(self, **kwargs):
        """Dynamically update parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                print(f"Updated {key} to {value}")

    def constrain_velocity(self, velocity):
        if not self.enable_constraints:
            return velocity
            
        horizontal_speed = np.linalg.norm(velocity[:2])
        vertical_speed = abs(velocity[2])
        
        if horizontal_speed > self.max_speed:
            scale = self.max_speed / horizontal_speed
            velocity[0] *= scale
            velocity[1] *= scale
        
        if vertical_speed > self.max_vertical_speed:
            velocity[2] = np.sign(velocity[2]) * self.max_vertical_speed
            
        return velocity
    
    def constrain_acceleration(self, acceleration):
        if not self.enable_constraints:
            return acceleration
            
        horizontal_accel = np.linalg.norm(acceleration[:2])
        vertical_accel = abs(acceleration[2])
        
        if horizontal_accel > self.max_acceleration:
            scale = self.max_acceleration / horizontal_accel
            acceleration[0] *= scale
            acceleration[1] *= scale
        
        if vertical_accel > self.max_vertical_acceleration:
            acceleration[2] = np.sign(acceleration[2]) * self.max_vertical_acceleration
            
        return acceleration
    
    def update_state(self, position, velocity, dt=0.1):
        if self.enable_noise:
            random_accel = np.random.normal(0, self.random_accel_std, 3)
            random_accel = self.constrain_acceleration(random_accel)
            velocity = velocity * self.velocity_decay + random_accel * dt
        else:
            velocity = velocity * self.velocity_decay
            
        velocity = self.constrain_velocity(velocity)
        
        new_position = position + velocity * dt
        
        if self.enable_noise and self.position_noise_std > 0:
            position_noise = np.random.normal(0, self.position_noise_std, 3)
            new_position += position_noise

        return new_position, velocity

class DroneSimNode(Node):
    def __init__(self):
        super().__init__('drone_sim_node')
        
        # Parameters that can be easily modified
        self.declare_parameter('update_rate', 10.0) # Hz
        self.declare_parameter('despawn_distance', 0.5) # meters
        self.declare_parameter('respawn_delay', 10.0) # seconds
        self.declare_parameter('spawn_distance', 100.0) # meters
        self.declare_parameter('log_level', 1)  # 0=debug, 1=info, 2=warn
        
        # Get parameters
        self.update_rate = self.get_parameter('update_rate').value
        self.despawn_distance = self.get_parameter('despawn_distance').value
        self.respawn_delay = self.get_parameter('respawn_delay').value
        self.spawn_distance = self.get_parameter('spawn_distance').value
        self.log_level = self.get_parameter('log_level').value
        
        # Publisher - using a standard ROS2 publisher
        from std_msgs.msg import String  # We'll use String msg to send serialized data
        self.drone_state_pub = self.create_publisher(String, '/drone/true_state', 10)
        
        # Alternatively, publish individual topics for each component
        self.position_pub = self.create_publisher(Point, '/drone/position', 10)
        self.velocity_pub = self.create_publisher(Vector3, '/drone/velocity', 10)
        self.header_pub = self.create_publisher(Header, '/drone/header', 10)
        
        # Timer for simulation updates
        self.timer_period = 1.0 / self.update_rate # seconds
        self.timer = self.create_timer(self.timer_period, self.update_simulation) # Timer callback
        
        # Drone state variables
        self.dynamic_model = UAVDynamicModel()
        self.origin = np.array([0, 0, 0])
        
        # Initialize drone state
        self.true_position, self.true_velocity = self.spawn_drone()
        self.drone_active = True
        self.despawn_time = None
        
        self.get_logger().info(f"Drone simulation node started with update rate: {self.update_rate}Hz")

    def spawn_drone(self):
        """Spawn drone at random location from origin"""
        # Generate random direction vector that is not pointing too much downward
        direction = np.random.normal(0, 1, 3)
        while direction[2] < -0.1:  # Avoid too much downward direction
            direction = np.random.normal(0, 1, 3)
        direction = direction / np.linalg.norm(direction)
        
        # Position at spawn distance
        spawn_position = direction * self.spawn_distance
        
        # Velocity towards origin
        velocity_direction = -direction
        initial_speed = np.random.uniform(3, 8)
        spawn_velocity = velocity_direction * initial_speed
        
        if self.log_level <= 1:
            self.get_logger().info(f"Drone spawned at: {spawn_position}, velocity: {spawn_velocity}")
        
        return spawn_position, spawn_velocity

    def update_drone_dynamics(self, true_position, true_velocity, dt=0.1):
        """Update drone position and maintain direction towards origin"""
        true_position, true_velocity = self.dynamic_model.update_state(true_position, true_velocity, dt)
        
        # Maintain consistent direction toward origin
        direction_to_origin = self.origin - true_position
        distance = np.linalg.norm(direction_to_origin)
        if distance > 0.1:
            direction_to_origin = direction_to_origin / distance
            true_velocity = direction_to_origin * np.linalg.norm(true_velocity)
        
        return true_position, true_velocity

    def check_despawn_condition(self, position):
        """Check if drone has reached origin (within despawn distance)"""
        distance_to_origin = np.linalg.norm(position)
        return distance_to_origin <= self.despawn_distance

    def update_simulation(self):
        """Update drone state and handle spawn/despawn logic"""
        current_time = time.time()
        
        if self.drone_active:
            # Update drone dynamics
            self.true_position, self.true_velocity = self.update_drone_dynamics(
                self.true_position, self.true_velocity, self.timer_period
            )
            
            # Check if drone should despawn
            if self.check_despawn_condition(self.true_position):
                self.drone_active = False
                self.despawn_time = current_time
                self.get_logger().info(f"Drone despawned at origin. Respawning in {self.respawn_delay} seconds...")
                return
            
            # Publish active drone state
            self.publish_drone_state()
            
        else:
            # Drone is despawning, check if respawn time has elapsed
            if current_time - self.despawn_time >= self.respawn_delay:
                self.true_position, self.true_velocity = self.spawn_drone()
                self.drone_active = True
                self.despawn_time = None
                self.get_logger().info("Drone respawned at new location!")
                self.publish_drone_state()

    def publish_drone_state(self):
        """Publish current drone state using standard ROS2 messages"""
        # Method 1: Publish serialized data as String
        drone_state = DroneState.create(
            position=self.true_position,
            velocity=self.true_velocity,
            node=self
        )
        
        # Serialize to JSON string
        from std_msgs.msg import String
        state_msg = String()
        state_msg.data = json.dumps(drone_state.to_dict())
        self.drone_state_pub.publish(state_msg)
        
        # Method 2: Also publish individual components (optional)
        header_msg = Header()
        header_msg.stamp = self.get_clock().now().to_msg()
        header_msg.frame_id = "world"
        
        pos_msg = Point()
        pos_msg.x = float(self.true_position[0])
        pos_msg.y = float(self.true_position[1])
        pos_msg.z = float(self.true_position[2])
        
        vel_msg = Vector3()
        vel_msg.x = float(self.true_velocity[0])
        vel_msg.y = float(self.true_velocity[1])
        vel_msg.z = float(self.true_velocity[2])
        
        self.header_pub.publish(header_msg)
        self.position_pub.publish(pos_msg)
        self.velocity_pub.publish(vel_msg)

def main():
    rclpy.init()
    node = DroneSimNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()