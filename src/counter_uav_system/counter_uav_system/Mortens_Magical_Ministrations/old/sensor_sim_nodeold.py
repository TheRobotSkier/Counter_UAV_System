#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Header, String
import json
import time

class SensorSimNode(Node):
    def __init__(self):
        super().__init__('sensor_sim_node')
        
        # Declare parameters with realistic ranges
        self.declare_parameters(
            namespace='',
            parameters=[
                ('doa_noise_degrees', 5.0),
                ('pp_noise_meters', 1.0),
                ('system_x', 0.0),
                ('system_y', 0.0),
                ('system_z', 0.0),
                ('confidence', 0.95),
                ('doa_range_m', 90.0),      # DOA system range: 90 meters
                ('pp_range_m', 70.0),       # PointPillars range: 70 meters
                ('doa_min_elevation', -10.0),  # Minimum elevation angle (degrees)
                ('doa_max_elevation', 80.0)    # Maximum elevation angle
            ]
        )
        
        # Get parameters
        self.system_position = np.array([
            self.get_parameter('system_x').value,
            self.get_parameter('system_y').value,
            self.get_parameter('system_z').value
        ])
        self.doa_noise_level = self.get_parameter('doa_noise_degrees').value
        self.pp_noise_std = self.get_parameter('pp_noise_meters').value
        self.confidence = self.get_parameter('confidence').value
        
        # Sensor ranges
        self.doa_range = self.get_parameter('doa_range_m').value
        self.pp_range = self.get_parameter('pp_range_m').value
        
        # DOA field of view constraints
        self.doa_min_elevation = np.deg2rad(self.get_parameter('doa_min_elevation').value)
        self.doa_max_elevation = np.deg2rad(self.get_parameter('doa_max_elevation').value)
        
        # Subscriber to true drone state
        self.drone_state_sub = self.create_subscription(
            String,
            '/drone/true_state',
            self.drone_state_callback,
            10
        )
        
        # Publishers for sensor data
        self.doa_pub = self.create_publisher(String, '/sensors/doa', 10)
        self.pp_pub = self.create_publisher(String, '/sensors/point_pillars', 10)
        
        # Publisher for sensor status (optional, for debugging)
        self.status_pub = self.create_publisher(String, '/sensors/status', 10)
        
        self.get_logger().info(f"Sensor simulation started:")
        self.get_logger().info(f"  DOA Range: {self.doa_range}m, Noise: ±{self.doa_noise_level}°")
        self.get_logger().info(f"  PP Range: {self.pp_range}m, Noise: ±{self.pp_noise_std}m")

    def is_within_doa_range_and_fov(self, position):
        """Check if position is within DOA range and field of view"""
        vector = position - self.system_position
        distance = np.linalg.norm(vector)
        
        if distance > self.doa_range or distance == 0:
            return False, distance, None
        
        # Check elevation angle (field of view)
        direction = vector / distance
        elevation = np.arcsin(direction[2])
        
        if not (self.doa_min_elevation <= elevation <= self.doa_max_elevation):
            return False, distance, np.rad2deg(elevation)
        
        return True, distance, np.rad2deg(elevation)

    def is_within_pp_range(self, position):
        """Check if position is within PointPillars range"""
        distance = np.linalg.norm(position - self.system_position)
        return distance <= self.pp_range, distance

    def generate_doa_data(self, true_position):
        """Generate DOA data with noise, only if within range and FOV"""
        in_range, distance, elevation_deg = self.is_within_doa_range_and_fov(true_position)
        
        if not in_range:
            return None, f"DOA: Out of range ({distance:.1f}m)" if distance > self.doa_range else "DOA: Out of FOV"
        
        vector = true_position - self.system_position
        direction = vector / distance
        
        azimuth = np.arctan2(direction[1], direction[0])
        elevation = np.arcsin(direction[2])
        
        # Add realistic noise
        azimuth += np.random.normal(0, np.deg2rad(self.doa_noise_level))
        elevation += np.random.normal(0, np.deg2rad(self.doa_noise_level))
        
        # Convert to degrees
        azimuth_deg = np.rad2deg(azimuth)
        elevation_deg = np.rad2deg(elevation)
        
        return (azimuth_deg, elevation_deg), None

    def generate_pp_position(self, true_position):
        """Generate PointPillars data with noise, only if within range"""
        in_range, distance = self.is_within_pp_range(true_position)
        
        if not in_range:
            return None, f"PP: Out of range ({distance:.1f}m)"
        
        # Generate noisy position
        noisy_position = true_position + np.random.normal(0, self.pp_noise_std, 3)
        return noisy_position, None

    def drone_state_callback(self, msg):
        """Process drone state and generate sensor data only if within range"""
        try:
            # Parse JSON string
            data = json.loads(msg.data)
            
            # Extract true position
            true_position = np.array([
                data['true_position']['x'],
                data['true_position']['y'], 
                data['true_position']['z']
            ])
            
            # Create header
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = "sensor_frame"
            
            status_messages = []
            
            # Generate DOA data (only if within 90m range and FOV)
            doa_data, doa_error = self.generate_doa_data(true_position)
            
            if doa_data is not None:
                azimuth, elevation = doa_data
                
                # Publish DOA data
                doa_msg = String()
                doa_msg.data = json.dumps({
                    'header': {
                        'stamp': {
                            'sec': header.stamp.sec,
                            'nanosec': header.stamp.nanosec
                        },
                        'frame_id': header.frame_id
                    },
                    'azimuth': float(azimuth),
                    'elevation': float(elevation),
                    'in_range': True
                })
                self.doa_pub.publish(doa_msg)
                status_messages.append("DOA: ✓")
            else:
                status_messages.append(doa_error)
            
            # Generate PointPillars data (only if within 70m range)
            pp_position, pp_error = self.generate_pp_position(true_position)
            
            if pp_position is not None:
                # Publish PointPillars data
                pp_msg = String()
                pp_msg.data = json.dumps({
                    'header': {
                        'stamp': {
                            'sec': header.stamp.sec,
                            'nanosec': header.stamp.nanosec
                        },
                        'frame_id': header.frame_id
                    },
                    'position': {
                        'x': float(pp_position[0]),
                        'y': float(pp_position[1]),
                        'z': float(pp_position[2])
                    },
                    'confidence': self.confidence,
                    'in_range': True
                })
                self.pp_pub.publish(pp_msg)
                status_messages.append("PP: ✓")
            else:
                status_messages.append(pp_error)
            
            # Optional: Publish status for debugging
            if len(status_messages) > 0:
                status_msg = String()
                status_msg.data = json.dumps({
                    'timestamp': time.time(),
                    'true_position': {
                        'x': float(true_position[0]),
                        'y': float(true_position[1]),
                        'z': float(true_position[2])
                    },
                    'distance': float(np.linalg.norm(true_position - self.system_position)),
                    'status': status_messages
                })
                self.status_pub.publish(status_msg)
                
                # Log occasionally
                self.get_logger().debug(f"Sensor status: {', '.join(status_messages)}")
                
        except (json.JSONDecodeError, KeyError) as e:
            self.get_logger().error(f"Failed to process drone state: {e}")

def main():
    rclpy.init()
    node = SensorSimNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()