#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import json
from std_msgs.msg import Header, String
import time
from dataclasses import asdict

# Import our message definitions
from message_definitions import (
    ParticleFilterParams,
    serialize_filter_state,
    serialize_particles,
    deserialize_drone_state
)

class UAVDynamicModel:
    """Dynamic model for UAV particle prediction"""
    def __init__(self, params: ParticleFilterParams = None):
        self.params = params or ParticleFilterParams()
        
    def constrain_velocity(self, velocities):
        """Constrain velocities to physically plausible limits - VECTORIZED"""
        horizontal_speeds = np.linalg.norm(velocities[:, :2], axis=1)
        
        # Handle horizontal speed constraints
        mask = horizontal_speeds > self.params.max_horizontal_speed
        if np.any(mask):
            scale_factors = np.ones(len(velocities))
            scale_factors[mask] = self.params.max_horizontal_speed / horizontal_speeds[mask]
            velocities[:, 0] *= scale_factors
            velocities[:, 1] *= scale_factors
        
        # Handle vertical speed constraints
        vertical_mask = np.abs(velocities[:, 2]) > self.params.max_vertical_speed
        if np.any(vertical_mask):
            velocities[vertical_mask, 2] = np.sign(velocities[vertical_mask, 2]) * self.params.max_vertical_speed
        
        return velocities

class ParticleFilter:
    """Optimized particle filter for 3D position tracking"""
    def __init__(self, system_position, params: ParticleFilterParams = None, logger=None):
        self.system_position = system_position
        self.params = params or ParticleFilterParams()
        self.num_particles = self.params.num_particles
        self.dynamic_model = UAVDynamicModel(self.params)
        self.logger = logger
        
        # Particle state: position and velocity
        self.particles = None
        self.velocities = None
        self.weights = None
        
        # Track if we need resampling
        self.needs_resampling = False
        
    def initialize_particles(self, initial_range=(-70, 70)):
        """Initialize particles with random positions around system"""
        self.particles = np.random.uniform(-initial_range[1], initial_range[1], 
                                         (self.num_particles, 3))
        self.particles[:, 2] = np.abs(self.particles[:, 2])
        
        self.velocities = np.random.uniform(-1, 1, (self.num_particles, 3))
        self.velocities = self.dynamic_model.constrain_velocity(self.velocities)
            
        self.weights = np.ones(self.num_particles) / self.num_particles
        
    def cartesian_to_angles(self, positions):
        """Convert Cartesian positions to angles relative to system origin - VECTORIZED"""
        relative_pos = positions - self.system_position
        distances = np.linalg.norm(relative_pos, axis=1)
        
        valid_distances = distances > 0
        azimuth = np.zeros_like(distances)
        elevation = np.zeros_like(distances)
        
        azimuth[valid_distances] = np.arctan2(relative_pos[valid_distances, 1], 
                                            relative_pos[valid_distances, 0])
        elevation[valid_distances] = np.arcsin(relative_pos[valid_distances, 2] / distances[valid_distances])
        
        return azimuth, elevation
        
    def predict(self, dt=0.5):
        """Optimized particle movement prediction"""
        if self.particles is None:
            self.initialize_particles()
            
        velocity_noise = np.random.normal(0, self.params.velocity_noise_std, 
                                        (self.num_particles, 3))
        self.velocities += velocity_noise
        
        self.velocities = self.dynamic_model.constrain_velocity(self.velocities)
        self.velocities *= self.params.velocity_decay
        
        self.particles += self.velocities * dt
        
        position_noise = np.random.normal(0, self.params.position_noise_std, 
                                        (self.num_particles, 3))
        self.particles += position_noise
        
        self.particles[:, 2] = np.maximum(self.particles[:, 2], 0.1)
            
    def update_with_doa(self, doa_data):
        """Optimized DOA weight update - VECTORIZED"""
        doa_azimuth = np.deg2rad(doa_data[0])
        doa_elevation = np.deg2rad(doa_data[1])
        
        pred_azimuth, pred_elevation = self.cartesian_to_angles(self.particles)
        
        azimuth_error = self.angle_difference_vectorized(pred_azimuth, doa_azimuth)
        elevation_error = self.angle_difference_vectorized(pred_elevation, doa_elevation)
        
        azimuth_likelihood = np.exp(-0.5 * (azimuth_error / self.params.doa_std_rad) ** 2)
        elevation_likelihood = np.exp(-0.5 * (elevation_error / self.params.doa_std_rad) ** 2)
        
        angular_likelihood = azimuth_likelihood * elevation_likelihood
        
        self.weights *= angular_likelihood
        
        self.normalize_weights()
        self.check_resampling_need()

    def update_with_pp(self, pp_position):
        """Optimized PointPillars weight update - VECTORIZED"""
        position_errors = np.linalg.norm(self.particles - pp_position, axis=1)
        position_likelihood = np.exp(-0.5 * (position_errors / self.params.pp_std_m) ** 2)
        
        self.weights *= position_likelihood
        
        self.normalize_weights()
        self.check_resampling_need()

    def normalize_weights(self):
        """Normalize weights with robustness check"""
        total_weight = np.sum(self.weights)
        if total_weight > 0:
            self.weights /= total_weight
        else:
            self.weights = np.ones(self.num_particles) / self.num_particles
            if self.logger:
                self.logger.warning("Particle weights degenerated - reset to uniform")

    def check_resampling_need(self):
        """Check if resampling is needed based on effective sample size"""
        effective_sample_size = 1.0 / np.sum(self.weights ** 2)
        threshold = self.num_particles * self.params.resample_threshold
        self.needs_resampling = effective_sample_size < threshold

    def angle_difference_vectorized(self, angles1, angles2):
        """Vectorized angle difference calculation"""
        diff = angles1 - angles2
        return np.arctan2(np.sin(diff), np.cos(diff))

    def resample(self):
        """Resampling"""
        if not self.needs_resampling:
            return
            
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0
        
        positions = (np.arange(self.num_particles) + np.random.random()) / self.num_particles
        indices = np.searchsorted(cumulative_sum, positions)
        
        self.particles = self.particles[indices]
        self.velocities = self.velocities[indices]
        
        resample_noise = np.random.normal(0, 0.1, self.particles.shape)
        self.particles += resample_noise
        
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.needs_resampling = False
        
    def process_measurement(self, sensor_type=None, sensor_data=None, dt=None):
        """Optimized particle filter procedure"""
        if self.particles is None:
            self.initialize_particles()
        
        dt = dt or self.params.prediction_dt
        self.predict(dt)
        
        if sensor_type == 'doa' and sensor_data is not None:
            self.update_with_doa(sensor_data)
        elif sensor_type == 'pp' and sensor_data is not None:
            self.update_with_pp(sensor_data)
        
        if self.needs_resampling:
            self.resample()
    
    def estimate_state(self):
        """Get position and velocity estimates using weighted average"""
        if self.particles is not None and self.weights is not None:
            position = np.average(self.particles, axis=0, weights=self.weights)
            velocity = np.average(self.velocities, axis=0, weights=self.weights)
            
            azimuth, elevation = self.cartesian_to_angles(position.reshape(1, -1))
            
            return position, velocity, azimuth[0], elevation[0]
        return np.array([0, 0, 0]), np.array([0, 0, 0]), 0.0, 0.0

    def get_particles(self):
        """Get current particles for visualization"""
        return self.particles.copy() if self.particles is not None else None

class ParticleFilterNode(Node):
    """Optimized ROS node with particle filter using standard messages"""
    def __init__(self):
        super().__init__('particle_filter_node')
        
        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('num_particles', 1000),
                ('doa_std_rad', 0.1),
                ('pp_std_m', 0.5),
                ('resample_threshold', 0.5),
                ('prediction_dt', 0.1),
                ('position_noise_std', 0.5),
                ('velocity_noise_std', 0.2),
                ('max_horizontal_speed', 22.0),
                ('max_vertical_speed', 5.0),
                ('velocity_decay', 0.98),
                ('system_x', 0.0),
                ('system_y', 0.0),
                ('system_z', 0.0)
            ]
        )
        
        # Get parameters
        pf_params = ParticleFilterParams(
            num_particles=self.get_parameter('num_particles').value,
            doa_std_rad=self.get_parameter('doa_std_rad').value,
            pp_std_m=self.get_parameter('pp_std_m').value,
            resample_threshold=self.get_parameter('resample_threshold').value,
            prediction_dt=self.get_parameter('prediction_dt').value,
            position_noise_std=self.get_parameter('position_noise_std').value,
            velocity_noise_std=self.get_parameter('velocity_noise_std').value,
            max_horizontal_speed=self.get_parameter('max_horizontal_speed').value,
            max_vertical_speed=self.get_parameter('max_vertical_speed').value,
            velocity_decay=self.get_parameter('velocity_decay').value
        )
        
        system_position = np.array([
            self.get_parameter('system_x').value,
            self.get_parameter('system_y').value,
            self.get_parameter('system_z').value
        ])
        
        self.pf = ParticleFilter(system_position, pf_params, self.get_logger())
        
        # Latest sensor data
        self.latest_doa_data = None
        self.latest_pp_data = None
        
        # Subscribers - using String messages for JSON serialization
        self.doa_sub = self.create_subscription(
            String,
            '/sensors/doa',
            self.doa_callback,
            10
        )
        
        self.pp_sub = self.create_subscription(
            String,
            '/sensors/point_pillars',
            self.pp_callback,
            10
        )
        
        # Publishers
        self.filter_state_pub = self.create_publisher(String, '/filter/state', 10)
        self.particles_pub = self.create_publisher(String, '/filter/particles', 10)
        
        # Main processing timer
        self.timer = self.create_timer(pf_params.prediction_dt, self.process_measurements)
        
        self.get_logger().info(f"Particle filter node started with {pf_params.num_particles} particles")

    def doa_callback(self, msg):
        """Process DOA data from JSON string"""
        try:
            data = json.loads(msg.data)
            if data.get('in_range', True):  # Only process if in range
                self.latest_doa_data = np.array([data['azimuth'], data['elevation']])
        except (json.JSONDecodeError, KeyError) as e:
            self.get_logger().warning(f"Failed to parse DOA data: {e}")

    def pp_callback(self, msg):
        """Process PointPillars data from JSON string"""
        try:
            data = json.loads(msg.data)
            if data.get('in_range', True):  # Only process if in range
                position = data['position']
                self.latest_pp_data = np.array([position['x'], position['y'], position['z']])
        except (json.JSONDecodeError, KeyError) as e:
            self.get_logger().warning(f"Failed to parse PP data: {e}")

    def process_measurements(self):
        """Main processing loop"""
        sensor_type = None
        sensor_data = None
        
        # Prioritize PP data if available
        if self.latest_pp_data is not None:
            sensor_type = 'pp'
            sensor_data = self.latest_pp_data
            self.latest_pp_data = None
        elif self.latest_doa_data is not None:
            sensor_type = 'doa'
            sensor_data = self.latest_doa_data
            self.latest_doa_data = None
        
        try:
            # Process measurement
            self.pf.process_measurement(sensor_type, sensor_data)
            
            # Get estimate
            est_position, est_velocity, est_azimuth, est_elevation = self.pf.estimate_state()
            
            # Create header
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = "world"
            
            # Publish filter state
            filter_state_msg = String()
            filter_state_msg.data = serialize_filter_state(est_position, est_velocity, header)
            self.filter_state_pub.publish(filter_state_msg)
            
            # Publish particles for visualization
            particles = self.pf.get_particles()
            if particles is not None:
                particles_msg = String()
                particles_msg.data = serialize_particles(particles, header)
                self.particles_pub.publish(particles_msg)
                
        except Exception as e:
            self.get_logger().error(f"Error in particle filter processing: {str(e)}")

def main():
    rclpy.init()
    node = ParticleFilterNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Particle filter node shutting down...")
    except Exception as e:
        node.get_logger().error(f"Unexpected error: {str(e)}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()