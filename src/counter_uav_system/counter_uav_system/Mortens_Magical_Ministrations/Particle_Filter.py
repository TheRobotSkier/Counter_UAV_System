#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Point, Vector3
from uav_interfaces.msg import DroneState, DOAData, PointPillarsData, ParticleFilterState

class UAVDynamicModel:
    """Dynamic model for UAV particle prediction"""
    def __init__(self):
        self.max_horizontal_speed = 22.0
        self.max_vertical_speed = 5.0
        self.velocity_decay = 0.98
        self.position_noise_std = 0.5
        self.velocity_noise_std = 0.2

    def constrain_velocity(self, velocities):
        """Constrain velocities to physically plausible limits - VECTORIZED"""
        horizontal_speeds = np.linalg.norm(velocities[:, :2], axis=1)
        
        # Handle horizontal speed constraints
        mask = horizontal_speeds > self.max_horizontal_speed
        if np.any(mask):
            scale_factors = np.ones(len(velocities))
            scale_factors[mask] = self.max_horizontal_speed / horizontal_speeds[mask]
            velocities[:, 0] *= scale_factors
            velocities[:, 1] *= scale_factors
        
        # Handle vertical speed constraints
        vertical_mask = np.abs(velocities[:, 2]) > self.max_vertical_speed
        if np.any(vertical_mask):
            velocities[vertical_mask, 2] = np.sign(velocities[vertical_mask, 2]) * self.max_vertical_speed
        
        return velocities

class ParticleFilter:
    """Optimized particle filter for 3D position tracking"""
    def __init__(self, system_position, num_particles=1000, logger=None):
        self.system_position = system_position
        self.num_particles = num_particles
        self.dynamic_model = UAVDynamicModel()
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
            
        velocity_noise = np.random.normal(0, self.dynamic_model.velocity_noise_std, 
                                        (self.num_particles, 3))
        self.velocities += velocity_noise
        
        self.velocities = self.dynamic_model.constrain_velocity(self.velocities)
        self.velocities *= self.dynamic_model.velocity_decay
        
        self.particles += self.velocities * dt
        
        position_noise = np.random.normal(0, self.dynamic_model.position_noise_std, 
                                        (self.num_particles, 3))
        self.particles += position_noise
        
        self.particles[:, 2] = np.maximum(self.particles[:, 2], 0.1)
            
    def update_with_doa(self, doa_data, doa_std_rad=0.1): #std is in radians aka 0.1 rad = 5.72957795 degrees
        """Optimized DOA weight update - VECTORIZED"""
        doa_azimuth = np.deg2rad(doa_data[0])
        doa_elevation = np.deg2rad(doa_data[1])
        
        pred_azimuth, pred_elevation = self.cartesian_to_angles(self.particles) # The predictions are than made into radians instead of degrees
        
        azimuth_error = self.angle_difference_vectorized(pred_azimuth, doa_azimuth)
        elevation_error = self.angle_difference_vectorized(pred_elevation, doa_elevation)
        #Here the uncertanty is applied and to both azimuth and elevation This is done by calculating the likelihood of each particle given the measurement and the standard deviation
        azimuth_likelihood = np.exp(-0.5 * (azimuth_error / doa_std_rad) ** 2) 
        elevation_likelihood = np.exp(-0.5 * (elevation_error / doa_std_rad) ** 2)
        
        angular_likelihood = azimuth_likelihood * elevation_likelihood
        
        self.weights *= angular_likelihood
        
        self.normalize_weights()
        self.check_resampling_need()

    def update_with_pp(self, pp_position, position_std=0.5):
        """Optimized PointPillars weight update - VECTORIZED"""
        position_errors = np.linalg.norm(self.particles - pp_position, axis=1)
        position_likelihood = np.exp(-0.5 * (position_errors / position_std) ** 2)
        
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
        effective_sample_size = 1.0 / np.sum(self.weights ** 2) # Calculate ESS
        self.needs_resampling = effective_sample_size < self.num_particles / 2 # Threshold

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
        
    def process_measurement(self, sensor_type=None, sensor_data=None, dt=0.2):
        """Optimized particle filter procedure"""
        if self.particles is None:
            self.initialize_particles()
        
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
    """Optimized ROS node with particle filter - NO VISUALIZATION"""
    def __init__(self):
        super().__init__('particle_filter_node')
        
        self.system_position = np.array([0, 0, 0])
        self.pf = ParticleFilter(self.system_position, num_particles=500, logger=self.get_logger())
        
        # Latest sensor data
        self.latest_doa_data = None
        self.latest_pp_data = None
        
        # Subscribers
        self.doa_sub = self.create_subscription(
            DOAData,
            '/sensors/doa',
            self.doa_callback,
            10
        )
        
        self.pp_sub = self.create_subscription(
            PointPillarsData,
            '/sensors/point_pillars',
            self.pp_callback,
            10
        )
        
        # Publishers
        self.filter_state_pub = self.create_publisher(ParticleFilterState, '/filter/state', 10)
        self.particles_pub = self.create_publisher(ParticleFilterState, '/filter/particles', 10)
        
        # Main processing timer - runs at full speed
        self.timer = self.create_timer(0.1, self.process_measurements)
        
        self.get_logger().info("High-speed particle filter node started (no visualization)")

    def doa_callback(self, msg):
        """Store latest DOA data"""
        self.latest_doa_data = np.array([msg.azimuth, msg.elevation])

    def pp_callback(self, msg):
        """Store latest PointPillars data"""
        if (np.isfinite(msg.position.x) and np.isfinite(msg.position.y) and np.isfinite(msg.position.z)):
            self.latest_pp_data = np.array([msg.position.x, msg.position.y, msg.position.z])
        else:
            self.get_logger().warning("Received invalid PointPillars data")

    def process_measurements(self):
        """Main processing - optimized without visualization overhead"""
        sensor_type = None
        sensor_data = None
        
        if self.latest_pp_data is not None:
            sensor_type = 'pp'
            sensor_data = self.latest_pp_data
            self.latest_pp_data = None
        elif self.latest_doa_data is not None:
            sensor_type = 'doa'
            sensor_data = self.latest_doa_data
            self.latest_doa_data = None
        
        try:
            self.pf.process_measurement(sensor_type, sensor_data, dt=0.1)
            
            est_position, est_velocity, est_azimuth, est_elevation = self.pf.estimate_state()
            
            self.publish_filter_state(est_position, est_velocity)
            self.publish_particles()
                
        except Exception as e:
            self.get_logger().error(f"Error in particle filter processing: {str(e)}")

    def publish_filter_state(self, position, velocity):
        """Publish filtered state"""
        filter_msg = ParticleFilterState()
        filter_msg.header.stamp = self.get_clock().now().to_msg()
        filter_msg.header.frame_id = "world"
        
        filter_msg.estimated_position.x = float(position[0])
        filter_msg.estimated_position.y = float(position[1])
        filter_msg.estimated_position.z = float(position[2])
        
        filter_msg.estimated_velocity.x = float(velocity[0])
        filter_msg.estimated_velocity.y = float(velocity[1])
        filter_msg.estimated_velocity.z = float(velocity[2])
        
        self.filter_state_pub.publish(filter_msg)

    def publish_particles(self):
        """Publish particles for visualization node"""
        particles = self.pf.get_particles()
        if particles is not None:
            # Sample particles for publishing (to avoid overloading the network)
            sample_indices = np.random.choice(len(particles), min(100, len(particles)), replace=False)
            sampled_particles = particles[sample_indices]
            
            # Create a custom message or reuse existing one
            # For now, we'll publish the first particle position as a placeholder
            # In practice, you might want to create a custom message type for particles
            particle_msg = ParticleFilterState()
            particle_msg.header.stamp = self.get_clock().now().to_msg()
            particle_msg.header.frame_id = "world"
            
            # Store particle count in a field (temporary solution)
            particle_msg.estimated_position.x = float(len(sampled_particles))
            
            self.particles_pub.publish(particle_msg)

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
