#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from geometry_msgs.msg import Point, Vector3
from uav_interfaces.msg import DroneState, DOAData, PointPillarsData, ParticleFilterState

class UAVDynamicModel:
    """Clean dynamic model for UAV - only used for particle prediction"""
    def __init__(self):
        self.max_horizontal_speed = 22.0
        self.max_vertical_speed = 5.0
        self.velocity_decay = 0.98
        self.position_noise_std = 0.1
        self.angle_noise_std = 0.05  # rad

    def constrain_velocity(self, velocity):
        horizontal_speed = np.linalg.norm(velocity[:2])
        if horizontal_speed > self.max_horizontal_speed:
            scale = self.max_horizontal_speed / horizontal_speed
            velocity[0] *= scale
            velocity[1] *= scale
        
        if abs(velocity[2]) > self.max_vertical_speed:
            velocity[2] = np.sign(velocity[2]) * self.max_vertical_speed
        return velocity

class ParticleFilter:
    """Unified particle filter with full state (XYZ + 2 angles from origin)"""
    def __init__(self, system_position, num_particles=500):
        self.system_position = system_position
        self.num_particles = num_particles
        self.dynamic_model = UAVDynamicModel()
        
        # Particles: each has [x, y, z, azimuth, elevation] 
        self.particles = None  # shape: (num_particles, 5)
        self.velocities = None  # shape: (num_particles, 3) - only for XYZ
        self.weights = None
        
    def initialize_particles(self, initial_range=(10, 50)):
        """Initialize particles with random positions and angles"""
        # Position particles randomly in 3D space around system
        positions = np.random.uniform(-initial_range[1], initial_range[1], (self.num_particles, 3))
        positions[:, 2] = np.abs(positions[:, 2])  # Keep z positive
        
        # Initialize with random angles (azimuth, elevation)
        angles = np.random.uniform(-np.pi, np.pi, (self.num_particles, 2))
        angles[:, 1] = np.clip(angles[:, 1], -np.pi/2, np.pi/2)  # Elevation between -90° and 90°
        
        # Combine positions and angles
        self.particles = np.hstack([positions, angles])
        
        # Initialize velocities (only for XYZ)
        self.velocities = np.random.uniform(-1, 1, (self.num_particles, 3))
        for i in range(self.num_particles):
            self.velocities[i] = self.dynamic_model.constrain_velocity(self.velocities[i])
            
        self.weights = np.ones(self.num_particles) / self.num_particles
        
    def particle_to_cartesian(self, particle):
        """Convert particle state to Cartesian coordinates"""
        return particle[:3]
    
    def particle_to_angles(self, particle):
        """Get azimuth and elevation from particle"""
        return particle[3], particle[4]
    
    def cartesian_to_angles(self, position):
        """Convert Cartesian position to angles relative to system origin"""
        relative_pos = position - self.system_position
        distance = np.linalg.norm(relative_pos)
        
        if distance > 0:
            azimuth = np.arctan2(relative_pos[1], relative_pos[0])
            elevation = np.arcsin(relative_pos[2] / distance)
        else:
            azimuth = 0.0
            elevation = 0.0
            
        return azimuth, elevation
        
    def predict(self, dt=0.1):
        """Predict particle movement - update both position and angles"""
        if self.particles is None:
            return
            
        # Add process noise to velocities
        velocity_noise = np.random.normal(0, 0.05, (self.num_particles, 3))
        self.velocities += velocity_noise
        
        # Constrain velocities
        for i in range(self.num_particles):
            self.velocities[i] = self.dynamic_model.constrain_velocity(self.velocities[i])
        
        # Update positions (XYZ)
        self.particles[:, :3] += self.velocities * dt
        
        # Add position noise
        position_noise = np.random.normal(0, self.dynamic_model.position_noise_std, (self.num_particles, 3))
        self.particles[:, :3] += position_noise
        
        # Update angles based on new positions (angles are derived from position relative to system)
        for i in range(self.num_particles):
            azimuth, elevation = self.cartesian_to_angles(self.particles[i, :3])
            # Add small angle noise
            angle_noise = np.random.normal(0, self.dynamic_model.angle_noise_std, 2)
            self.particles[i, 3] = azimuth + angle_noise[0]
            self.particles[i, 4] = elevation + angle_noise[1]
            # Keep elevation in valid range
            self.particles[i, 4] = np.clip(self.particles[i, 4], -np.pi/2, np.pi/2)
            
    def update_with_doa(self, doa_data, doa_std_rad=0.1):
        """Update weights based on DOA measurement (azimuth, elevation)"""
        doa_azimuth = np.deg2rad(doa_data[0])
        doa_elevation = np.deg2rad(doa_data[1])
        
        for i in range(self.num_particles):
            particle_azimuth, particle_elevation = self.particle_to_angles(self.particles[i])
            
            # Calculate angular errors
            azimuth_error = self.angle_difference(particle_azimuth, doa_azimuth)
            elevation_error = self.angle_difference(particle_elevation, doa_elevation)
            
            # Combined angular likelihood
            azimuth_likelihood = np.exp(-0.5 * (azimuth_error / doa_std_rad) ** 2)
            elevation_likelihood = np.exp(-0.5 * (elevation_error / doa_std_rad) ** 2)
            
            angular_likelihood = azimuth_likelihood * elevation_likelihood
            
            # Update weight
            self.weights[i] *= angular_likelihood
        
        # Normalize weights
        if np.sum(self.weights) > 0:
            self.weights /= np.sum(self.weights)
        else:
            self.weights = np.ones(self.num_particles) / self.num_particles

    def update_with_pp(self, pp_position, position_std=2.0):
        """Update weights based on PointPillars position measurement"""
        for i in range(self.num_particles):
            particle_position = self.particle_to_cartesian(self.particles[i])
            
            # Calculate position error
            position_error = np.linalg.norm(particle_position - pp_position)
            position_likelihood = np.exp(-0.5 * (position_error / position_std) ** 2)
            
            # Update weight
            self.weights[i] *= position_likelihood
        
        # Normalize weights
        if np.sum(self.weights) > 0:
            self.weights /= np.sum(self.weights)
        else:
            self.weights = np.ones(self.num_particles) / self.num_particles

    def angle_difference(self, angle1, angle2):
        """Calculate smallest difference between two angles"""
        diff = angle1 - angle2
        return np.arctan2(np.sin(diff), np.cos(diff))

    def resample(self):
        """Systematic resampling"""
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0
        
        positions = (np.arange(self.num_particles) + np.random.random()) / self.num_particles
        indices = np.zeros(self.num_particles, dtype=int)
        
        i, j = 0, 0
        while i < self.num_particles:
            if positions[i] < cumulative_sum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1
                
        self.particles = self.particles[indices]
        self.velocities = self.velocities[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles
        
    def process_measurement(self, sensor_type, sensor_data, dt=0.1):
        """Main processing function for any sensor type"""
        if self.particles is None:
            self.initialize_particles()
        
        # 1. PREDICT: Move particles according to dynamics
        self.predict(dt)
        
        # 2. UPDATE: Update weights based on sensor type
        if sensor_type == 'doa':
            self.update_with_doa(sensor_data)
        elif sensor_type == 'pp':
            self.update_with_pp(sensor_data)
        
        # 3. RESAMPLE: Keep good particles, discard bad ones
        self.resample()
    
    def estimate_state(self):
        """Get position and velocity estimates from most likely particle"""
        if self.particles is not None and self.weights is not None:
            most_likely_idx = np.argmax(self.weights)
            position = self.particle_to_cartesian(self.particles[most_likely_idx])
            velocity = self.velocities[most_likely_idx]
            azimuth, elevation = self.particle_to_angles(self.particles[most_likely_idx])
            return position, velocity, azimuth, elevation
        return np.array([0, 0, 0]), np.array([0, 0, 0]), 0.0, 0.0

class ParticleFilterNode(Node):
    """Clean ROS node with unified particle filter"""
    def __init__(self):
        super().__init__('particle_filter_node')
        
        # System position (sensor location)
        self.system_position = np.array([0, 0, 0])
        
        # Initialize unified particle filter
        self.pf = ParticleFilter(self.system_position, num_particles=500)
        
        # Data storage for visualization
        self.true_positions = []
        self.estimated_positions = []
        self.doa_measurements = []
        self.pp_measurements = []
        
        # Latest sensor data
        self.latest_doa_data = None
        self.latest_pp_data = None
        self.latest_true_state = None
        
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
        
        self.true_state_sub = self.create_subscription(
            DroneState,
            '/drone/true_state',
            self.true_state_callback,
            10
        )
        
        # Publisher for filter results
        self.filter_state_pub = self.create_publisher(ParticleFilterState, '/filter/state', 10)
        
        # Visualization setup
        self.setup_plot()
        
        # Main processing timer
        self.timer = self.create_timer(0.1, self.process_measurements)
        
        self.get_logger().info("Unified particle filter node started")

    def setup_plot(self):
        """Initialize 3D plot for visualization"""
        plt.ion()
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim([-50, 50])
        self.ax.set_ylim([-50, 50])
        self.ax.set_zlim([0, 50])
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('Unified Particle Filter - Full State Estimation')

    def doa_callback(self, msg):
        """Store latest DOA data"""
        self.latest_doa_data = np.array([msg.azimuth, msg.elevation])
        self.get_logger().debug(f"Received DOA: az={msg.azimuth:.1f}°, el={msg.elevation:.1f}°")

    def pp_callback(self, msg):
        """Store latest PointPillars data"""
        self.latest_pp_data = np.array([msg.position.x, msg.position.y, msg.position.z])
        self.get_logger().debug(f"Received PP: ({self.latest_pp_data[0]:.1f}, {self.latest_pp_data[1]:.1f}, {self.latest_pp_data[2]:.1f})")

    def true_state_callback(self, msg):
        """Store latest true state for visualization only"""
        self.latest_true_state = {
            'position': np.array([msg.true_position.x, msg.true_position.y, msg.true_position.z])
        }
        self.true_positions.append(self.latest_true_state['position'].copy())

    def process_measurements(self):
        """Main processing - filter incoming sensor data"""
        has_new_data = False
        
        # Process DOA data
        if self.latest_doa_data is not None:
            self.pf.process_measurement('doa', self.latest_doa_data)
            
            # Convert DOA to 3D point for visualization (at fixed distance)
            azimuth_rad = np.deg2rad(self.latest_doa_data[0])
            elevation_rad = np.deg2rad(self.latest_doa_data[1])
            x = 30 * np.cos(elevation_rad) * np.cos(azimuth_rad)
            y = 30 * np.cos(elevation_rad) * np.sin(azimuth_rad)
            z = 30 * np.sin(elevation_rad)
            self.doa_measurements.append(np.array([x, y, z]))
            
            has_new_data = True
            self.latest_doa_data = None  # Clear after processing
            
        # Process PointPillars data
        if self.latest_pp_data is not None:
            self.pf.process_measurement('pp', self.latest_pp_data)
            self.pp_measurements.append(self.latest_pp_data.copy())
            has_new_data = True
            self.latest_pp_data = None  # Clear after processing
        
        if has_new_data:
            # Get estimate
            est_position, est_velocity, est_azimuth, est_elevation = self.pf.estimate_state()
            self.estimated_positions.append(est_position.copy())
            
            # Publish filter result (only position and velocity - no azimuth/elevation in message)
            self.publish_filter_state(est_position, est_velocity)
            
            # Update visualization
            self.update_plot()

    def publish_filter_state(self, position, velocity):
        """Publish filtered state - only position and velocity"""
        filter_msg = ParticleFilterState()
        filter_msg.header.stamp = self.get_clock().now().to_msg()
        filter_msg.header.frame_id = "world"
        
        # Position
        filter_msg.estimated_position.x = float(position[0])
        filter_msg.estimated_position.y = float(position[1])
        filter_msg.estimated_position.z = float(position[2])
        
        # Velocity
        filter_msg.estimated_velocity.x = float(velocity[0])
        filter_msg.estimated_velocity.y = float(velocity[1])
        filter_msg.estimated_velocity.z = float(velocity[2])
        
        self.filter_state_pub.publish(filter_msg)

    def update_plot(self):
        """Update visualization with current data"""
        self.ax.clear()
        
        self.ax.set_xlim([-50, 50])
        self.ax.set_ylim([-50, 50])
        self.ax.set_zlim([0, 50])
        self.ax.set_title('Unified Particle Filter - Full State Estimation')
        
        # Plot system origin
        self.ax.scatter(*self.system_position, c='black', s=100, marker='*', label='System Origin')
        
        # Plot true trajectory if available
        if len(self.true_positions) > 1:
            true_traj = np.array(self.true_positions)
            self.ax.plot(true_traj[:, 0], true_traj[:, 1], true_traj[:, 2], 
                        'g-', linewidth=2, label='True Trajectory')
        
        # Plot estimated trajectory
        if len(self.estimated_positions) > 1:
            est_traj = np.array(self.estimated_positions)
            self.ax.plot(est_traj[:, 0], est_traj[:, 1], est_traj[:, 2], 
                        'b-', linewidth=2, label='Estimated Trajectory')
        
        # Plot particles (XYZ positions only)
        if self.pf.particles is not None:
            positions = self.pf.particles[:, :3]
            self.ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                          c='red', alpha=0.3, s=10, label='Particles')
        
        # Plot current estimate
        if len(self.estimated_positions) > 0:
            current_est = self.estimated_positions[-1]
            self.ax.scatter(*current_est, c='blue', s=100, marker='s', 
                          label='Current Estimate')
        
        # Plot current true position if available
        if self.latest_true_state is not None:
            self.ax.scatter(*self.latest_true_state['position'], c='green', s=100, 
                          marker='o', label='True Position')
        
        self.ax.legend()
        plt.draw()
        plt.pause(0.01)

def main():
    rclpy.init()
    node = ParticleFilterNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        plt.close('all')

if __name__ == "__main__":
    main()