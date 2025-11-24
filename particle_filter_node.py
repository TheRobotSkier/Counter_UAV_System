#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from geometry_msgs.msg import Point, Vector3
from uav_interfaces.msg import DroneState, DOAData, PointPillarsData, ParticleFilterState
from collections import deque

class UAVDynamicModel:
    """Realistic dynamic model for DJI Mavic-like UAV"""
    def __init__(self):
        # More realistic parameters for DJI Mavic
        self.max_horizontal_speed = 22.0  # m/s (~80 km/h)
        self.max_vertical_speed = 5.0     # m/s
        self.max_horizontal_acceleration = 8.0  # m/s²
        self.max_vertical_acceleration = 2.0    # m/s²
        self.velocity_decay = 0.98        # Small drag effect
        self.position_noise_std = 0.05    # Reduced position noise
        
        # Realistic control inputs and disturbances
        self.wind_std = 0.2               # Wind disturbance
        self.control_noise_std = 0.1      # Control input noise

    def constrain_velocity(self, velocity):
        horizontal_speed = np.linalg.norm(velocity[:2])
        vertical_speed = abs(velocity[2])
        
        # Constrain horizontal speed
        if horizontal_speed > self.max_horizontal_speed:
            scale = self.max_horizontal_speed / horizontal_speed
            velocity[0] *= scale
            velocity[1] *= scale
        
        # Constrain vertical speed
        if vertical_speed > self.max_vertical_speed:
            velocity[2] = np.sign(velocity[2]) * self.max_vertical_speed
            
        return velocity
    
    def constrain_acceleration(self, acceleration):
        horizontal_accel = np.linalg.norm(acceleration[:2])
        vertical_accel = abs(acceleration[2])
        
        # Constrain horizontal acceleration
        if horizontal_accel > self.max_horizontal_acceleration:
            scale = self.max_horizontal_acceleration / horizontal_accel
            acceleration[0] *= scale
            acceleration[1] *= scale
        
        # Constrain vertical acceleration
        if vertical_accel > self.max_vertical_acceleration:
            acceleration[2] = np.sign(acceleration[2]) * self.max_vertical_acceleration
            
        return acceleration
    
    def update_state(self, position, velocity, dt=0.1):
        # Realistic acceleration model with wind and control disturbances
        wind_disturbance = np.random.normal(0, self.wind_std, 3)
        control_noise = np.random.normal(0, self.control_noise_std, 3)
        
        # Combined random acceleration (wind + control noise)
        random_accel = wind_disturbance + control_noise
        random_accel = self.constrain_acceleration(random_accel)
        
        # Update velocity with drag and acceleration
        velocity = velocity * self.velocity_decay + random_accel * dt
        velocity = self.constrain_velocity(velocity)
        
        # Update position
        new_position = position + velocity * dt
        position_noise = np.random.normal(0, self.position_noise_std, 3)
        new_position += position_noise

        return new_position, velocity

class SensorParticleFilter:
    """Base class for sensor-specific particle filters"""
    def __init__(self, system_position, num_particles=1000):
        self.system_position = system_position
        self.num_particles = num_particles
        self.dynamic_model = UAVDynamicModel()
        self.particles = None
        self.velocities = None
        self.weights = None
        
    def initialize_particles(self, initial_positions, initial_velocities):
        """Initialize particles with given positions and velocities"""
        self.particles = np.array(initial_positions)
        self.velocities = np.array(initial_velocities)
        self.weights = np.ones(self.num_particles) / self.num_particles
        
    def predict(self, dt=0.1):
        """Predict particle movement with realistic dynamics"""
        for i in range(self.num_particles):
            # Add realistic process noise
            velocity_noise = np.random.normal(0, 0.05, 3)  # Reduced noise
            self.velocities[i] += velocity_noise
            
            # Constrain velocities to realistic limits
            self.velocities[i] = self.dynamic_model.constrain_velocity(self.velocities[i])
            
            # Update position
            self.particles[i] += self.velocities[i] * dt
            
            # Add small position noise
            position_noise = np.random.normal(0, 0.02, 3)  # Reduced noise
            self.particles[i] += position_noise
            
    def resample(self, min_weight_threshold=0.001):
        """Resample particles using systematic resampling with weight threshold"""
        indices = self._systematic_resample()
        self.particles = self.particles[indices]
        self.velocities = self.velocities[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def _systematic_resample(self):
        """Systematic resampling for more balanced particle distribution"""
        positions = (np.arange(self.num_particles) + np.random.random()) / self.num_particles
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0  # Ensure sum is exactly 1
        
        indices = np.zeros(self.num_particles, dtype=int)
        i, j = 0, 0
        while i < self.num_particles:
            if positions[i] < cumulative_sum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1
        return indices
        
    def estimate_position(self):
        """Estimate position from most likely particle"""
        if self.particles is not None and self.weights is not None:
            # Find the index of the particle with highest weight
            most_likely_idx = np.argmax(self.weights)
            return self.particles[most_likely_idx]
        return np.array([0, 0, 0])
    
    def estimate_velocity(self):
        """Estimate velocity from most likely particle"""
        if self.velocities is not None and self.weights is not None:
            # Use velocity of the most likely particle
            most_likely_idx = np.argmax(self.weights)
            return self.velocities[most_likely_idx]
        return np.array([0, 0, 0])
    
    def process_measurement(self, sensor_data, dt=0.1):
        """Standard particle filter workflow: predict -> update weights -> resample"""
        if self.particles is None:
            self.initialize_from_sensor(sensor_data)
        else:
            # 1. PREDICT: Move particles according to dynamics
            self.predict(dt)
            
            # 2. UPDATE: Only update weights based on sensor data
            self.update_weights(sensor_data)
            
            # 3. RESAMPLE: Keep good particles, discard bad ones
            self.resample()

class DOAParticleFilter(SensorParticleFilter):
    """Particle filter for DOA sensor data"""
    def __init__(self, system_position, num_particles=1000):
        super().__init__(system_position, num_particles)
        self.angle_std_rad = np.deg2rad(5)  # 5 degree standard deviation
        
    def doa_to_world_coordinates(self, doa_data):
        """Convert DOA angles to a unit direction vector in world coordinates"""
        azimuth_rad = np.deg2rad(doa_data[0])
        elevation_rad = np.deg2rad(doa_data[1])
        
        x = np.cos(elevation_rad) * np.cos(azimuth_rad)
        y = np.cos(elevation_rad) * np.sin(azimuth_rad)
        z = np.sin(elevation_rad)
        
        direction_vector = np.array([x, y, z])
        norm = np.linalg.norm(direction_vector)
        if norm > 0:
            direction_vector /= norm
        
        return direction_vector

    def initialize_from_sensor(self, doa_data, max_distance=70):
        """Initialize particles based on DOA data"""
        direction_vector = self.doa_to_world_coordinates(doa_data)
        
        positions = []
        velocities = []
        
        for _ in range(self.num_particles):
            distance = np.random.uniform(5, max_distance)
            position = self.system_position + direction_vector * distance
            
            # Initialize with small random velocities
            velocity = np.random.uniform(-1, 1, 3)
            velocity = self.dynamic_model.constrain_velocity(velocity)
            
            positions.append(position)
            velocities.append(velocity)
        
        self.initialize_particles(positions, velocities)

    def update_weights(self, doa_data):
        """ONLY update weights based on DOA measurement - don't modify particle positions"""
        doa_vector = self.doa_to_world_coordinates(doa_data)
        
        for i in range(self.num_particles):
            particle_vector = self.particles[i] - self.system_position
            particle_distance = np.linalg.norm(particle_vector)
            
            if particle_distance > 0:
                particle_direction = particle_vector / particle_distance
                
                # Angular consistency
                dot_product = np.clip(np.dot(doa_vector, particle_direction), -1, 1)
                angle_error = np.arccos(dot_product)
                angle_likelihood = np.exp(-0.5 * (angle_error / self.angle_std_rad) ** 2)
                
                # ONLY update weight - no position modification!
                self.weights[i] *= angle_likelihood
        
        # Normalize weights
        if np.sum(self.weights) > 0:
            self.weights /= np.sum(self.weights)
        else:
            self.weights = np.ones(self.num_particles) / self.num_particles

class PPParticleFilter(SensorParticleFilter):
    """Particle filter for PointPillars sensor data"""
    def __init__(self, system_position, num_particles=1000):
        super().__init__(system_position, num_particles)
        self.position_std = 1.0  # Standard deviation for position measurements
        
    def initialize_from_sensor(self, pp_position):
        """Initialize particles around PP position"""
        positions = []
        velocities = []
        
        for _ in range(self.num_particles):
            # Spread particles around PP position with some noise
            position = pp_position + np.random.normal(0, 2.0, 3)
            
            # Initialize with small random velocities
            velocity = np.random.uniform(-1, 1, 3)
            velocity = self.dynamic_model.constrain_velocity(velocity)
            
            positions.append(position)
            velocities.append(velocity)
        
        self.initialize_particles(positions, velocities)

    def update_weights(self, pp_position):
        """ONLY update weights based on PP measurement - don't modify particle positions"""
        for i in range(self.num_particles):
            # Calculate position likelihood
            position_error = np.linalg.norm(self.particles[i] - pp_position)
            position_likelihood = np.exp(-0.5 * (position_error / self.position_std) ** 2)
            
            # ONLY update weight - no position modification!
            self.weights[i] *= position_likelihood
        
        # Normalize weights
        if np.sum(self.weights) > 0:
            self.weights /= np.sum(self.weights)
        else:
            self.weights = np.ones(self.num_particles) / self.num_particles

class MultiSensorParticleFilter:
    """Main particle filter that fuses multiple sensor sources"""
    def __init__(self, system_position=np.array([0, 0, 0]), num_particles_per_sensor=1000):
        self.system_position = system_position
        
        # Initialize sensor-specific filters
        self.doa_filter = DOAParticleFilter(system_position, num_particles_per_sensor)
        self.pp_filter = PPParticleFilter(system_position, num_particles_per_sensor)
        
        # Fused results
        self.fused_particles = None
        self.fused_weights = None
        self.fused_velocities = None
        
        # Sensor reliability (can be adaptive)
        self.doa_reliability = 0.7
        self.pp_reliability = 0.9
        
    def process_doa_measurement(self, doa_data, dt=0.1):
        """Process DOA measurement using standard particle filter workflow"""
        self.doa_filter.process_measurement(doa_data, dt)
            
    def process_pp_measurement(self, pp_data, dt=0.1):
        """Process PointPillars measurement using standard particle filter workflow"""
        self.pp_filter.process_measurement(pp_data, dt)
            
    def fuse_particles(self):
        """Fuse particles from all sensors"""
        if (self.doa_filter.particles is not None and 
            self.pp_filter.particles is not None):
            
            # Combine particles
            self.fused_particles = np.vstack([
                self.doa_filter.particles, 
                self.pp_filter.particles
            ])
            
            self.fused_velocities = np.vstack([
                self.doa_filter.velocities,
                self.pp_filter.velocities
            ])
            
            # Combine weights with reliability factors
            doa_weights = self.doa_filter.weights * self.doa_reliability
            pp_weights = self.pp_filter.weights * self.pp_reliability
            
            self.fused_weights = np.concatenate([doa_weights, pp_weights])
            
            # Normalize fused weights
            if np.sum(self.fused_weights) > 0:
                self.fused_weights /= np.sum(self.fused_weights)
            else:
                self.fused_weights = np.ones(len(self.fused_weights)) / len(self.fused_weights)
                
        elif self.doa_filter.particles is not None:
            # Only DOA available
            self.fused_particles = self.doa_filter.particles.copy()
            self.fused_velocities = self.doa_filter.velocities.copy()
            self.fused_weights = self.doa_filter.weights.copy()
            
        elif self.pp_filter.particles is not None:
            # Only PP available
            self.fused_particles = self.pp_filter.particles.copy()
            self.fused_velocities = self.pp_filter.velocities.copy()
            self.fused_weights = self.pp_filter.weights.copy()
            
    def estimate_position(self):
        """Estimate position from most likely fused particle"""
        if self.fused_particles is not None and self.fused_weights is not None:
            # Find the index of the particle with highest weight
            most_likely_idx = np.argmax(self.fused_weights)
            return self.fused_particles[most_likely_idx]
        return np.array([0, 0, 0])
    
    def estimate_velocity(self):
        """Estimate velocity from most likely fused particle"""
        if self.fused_velocities is not None and self.fused_weights is not None:
            # Use velocity of the most likely particle
            most_likely_idx = np.argmax(self.fused_weights)
            return self.fused_velocities[most_likely_idx]
        return np.array([0, 0, 0])

class ParticleFilterNode(Node):
    def __init__(self):
        super().__init__('particle_filter_node')
        
        # Initialize multi-sensor particle filter
        self.particle_filter = MultiSensorParticleFilter(num_particles_per_sensor=1000)
        
        # Data storage
        self.true_positions = deque(maxlen=100)
        self.estimated_positions = deque(maxlen=100)
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
        
        # Visualization
        self.setup_plot()
        
        # Timer for processing
        self.timer = self.create_timer(0.1, self.process_measurements)  # 10 Hz
        
        self.keep_running = True
        
        self.get_logger().info("Multi-sensor particle filter node started")

    def setup_plot(self):
        """Initialize the 3D plot"""
        plt.ion()  # Interactive mode
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim([-70, 70])
        self.ax.set_ylim([-70, 70])
        self.ax.set_zlim([0, 70])
        self.ax.set_xlabel('X axis (meters)')
        self.ax.set_ylabel('Y axis (meters)')
        self.ax.set_zlabel('Z axis (meters)')
        self.ax.set_title('Particle Filter: True Position + Sensor Data + Particles')
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def on_key_press(self, event):
        """Handle keyboard events"""
        if event.key in ['q', 'escape']:
            self.keep_running = False
            self.get_logger().info("Quitting...")

    def doa_callback(self, msg):
        """Store latest DOA data"""
        self.latest_doa_data = np.array([msg.azimuth, msg.elevation])

    def pp_callback(self, msg):
        """Store latest PointPillars data"""
        self.latest_pp_data = np.array([msg.position.x, msg.position.y, msg.position.z])

    def true_state_callback(self, msg):
        """Store latest true state for visualization"""
        self.latest_true_state = {
            'position': np.array([msg.true_position.x, msg.true_position.y, msg.true_position.z]),
            'velocity': np.array([msg.true_velocity.x, msg.true_velocity.y, msg.true_velocity.z])
        }
        self.true_positions.append(self.latest_true_state['position'].copy())

    def process_measurements(self):
        """Process incoming sensor data and update filter"""
        if not self.keep_running:
            self.destroy_node()
            return
            
        # Process DOA data if available
        if self.latest_doa_data is not None:
            self.particle_filter.process_doa_measurement(self.latest_doa_data)
            
        # Process PP data if available  
        if self.latest_pp_data is not None:
            self.particle_filter.process_pp_measurement(self.latest_pp_data)
            
        # Fuse particles from all sensors
        self.particle_filter.fuse_particles()
        
        # Get estimates
        estimated_position = self.particle_filter.estimate_position()
        estimated_velocity = self.particle_filter.estimate_velocity()
        
        self.estimated_positions.append(estimated_position.copy())
        
        # Publish filter state
        filter_msg = ParticleFilterState()
        filter_msg.header.stamp = self.get_clock().now().to_msg()
        filter_msg.header.frame_id = "world"
        
        filter_msg.estimated_position.x = float(estimated_position[0])
        filter_msg.estimated_position.y = float(estimated_position[1])
        filter_msg.estimated_position.z = float(estimated_position[2])
        
        filter_msg.estimated_velocity.x = float(estimated_velocity[0])
        filter_msg.estimated_velocity.y = float(estimated_velocity[1])
        filter_msg.estimated_velocity.z = float(estimated_velocity[2])
        
        self.filter_state_pub.publish(filter_msg)
        
        # Update visualization
        self.update_plot()

    def update_plot(self):
        """Update visualization with proper sensor data display"""
        self.ax.clear()
        
        self.ax.set_xlim([-70, 70])
        self.ax.set_ylim([-70, 70])
        self.ax.set_zlim([0, 70])
        self.ax.set_title('Particle Filter: True Position + Sensor Data + Particles')
        
        origin = self.particle_filter.system_position
        
        # 1. Plot TRUE position (green)
        if self.latest_true_state is not None:
            true_pos = self.latest_true_state['position']
            self.ax.scatter(*true_pos, color='green', s=150, marker='o', label='True Position')
        
        # 2. Plot RAW SENSOR DATA (not particles!)
        # DOA: Show as direction vector (blue)
        if self.latest_doa_data is not None:
            doa_vector = self.particle_filter.doa_filter.doa_to_world_coordinates(self.latest_doa_data)
            # Scale vector for visibility
            doa_end = origin + doa_vector * 50  # 50m long vector
            self.ax.quiver(*origin, *(doa_end-origin), color='blue', 
                          arrow_length_ratio=0.1, linewidth=2, label='DOA Direction')
        
        # PP: Show as single point (red)
        if self.latest_pp_data is not None:
            self.ax.scatter(*self.latest_pp_data, color='red', s=100, marker='X', label='PP Position')
        
        # 3. Plot PARTICLES (with color indicating weight)
        if (self.particle_filter.doa_filter.particles is not None and 
            self.particle_filter.doa_filter.weights is not None):
            
            particles = self.particle_filter.doa_filter.particles
            weights = self.particle_filter.doa_filter.weights
            
            # Color particles by weight (darker = higher weight)
            colors = plt.cm.Reds(weights / np.max(weights) if np.max(weights) > 0 else weights)
            self.ax.scatter(particles[:, 0], particles[:, 1], particles[:, 2], 
                          c=colors, s=10, alpha=0.6, label='DOA Particles')
        
        if (self.particle_filter.pp_filter.particles is not None and 
            self.particle_filter.pp_filter.weights is not None):
            
            particles = self.particle_filter.pp_filter.particles
            weights = self.particle_filter.pp_filter.weights
            
            # Color particles by weight
            colors = plt.cm.Greens(weights / np.max(weights) if np.max(weights) > 0 else weights)
            self.ax.scatter(particles[:, 0], particles[:, 1], particles[:, 2], 
                          c=colors, s=10, alpha=0.6, label='PP Particles')
        
        # 4. Plot ESTIMATED position (from fused particles)
        if len(self.estimated_positions) > 0:
            est_pos = self.estimated_positions[-1]
            self.ax.scatter(*est_pos, color='purple', s=150, marker='s', label='Estimated Position')
            
            # Plot estimated trajectory
            if len(self.estimated_positions) > 1:
                estimated_trajectory = np.array(self.estimated_positions)
                self.ax.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], 
                           estimated_trajectory[:, 2], 'purple', linewidth=2, 
                           label='Estimated Trajectory')
        
        # 5. Plot TRUE trajectory
        if len(self.true_positions) > 1:
            true_trajectory = np.array(self.true_positions)
            self.ax.plot(true_trajectory[:, 0], true_trajectory[:, 1], true_trajectory[:, 2],
                       'green', linewidth=2, label='True Trajectory')
        
        self.ax.legend()
        plt.draw()
        plt.pause(0.01)

def main():
    rclpy.init()
    node = ParticleFilterNode()
    
    try:
        while node.keep_running and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        plt.close('all')

if __name__ == "__main__":
    main()