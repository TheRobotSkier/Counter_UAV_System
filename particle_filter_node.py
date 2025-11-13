#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from geometry_msgs.msg import Point, Vector3
from uav_interfaces.msg import DroneState, DOAData, PointPillarsData, ParticleFilterState
import threading
from collections import deque

class UAVDynamicModel:
    """Simple dynamic model for DJI Mavic-like UAV"""
    def __init__(self):
        self.max_speed = 500.0
        self.max_acceleration = 100.0
        self.max_vertical_speed = 200.0
        self.max_vertical_acceleration = 50.0
        self.velocity_decay = 1
        self.position_noise_std = 0.1

    def constrain_velocity(self, velocity):
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
        random_accel = np.random.normal(0, 0.5, 3)
        random_accel = self.constrain_acceleration(random_accel)
        
        velocity = velocity * self.velocity_decay + random_accel * dt
        velocity = self.constrain_velocity(velocity)
        
        new_position = position + velocity * dt
        position_noise = np.random.normal(0, self.position_noise_std, 3)
        new_position += position_noise

        return new_position, velocity

class UAVParticleFilter:
    def __init__(self, system_position=np.array([0, 0, 0]), 
                 system_orientation=np.array([0, 0, 0]),
                 num_particles=1000):
        self.system_position = system_position
        self.system_orientation = system_orientation
        self.num_particles = num_particles
        self.dynamic_model = UAVDynamicModel()
        self.particles = None
        self.velocities = None
        self.weights = None
        
    def initialize_particles(self, doa_data, max_distance=70):
        direction_vector = self.doa_to_world_coordinates(doa_data)
        
        positions = []
        velocities = []
        
        for _ in range(self.num_particles):
            distance = np.random.uniform(5, max_distance)
            position = self.system_position + direction_vector * distance
            
            velocity = np.random.uniform(-2, 2, 3)
            velocity = self.dynamic_model.constrain_velocity(velocity)
            
            positions.append(position)
            velocities.append(velocity)
        
        self.particles = np.array(positions)
        self.velocities = np.array(velocities)
        self.weights = np.ones(self.num_particles) / self.num_particles
    
    def doa_to_world_coordinates(self, doa_data):
        azimuth = np.deg2rad(doa_data[0])
        elevation = np.deg2rad(doa_data[1])
        
        x = np.cos(elevation) * np.cos(azimuth)
        y = np.cos(elevation) * np.sin(azimuth)
        z = np.sin(elevation)
        
        return np.array([x, y, z])
    
    def predict(self, dt=0.1):
        for i in range(self.num_particles):
            new_position, new_velocity = self.dynamic_model.update_state(
                self.particles[i], self.velocities[i], dt
            )
            self.particles[i] = new_position
            self.velocities[i] = new_velocity
    
    def update_with_pp(self, pp_position, correction_factor=0.1):
        for i in range(self.num_particles):
            correction_vector = pp_position - self.particles[i]
            self.particles[i] += correction_factor * correction_vector
    
    def update_doa(self, doa_data, angle_std_deg=5):
        doa_vector = self.doa_to_world_coordinates(doa_data)
        angle_std_rad = np.deg2rad(angle_std_deg)
        
        for i in range(self.num_particles):
            particle_vector = self.particles[i] - self.system_position
            particle_distance = np.linalg.norm(particle_vector)
            
            if particle_distance > 0:
                particle_direction = particle_vector / particle_distance
                dot_product = np.clip(np.dot(doa_vector, particle_direction), -1, 1)
                angle_error = np.arccos(dot_product)
                weight = np.exp(-0.5 * (angle_error / angle_std_rad) ** 2)
                self.weights[i] = weight
        
        if np.sum(self.weights) > 0:
            self.weights /= np.sum(self.weights)
        else:
            self.weights = np.ones(self.num_particles) / self.num_particles
    
    def resample(self):
        indices = np.random.choice(
            self.num_particles, 
            size=self.num_particles, 
            p=self.weights
        )
        
        self.particles = self.particles[indices]
        self.velocities = self.velocities[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles
    
    def estimate_position(self):
        return np.average(self.particles, weights=self.weights, axis=0)
    
    def estimate_velocity(self):
        return np.average(self.velocities, weights=self.weights, axis=0)
    
    def process_measurement_doa_only(self, doa_data, dt=0.1):
        if self.particles is None:
            self.initialize_particles(doa_data)
        else:
            self.predict(dt)
            self.update_doa(doa_data)
            self.resample()
        
        return self.estimate_position(), self.estimate_velocity()
    
    def process_measurement_with_pp(self, doa_data, pp_position, dt=0.1, pp_correction_factor=0.1):
        if self.particles is None:
            self.initialize_particles(doa_data)
        else:
            self.predict(dt)
            self.update_with_pp(pp_position, pp_correction_factor)
            self.update_doa(doa_data)
            self.resample()
        
        return self.estimate_position(), self.estimate_velocity()

class ParticleFilterNode(Node):
    def __init__(self):
        super().__init__('particle_filter_node')
        
        # Initialize particle filter
        self.particle_filter = UAVParticleFilter(num_particles=2000)
        
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
        
        # Mode control
        self.use_pp_data = False
        self.keep_running = True
        
        self.get_logger().info("Particle filter node started")

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
        self.ax.set_title('UAV Tracking with Dynamic Model Particle Filter')
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def on_key_press(self, event):
        """Handle keyboard events"""
        if event.key in ['q', 'escape']:
            self.keep_running = False
            self.get_logger().info("Quitting...")
        elif event.key in ['l']:
            self.use_pp_data = not self.use_pp_data
            mode = "PP+DOA" if self.use_pp_data else "DOA Only"
            self.get_logger().info(f"Switched to {mode} mode")

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

    def calculate_dynamic_timestep(self, estimated_position, base_dt=0.1, close_range_dt=1, threshold_distance=30):
        """Calculate dynamic timestep based on distance from sensor origin"""
        distance = np.linalg.norm(estimated_position - self.particle_filter.system_position)
        
        if distance <= threshold_distance:
            return close_range_dt
        else:
            return base_dt

    def process_measurements(self):
        """Process incoming sensor data and update filter"""
        if not self.keep_running:
            self.destroy_node()
            return
            
        if self.latest_doa_data is None:
            return  # Wait for DOA data
            
        # Calculate dynamic timestep
        current_estimate = (self.particle_filter.estimate_position() 
                          if self.particle_filter.particles is not None 
                          else np.array([0, 20, 50]))
        current_dt = self.calculate_dynamic_timestep(current_estimate)
        
        # Process measurement
        if self.use_pp_data and self.latest_pp_data is not None:
            estimated_position, estimated_velocity = self.particle_filter.process_measurement_with_pp(
                self.latest_doa_data, self.latest_pp_data, dt=current_dt
            )
        else:
            estimated_position, estimated_velocity = self.particle_filter.process_measurement_doa_only(
                self.latest_doa_data, dt=current_dt
            )
        
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
        """Update the visualization"""
        if self.latest_true_state is None:
            return
            
        self.ax.clear()
        
        self.ax.set_xlim([-70, 70])
        self.ax.set_ylim([-70, 70])
        self.ax.set_zlim([0, 70])
        
        mode_display = "PP+DOA" if self.use_pp_data else "DOA Only"
        self.ax.set_title(f'UAV Tracking - Mode: {mode_display}\n(Q: Quit, L: Toggle Mode)')
        
        # Plot particles
        if self.particle_filter.particles is not None:
            self.ax.scatter(self.particle_filter.particles[:, 0], 
                          self.particle_filter.particles[:, 1], 
                          self.particle_filter.particles[:, 2], 
                          color='r', s=1, alpha=0.3, label='Particles')
        
        # Plot trajectories
        if len(self.true_positions) > 1 and len(self.estimated_positions) > 1:
            true_trajectory = np.array(self.true_positions)
            estimated_trajectory = np.array(self.estimated_positions)
            
            self.ax.plot(true_trajectory[:, 0], true_trajectory[:, 1], true_trajectory[:, 2],
                       'g-', linewidth=2, label='True Trajectory')
            self.ax.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], estimated_trajectory[:, 2],
                       'b-', linewidth=2, label='Estimated Trajectory')
        
        # Plot current positions
        true_pos = self.latest_true_state['position']
        est_pos = self.estimated_positions[-1] if self.estimated_positions else true_pos
        
        self.ax.scatter(*true_pos, color='green', s=100, marker='o', label='True Position')
        self.ax.scatter(*est_pos, color='blue', s=100, marker='s', label='Estimated Position')
        
        # Plot velocity vectors
        true_vel = self.latest_true_state['velocity']
        est_vel = (self.particle_filter.estimate_velocity() 
                  if self.particle_filter.particles is not None 
                  else np.array([0, 0, 0]))
        
        self.ax.quiver(*true_pos, *true_vel, color='green', length=5, normalize=True, label='True Velocity')
        self.ax.quiver(*est_pos, *est_vel, color='blue', length=5, normalize=True, label='Estimated Velocity')
        
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