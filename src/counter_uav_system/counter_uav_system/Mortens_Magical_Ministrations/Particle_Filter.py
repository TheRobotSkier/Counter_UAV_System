#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import json
from std_msgs.msg import String
import time
import math

class ParticleFilter(Node):
    """Improved particle filter for UAV tracking with Brownian motion"""
    
    def __init__(self):
        super().__init__('particle_filter')
        
        # Parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('num_particles', 2000),
                ('doa_std_deg', 5.0),        # Degrees (easier to understand)
                ('pp_std_m', 1.5),
                ('prediction_dt', 0.1),
                ('position_diffusion', 1.0),
                ('velocity_diffusion', 1.0),
                ('system_x', 0.0),
                ('system_y', 0.0),
                ('system_z', 0.0),
                ('min_height', -10.0),
                ('max_height', 200.0),
                ('init_radius', 100.0),
                ('resample_ratio', 0.5),
                ('max_speed', 25.0)          # m/s, realistic UAV speed
            ]
        )
        
        # Get parameters
        self.system_position = np.array([
            self.get_parameter('system_x').value,
            self.get_parameter('system_y').value,
            self.get_parameter('system_z').value
        ])
        
        self.num_particles = self.get_parameter('num_particles').value
        self.doa_std_deg = self.get_parameter('doa_std_deg').value
        self.doa_std_rad = np.deg2rad(self.doa_std_deg)
        self.pp_std = self.get_parameter('pp_std_m').value
        self.prediction_dt = self.get_parameter('prediction_dt').value
        self.position_diffusion = self.get_parameter('position_diffusion').value
        self.velocity_diffusion = self.get_parameter('velocity_diffusion').value
        self.min_height = self.get_parameter('min_height').value
        self.max_height = self.get_parameter('max_height').value
        self.init_radius = self.get_parameter('init_radius').value
        self.resample_ratio = self.get_parameter('resample_ratio').value
        self.max_speed = self.get_parameter('max_speed').value
        
        # State variables
        self.particles = None
        self.weights = None
        self.particle_velocities = None
        self.initialized = False
        
        # Measurement buffers
        self.latest_doa = None
        self.latest_doa_time = None
        self.latest_pp = None
        self.latest_pp_time = None
        
        # State history
        self.estimated_states = []
        self.max_history = 10
        
        # Subscribers
        self.create_subscription(String, '/sensors/doa', self.doa_callback, 10)
        self.create_subscription(String, '/sensors/point_pillars', self.pp_callback, 10)
        
        # Publishers
        self.state_pub = self.create_publisher(String, '/filter/state', 10)
        self.particles_pub = self.create_publisher(String, '/filter/particles', 10)
        self.debug_pub = self.create_publisher(String, '/filter/debug', 10)
        
        # Initialize
        self.initialize_particles()
        
        # Timer for prediction (runs at fixed rate)
        self.timer = self.create_timer(self.prediction_dt, self.update_filter)
        
        # Statistics
        self.update_count = 0
        
        self.get_logger().info(f"""
        Particle Filter Initialized:
          Particles: {self.num_particles}
          DOA STD: {self.doa_std_deg}° ({self.doa_std_rad:.3f} rad)
          PP STD: {self.pp_std}m
          Update rate: {1/self.prediction_dt}Hz
        """)

    def initialize_particles(self):
        """Initialize particles uniformly in a sphere around the sensor"""
        self.get_logger().info("Initializing particles...")
        
        # Method 1: Uniform in sphere (better for 3D)
        # Generate random points in unit sphere
        u = np.random.uniform(0, 1, self.num_particles)
        theta = np.random.uniform(0, 2*np.pi, self.num_particles)
        phi = np.arccos(2*np.random.uniform(0, 1, self.num_particles) - 1)
        
        # Convert to Cartesian with radius
        r = self.init_radius * (u ** (1/3))  # Cube root for uniform volume distribution
        self.particles = np.zeros((self.num_particles, 3))
        self.particles[:, 0] = r * np.sin(phi) * np.cos(theta)
        self.particles[:, 1] = r * np.sin(phi) * np.sin(theta)
        self.particles[:, 2] = r * np.cos(phi)
        
        # Shift to sensor position
        self.particles += self.system_position
        
        # Ensure minimum height
        self.particles[:, 2] = np.maximum(self.particles[:, 2], self.min_height)
        
        # Initialize velocities (small random values)
        self.particle_velocities = np.random.normal(0, 2.0, (self.num_particles, 3))
        
        # Initialize weights uniformly
        self.weights = np.ones(self.num_particles) / self.num_particles
        
        self.initialized = True
        self.get_logger().info(f"Initialized {self.num_particles} particles in {self.init_radius}m radius sphere")

    def predict(self):
        """Predict particle motion using Brownian motion model"""
        if not self.initialized:
            return
        
        dt = self.prediction_dt
        
        # 1. Update position based on velocity
        self.particles += self.particle_velocities * dt
        
        # 2. Add Brownian motion to position
        pos_noise_std = np.sqrt(2 * self.position_diffusion * dt)
        self.particles += np.random.normal(0, pos_noise_std, self.particles.shape)
        
        # 3. Add Brownian motion to velocity
        vel_noise_std = np.sqrt(2 * self.velocity_diffusion * dt)
        self.particle_velocities += np.random.normal(0, vel_noise_std, self.particle_velocities.shape)
        
        # 4. Apply constraints
        # Keep above minimum height
        self.particles[:, 2] = np.maximum(self.particles[:, 2], self.min_height)
        
        # Limit maximum height
        self.particles[:, 2] = np.minimum(self.particles[:, 2], self.max_height)
        
        # Limit velocity magnitude
        speeds = np.linalg.norm(self.particle_velocities, axis=1)
        too_fast = speeds > self.max_speed
        if np.any(too_fast):
            # Normalize velocities that are too fast
            for i in np.where(too_fast)[0]:
                self.particle_velocities[i] = self.particle_velocities[i] * self.max_speed / speeds[i]

    def update_with_doa(self, azimuth_deg, elevation_deg):
        """Update weights based on DOA (Direction of Arrival) measurement"""
        azimuth = np.deg2rad(azimuth_deg)
        elevation = np.deg2rad(elevation_deg)
        
        # Calculate vector from sensor to each particle
        vectors = self.particles - self.system_position
        distances = np.linalg.norm(vectors, axis=1)
        
        # Avoid division by zero for particles at sensor position
        valid = distances > 0.1
        pred_azimuth = np.zeros(self.num_particles)
        pred_elevation = np.zeros(self.num_particles)
        
        if np.any(valid):
            # Normalize vectors
            vectors_norm = vectors[valid] / distances[valid].reshape(-1, 1)
            
            # Calculate predicted angles
            # Azimuth: arctan2(y, x) in sensor frame
            pred_azimuth[valid] = np.arctan2(vectors_norm[:, 1], vectors_norm[:, 0])
            
            # Elevation: arcsin(z) in sensor frame
            pred_elevation[valid] = np.arcsin(vectors_norm[:, 2])
        
        # Calculate angular differences (handle circular wrapping)
        az_diff = self.angle_difference(pred_azimuth, azimuth)
        el_diff = self.angle_difference(pred_elevation, elevation)
        
        # Calculate Gaussian likelihoods
        az_likelihood = np.exp(-0.5 * (az_diff / self.doa_std_rad) ** 2)
        el_likelihood = np.exp(-0.5 * (el_diff / self.doa_std_rad) ** 2)
        
        # Combined likelihood (assuming independence)
        likelihood = az_likelihood * el_likelihood
        
        # Update weights (multiplicative)
        self.weights *= likelihood
        
        # Normalize weights
        self.normalize_weights()
        
        # Debug output occasionally
        if self.update_count % 50 == 0:
            ess = self.effective_sample_size()
            self.get_logger().debug(f"DOA Update - ESS: {ess:.1f}/{self.num_particles}")

    def angle_difference(self, angle1, angle2):
        """Calculate smallest difference between two angles (in radians)"""
        diff = angle1 - angle2
        # Wrap to [-pi, pi]
        return np.arctan2(np.sin(diff), np.cos(diff))

    def update_with_pp(self, pp_position):
        """Update weights based on PointPillars measurement"""
        # Calculate Euclidean distances from particles to measurement
        distances = np.linalg.norm(self.particles - pp_position, axis=1)
        
        # Gaussian likelihood based on distance
        likelihood = np.exp(-0.5 * (distances / self.pp_std) ** 2)
        
        # Update weights
        self.weights *= likelihood
        
        # Normalize weights
        self.normalize_weights()
        
        # Debug output occasionally
        if self.update_count % 50 == 0:
            ess = self.effective_sample_size()
            self.get_logger().debug(f"PP Update - ESS: {ess:.1f}/{self.num_particles}")

    def normalize_weights(self):
        """Normalize particle weights with numerical stability"""
        total_weight = np.sum(self.weights)
        
        if total_weight < 1e-10:
            # Weights are too small - reset to uniform
            self.weights = np.ones(self.num_particles) / self.num_particles
            self.get_logger().warn("Weights underflow - resetting to uniform")
        else:
            self.weights /= total_weight

    def effective_sample_size(self):
        """Calculate effective sample size (ESS)"""
        return 1.0 / np.sum(self.weights ** 2)

    def resample(self):
        """Perform systematic resampling"""
        # Calculate cumulative sum of weights
        cumulative_weights = np.cumsum(self.weights)
        
        # Ensure last weight is exactly 1.0
        cumulative_weights[-1] = 1.0
        
        # Generate sorted random numbers
        step = 1.0 / self.num_particles
        start = np.random.uniform(0, step)
        positions = start + np.arange(self.num_particles) * step
        
        # Find indices for resampling
        indices = np.zeros(self.num_particles, dtype=int)
        j = 0
        for i in range(self.num_particles):
            while positions[i] > cumulative_weights[j]:
                j += 1
            indices[i] = j
        
        # Resample particles and velocities
        self.particles = self.particles[indices]
        self.particle_velocities = self.particle_velocities[indices]
        
        # Reset weights to uniform
        self.weights = np.ones(self.num_particles) / self.num_particles
        
        # Add small jitter to prevent particle deprivation
        jitter_pos = np.random.normal(0, 0.1, self.particles.shape)
        jitter_vel = np.random.normal(0, 0.05, self.particle_velocities.shape)
        
        self.particles += jitter_pos
        self.particle_velocities += jitter_vel
        
        self.get_logger().debug("Resampling performed")

    def doa_callback(self, msg):
        """Handle incoming DOA measurements"""
        try:
            data = json.loads(msg.data)
            if 'azimuth' in data and 'elevation' in data:
                self.latest_doa = (float(data['azimuth']), float(data['elevation']))
                self.latest_doa_time = time.time()
                # self.get_logger().debug(f"DOA: {self.latest_doa}")
        except (json.JSONDecodeError, KeyError) as e:
            self.get_logger().warn(f"Failed to parse DOA: {e}")

    def pp_callback(self, msg):
        """Handle incoming PointPillars measurements"""
        try:
            data = json.loads(msg.data)
            if 'position' in data:
                pos = data['position']
                self.latest_pp = np.array([
                    float(pos['x']),
                    float(pos['y']),
                    float(pos['z'])
                ])
                self.latest_pp_time = time.time()
                # self.get_logger().debug(f"PP: {self.latest_pp}")
        except (json.JSONDecodeError, KeyError) as e:
            self.get_logger().warn(f"Failed to parse PP: {e}")

    def update_filter(self):
        """Main filter update loop"""
        if not self.initialized:
            return
        
        self.update_count += 1
        
        # Always predict motion
        self.predict()
        
        # Determine which measurements are available
        has_doa = self.latest_doa is not None
        has_pp = self.latest_pp is not None
        
        # Update with available measurements
        if has_doa and has_pp:
            # Both sensors available - use both
            self.update_with_pp(self.latest_pp)
            self.update_with_doa(*self.latest_doa)
            measurement_type = "DOA+PP"
        elif has_doa:
            # Only DOA available
            self.update_with_doa(*self.latest_doa)
            measurement_type = "DOA"
        elif has_pp:
            # Only PP available
            self.update_with_pp(self.latest_pp)
            measurement_type = "PP"
        else:
            # No measurements - prediction only
            measurement_type = "PREDICTION"
        
        # Check if resampling is needed
        ess = self.effective_sample_size()
        if ess < self.num_particles * self.resample_ratio:
            self.get_logger().debug(f"Low ESS ({ess:.1f}) - resampling")
            self.resample()
        
        # Clear measurements after processing
        self.latest_doa = None
        self.latest_pp = None
        
        # Publish results
        self.publish_state(measurement_type)
        self.publish_particles()
        
        # Log occasionally
        if self.update_count % 100 == 0:
            self.log_status()

    def log_status(self):
        """Log filter status occasionally"""
        if not self.initialized:
            return
        
        # Calculate statistics
        position = np.average(self.particles, axis=0, weights=self.weights)
        velocity = np.average(self.particle_velocities, axis=0, weights=self.weights)
        ess = self.effective_sample_size()
        
        self.get_logger().info(
            f"Filter Status - "
            f"Position: [{position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f}] "
            f"Vel: [{velocity[0]:.1f}, {velocity[1]:.1f}, {velocity[2]:.1f}] "
            f"ESS: {ess:.0f}/{self.num_particles}"
        )

    def publish_state(self, measurement_type=""):
        """Publish the current state estimate"""
        if not self.initialized:
            return
        
        # Calculate weighted average position
        position = np.average(self.particles, axis=0, weights=self.weights)
        
        # Calculate weighted average velocity
        velocity = np.average(self.particle_velocities, axis=0, weights=self.weights)
        
        # Calculate position covariance (uncertainty)
        weighted_diff = self.particles - position
        covariance = np.cov(weighted_diff.T, aweights=self.weights)
        
        # Calculate effective sample size
        ess = self.effective_sample_size()
        
        # Create message
        msg = String()
        msg.data = json.dumps({
            'header': {
                'stamp': {
                    'sec': self.get_clock().now().seconds_nanoseconds()[0],
                    'nanosec': self.get_clock().now().seconds_nanoseconds()[1]
                },
                'frame_id': 'map'
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
            },
            'covariance': covariance.flatten().tolist(),
            'measurement_type': measurement_type,
            'effective_particles': float(ess),
            'timestamp': time.time()
        })
        
        self.state_pub.publish(msg)

    def publish_particles(self):
        """Publish particles for visualization (sampled if too many)"""
        if not self.initialized:
            return
        
        # Limit number of particles for visualization
        max_viz_particles = 500
        if self.num_particles > max_viz_particles:
            # Sample particles according to their weights
            indices = np.random.choice(
                self.num_particles, 
                max_viz_particles, 
                p=self.weights
            )
            viz_particles = self.particles[indices]
        else:
            viz_particles = self.particles
        
        # Create message
        msg = String()
        msg.data = json.dumps({
            'header': {
                'stamp': {
                    'sec': self.get_clock().now().seconds_nanoseconds()[0],
                    'nanosec': self.get_clock().now().seconds_nanoseconds()[1]
                },
                'frame_id': 'map'
            },
            'particles': viz_particles.tolist(),
            'num_particles': len(viz_particles),
            'timestamp': time.time()
        })
        
        self.particles_pub.publish(msg)

def main():
    rclpy.init()
    node = ParticleFilter()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Particle filter shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
