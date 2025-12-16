# particle_filter_simple.py
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import json
from std_msgs.msg import String
from message_definitions import ParticleFilterParams
import time

class ParticleFilter(Node):
    """Simple particle filter - just processes whatever sensor data it receives"""
    def __init__(self):
        super().__init__('particle_filter')
        
        # Parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('num_particles', 1000),
                ('doa_std_rad', 0.2),      # 5.7 degrees
                ('pp_std_m', 0.5),
                ('prediction_dt', 0.1),
                ('position_diffusion', 1.0),  # Brownian motion strength
                ('system_x', 0.0),
                ('system_y', 0.0),
                ('system_z', 0.0)
            ]
        )
        
        # Initialize
        self.system_position = np.array([
            self.get_parameter('system_x').value,
            self.get_parameter('system_y').value,
            self.get_parameter('system_z').value
        ])
        
        self.num_particles = self.get_parameter('num_particles').value
        self.doa_std = self.get_parameter('doa_std_rad').value
        self.pp_std = self.get_parameter('pp_std_m').value
        
        self.particles = None
        self.weights = None
        
        # Subscribers (simple - just store latest data)
        self.create_subscription(String, '/sensors/doa', self.doa_callback, 10)
        self.create_subscription(String, '/sensors/point_pillars', self.pp_callback, 10)
        
        # Publishers
        self.state_pub = self.create_publisher(String, '/filter/state', 10)
        self.particles_pub = self.create_publisher(String, '/filter/particles', 10)
        
        # Timer for prediction (runs at fixed rate)
        self.create_timer(self.get_parameter('prediction_dt').value, self.update)
        
        # Store latest measurements
        self.latest_doa = None
        self.latest_pp = None
        
        self.get_logger().info(f"Particle filter started with {self.num_particles} particles")

    def initialize_particles(self):
        """Initialize particles uniformly around possible area"""
        # Spread particles in a 200m cube centered on sensor
        self.particles = np.random.uniform(-100, 100, (self.num_particles, 3))
        self.particles[:, 2] = np.abs(self.particles[:, 2])  # Keep above ground
        self.weights = np.ones(self.num_particles) / self.num_particles

    def brownian_predict(self):
        """Simple Brownian motion prediction"""
        if self.particles is None:
            self.initialize_particles()
        
        # Add Gaussian noise (Brownian motion)
        diffusion = self.get_parameter('position_diffusion').value
        dt = self.get_parameter('prediction_dt').value
        noise_std = np.sqrt(2 * diffusion * dt)
        
        self.particles += np.random.normal(0, noise_std, self.particles.shape)
        self.particles[:, 2] = np.maximum(self.particles[:, 2], 0.1)

    def update_with_doa(self, azimuth_deg, elevation_deg):
        azimuth = np.deg2rad(azimuth_deg)
        elevation = np.deg2rad(elevation_deg)
        
        # Convert particles to angles
        vectors = self.particles - self.system_position
        distances = np.linalg.norm(vectors, axis=1)
        
        # Avoid division by zero
        valid = distances > 0.1
        pred_azimuth = np.zeros_like(distances)
        pred_elevation = np.zeros_like(distances)
        
        if np.any(valid):
            pred_azimuth[valid] = np.arctan2(vectors[valid, 1], vectors[valid, 0])
            pred_elevation[valid] = np.arcsin(vectors[valid, 2] / distances[valid])
        
        # Calculate angular differences
        az_diff = np.arctan2(np.sin(pred_azimuth - azimuth), np.cos(pred_azimuth - azimuth))
        el_diff = np.arctan2(np.sin(pred_elevation - elevation), np.cos(pred_elevation - elevation))
        
        # Update weights
        az_likelihood = np.exp(-0.5 * (az_diff / self.doa_std) ** 2)
        el_likelihood = np.exp(-0.5 * (el_diff / self.doa_std) ** 2)
        self.weights *= az_likelihood * el_likelihood

    def update_with_pp(self, pp_position):
        """Update weights based on PointPillars measurement"""
        distances = np.linalg.norm(self.particles - pp_position, axis=1)
        likelihood = np.exp(-0.5 * (distances / self.pp_std) ** 2)
        self.weights *= likelihood
        self.normalize_weights()

    def normalize_weights(self):
        """Normalize particle weights"""
        total = np.sum(self.weights)
        if total > 0:
            self.weights /= total
        else:
            self.weights = np.ones(self.num_particles) / self.num_particles

    def resample_if_needed(self):
        """Resample when effective sample size is low"""
        ess = 1.0 / np.sum(self.weights ** 2)
        if ess < self.num_particles / 2:
            self.resample()

    def resample(self):
        """Systematic resampling"""
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def doa_callback(self, msg):
        """Simply store DOA data when it arrives"""
        try:
            data = json.loads(msg.data)
            self.latest_doa = (data['azimuth'], data['elevation'])
        except:
            pass

    def pp_callback(self, msg):
        """Simply store PP data when it arrives"""
        try:
            data = json.loads(msg.data)
            pos = data['position']
            self.latest_pp = np.array([pos['x'], pos['y'], pos['z']])
        except:
            pass

    def update(self):
        """Main filter update loop"""
        # Always predict (Brownian motion)
        self.brownian_predict()
        
        # Update with measurements if available
        if self.latest_pp is not None:
            self.update_with_pp(self.latest_pp)
            self.latest_pp = None
        elif self.latest_doa is not None:
            self.update_with_doa(*self.latest_doa)
            self.latest_doa = None
        
        # Resample if needed
        self.resample_if_needed()
        
        # Publish results
        self.publish_state()
        self.publish_particles()

    def publish_state(self):
        """Publish estimated state"""
        if self.particles is not None:
            # Weighted average
            estimate = np.average(self.particles, axis=0, weights=self.weights)
            
            msg = String()
            msg.data = json.dumps({
                'estimated_position': {
                    'x': float(estimate[0]),
                    'y': float(estimate[1]),
                    'z': float(estimate[2])
                },
                'header': {
                    'stamp': {
                        'sec': self.get_clock().now().to_msg().sec,
                        'nanosec': self.get_clock().now().to_msg().nanosec
                    }
                }
            })
            self.state_pub.publish(msg)

    def publish_particles(self):
        """Publish particles for visualization"""
        if self.particles is not None:
            msg = String()
            msg.data = json.dumps({
                'particles': self.particles.tolist(),
                'weights': self.weights.tolist(),
                'timestamp': time.time()
            })
            self.particles_pub.publish(msg)

def main():
    rclpy.init()
    node = ParticleFilter()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Filter stopped")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()