import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
# input will be on the form of:
# From point pillars:
# 1 Position of UAV bounding box (x,y,z) in world coordinates
# 2 Orientation of UAV bounding box in world coordinates (ask about format)
# 3 Dimentions of the bounding box (length, width, height)
# 4 Confidence score
# 5 timestamp

# Aquistic DOA vector:
# 6 azimuth angle X
# 7 elevation angle X 
# 8 confidence score 
# 9 timestamp
# maybe multiple ones of these per frame 
# no previos weight to account for inacurasy 

#additionsal information:
# timestamp
# camera frame position and orientation in world coordinates 

# Output will be used to controll the pan tilt system to point at the UAV:
# 1 Position of UAV bounding box (x,y,z) in world coordinates (ask about format) 
# 2 Orientation of UAV bounding box in world coordinates (ask about format)
# 3 Timestamp
# maybe add confidence score of the estimated position
# maybe add velocity vector of the UAV
# maybe add acceleration vector of the UAV
# maybe add number of particles used in the estimation
# maybe add resampling information

# Update rate in Hz
# ADoA_UPDATE_RATE = ??
# PP_UAV_UPDATE_RATE = ??

class UAVDynamicModel:
    """Simple dynamic model for DJI Mavic-like UAV"""
    def __init__(self):
        # DJI Mavic specifications (approximate)
        self.max_speed = 15.0  # m/s (54 km/h in S-mode)
        self.max_acceleration = 5.0  # m/s² (reasonable for consumer drone)
        self.max_vertical_speed = 4.0  # m/s (ascend/descend)
        self.max_vertical_acceleration = 2.0  # m/s²
        
        # Typical flight characteristics
        self.velocity_decay = 0.95  # Velocity tends to decay without input
        self.position_noise_std = 0.1  # Position uncertainty
        
    def constrain_velocity(self, velocity):
        """Constrain velocity to realistic limits"""
        horizontal_speed = np.linalg.norm(velocity[:2])
        vertical_speed = abs(velocity[2])
        
        # Scale horizontal velocity if exceeds max
        if horizontal_speed > self.max_speed:
            scale = self.max_speed / horizontal_speed
            velocity[0] *= scale
            velocity[1] *= scale
        
        # Constrain vertical velocity
        if vertical_speed > self.max_vertical_speed:
            velocity[2] = np.sign(velocity[2]) * self.max_vertical_speed
            
        return velocity
    
    def constrain_acceleration(self, acceleration):
        """Constrain acceleration to realistic limits"""
        horizontal_accel = np.linalg.norm(acceleration[:2])
        vertical_accel = abs(acceleration[2])
        
        # Scale horizontal acceleration if exceeds max
        if horizontal_accel > self.max_acceleration:
            scale = self.max_acceleration / horizontal_accel
            acceleration[0] *= scale
            acceleration[1] *= scale
        
        # Constrain vertical acceleration
        if vertical_accel > self.max_vertical_acceleration:
            acceleration[2] = np.sign(acceleration[2]) * self.max_vertical_acceleration
            
        return acceleration
    
    def update_state(self, position, velocity, dt=0.1):
        """Update UAV state with dynamic constraints"""
        # Add some random acceleration (simulating control inputs)
        random_accel = np.random.normal(0, 0.5, 3)
        random_accel = self.constrain_acceleration(random_accel)
        
        # Update velocity with acceleration and decay
        velocity = velocity * self.velocity_decay + random_accel * dt
        velocity = self.constrain_velocity(velocity)
        
        # Update position
        new_position = position + velocity * dt
        
        # Add small position noise
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
        #tissemand
        # Particle state: each particle has position and velocity
        self.particles = None
        self.velocities = None
        self.weights = None
        
    def initialize_particles(self, doa_data, max_distance=70):
        """Initialize particles along DOA direction with random velocities"""
        direction_vector = self.doa_to_world_coordinates(doa_data)
        
        positions = []
        velocities = []
        
        for _ in range(self.num_particles):
            # Random distance along DOA
            distance = np.random.uniform(5, max_distance)
            position = self.system_position + direction_vector * distance
            
            # Random initial velocity (constrained)
            velocity = np.random.uniform(-2, 2, 3)
            velocity = self.dynamic_model.constrain_velocity(velocity)
            
            positions.append(position)
            velocities.append(velocity)
        
        self.particles = np.array(positions)
        self.velocities = np.array(velocities)
        self.weights = np.ones(self.num_particles) / self.num_particles
        
    def doa_to_world_coordinates(self, doa_data):
        """Convert DOA angles to world coordinates direction vector"""
        azimuth = np.deg2rad(doa_data[0])
        elevation = np.deg2rad(doa_data[1])
        
        x = np.cos(elevation) * np.cos(azimuth)
        y = np.cos(elevation) * np.sin(azimuth)
        z = np.sin(elevation)
        
        return np.array([x, y, z])
    
    def predict(self, dt=0.1):
        """Predict particle states using dynamic model"""
        for i in range(self.num_particles):
            self.particles[i], self.velocities[i] = self.dynamic_model.update_state(
                self.particles[i], self.velocities[i], dt
            )
    
    def update_doa(self, doa_data, angle_std_deg=5):
        """Update particle weights based on DOA measurement"""
        doa_vector = self.doa_to_world_coordinates(doa_data)
        angle_std_rad = np.deg2rad(angle_std_deg)
        
        for i in range(self.num_particles):
            # Vector from system to particle
            particle_vector = self.particles[i] - self.system_position
            particle_distance = np.linalg.norm(particle_vector)
            
            if particle_distance > 0:
                particle_direction = particle_vector / particle_distance
                
                # Calculate angle between DOA vector and particle direction
                dot_product = np.clip(np.dot(doa_vector, particle_direction), -1, 1)
                angle_error = np.arccos(dot_product)
                
                # Weight based on angular error (Gaussian)
                weight = np.exp(-0.5 * (angle_error / angle_std_rad) ** 2)
                self.weights[i] = weight
        
        # Normalize weights
        if np.sum(self.weights) > 0:
            self.weights /= np.sum(self.weights)
        else:
            self.weights = np.ones(self.num_particles) / self.num_particles
    
    def resample(self):
        """Resample particles based on weights"""
        indices = np.random.choice(
            self.num_particles, 
            size=self.num_particles, 
            p=self.weights
        )
        
        self.particles = self.particles[indices]
        self.velocities = self.velocities[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles
    
    def estimate_position(self):
        """Estimate UAV position as weighted average of particles"""
        return np.average(self.particles, weights=self.weights, axis=0)
    
    def estimate_velocity(self):
        """Estimate UAV velocity as weighted average of particle velocities"""
        return np.average(self.velocities, weights=self.weights, axis=0)
    
    def process_measurement(self, doa_data, dt=0.1):
        """Full particle filter update cycle"""
        if self.particles is None:
            self.initialize_particles(doa_data)
        else:
            self.predict(dt)
            self.update_doa(doa_data)
            self.resample()
        
        return self.estimate_position(), self.estimate_velocity()

def generate_new_doa_angles(true_position, system_position, noise_level=1.0):
    """Generate DOA angles from true position with noise"""
    vector = true_position - system_position
    distance = np.linalg.norm(vector)
    
    if distance > 0:
        direction = vector / distance
        
        # Convert to spherical coordinates
        azimuth = np.arctan2(direction[1], direction[0])
        elevation = np.arcsin(direction[2])
        
        # Add noise
        azimuth += np.random.normal(0, np.deg2rad(noise_level))
        elevation += np.random.normal(0, np.deg2rad(noise_level))
        
        return np.array([np.rad2deg(azimuth), np.rad2deg(elevation)])
    else:
        return np.array([0, 90])

def setup_plot():
    """Setup the 3D plot for visualization"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-50, 50])
    ax.set_ylim([-50, 50])
    ax.set_zlim([0, 50])
    ax.set_xlabel('X axis (meters)')
    ax.set_ylabel('Y axis (meters)')
    ax.set_zlabel('Z axis (meters)')
    ax.set_title('UAV Tracking with Dynamic Model Particle Filter')
    return fig, ax

def main():
    # Initialize particle filter
    particle_filter = UAVParticleFilter(num_particles=2000)
    
    # Simulated true UAV trajectory
    #true_position = np.array([10, 10, 15])
    #random start position within 70 meters of origin
    true_position = np.random.uniform(0, 70, 3)
    
    true_velocity = np.array([2, 1, 0.5])
    
    # Setup visualization
    fig, ax = setup_plot()
    
    # Add key press event handler
    keep_running = True
    
    def on_key_press(event):
        nonlocal keep_running
        if event.key in ['q', 'escape']:
            keep_running = False
    
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    
    # Storage for plotting
    true_positions = []
    estimated_positions = []
    
    frame = 0
    while keep_running:
        plt.cla()
        frame += 1
        
        # Update true position using dynamic model (only if not manually updated this frame)
        true_position, true_velocity = particle_filter.dynamic_model.update_state(
            true_position, true_velocity
        )
        true_positions.append(true_position.copy())
        
        # Generate DOA measurement from true position
        doa_data = generate_new_doa_angles(true_position, particle_filter.system_position)
        
        # Update particle filter
        estimated_position, estimated_velocity = particle_filter.process_measurement(doa_data)
        estimated_positions.append(estimated_position.copy())
        
        # Convert to arrays for plotting
        true_trajectory = np.array(true_positions)
        estimated_trajectory = np.array(estimated_positions)
        
        # Update plot
        ax.set_xlim([-70, 70])
        ax.set_ylim([-70, 70])
        ax.set_zlim([0, 70])
        ax.set_title(f'UAV Tracking - Frame {frame} (Q: Quit)')
        
        # Plot particles
        if particle_filter.particles is not None:
            ax.scatter(particle_filter.particles[:, 0], 
                      particle_filter.particles[:, 1], 
                      particle_filter.particles[:, 2], 
                      color='r', s=1, alpha=0.3, label='Particles')
        
        # Plot trajectories
        if len(true_trajectory) > 1:
            ax.plot(true_trajectory[:, 0], true_trajectory[:, 1], true_trajectory[:, 2],
                   'g-', linewidth=2, label='True Trajectory')
            ax.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], estimated_trajectory[:, 2],
                   'b-', linewidth=2, label='Estimated Trajectory')
        
        # Plot current positions
        ax.scatter(*true_position, color='green', s=100, marker='o', label='True Position')
        ax.scatter(*estimated_position, color='blue', s=100, marker='s', label='Estimated Position')
        
        # Add velocity vectors
        ax.quiver(*true_position, *true_velocity, color='green', length=5, normalize=True, label='True Velocity')
        ax.quiver(*estimated_position, *estimated_velocity, color='blue', length=5, normalize=True, label='Estimated Velocity')
        
        ax.legend()
        plt.pause(0.1)
    
    plt.show()

if __name__ == "__main__":
    main()