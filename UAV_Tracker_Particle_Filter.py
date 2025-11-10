import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
# Update rate in Hz
# ADoA_UPDATE_RATE = ??
# PP_UAV_UPDATE_RATE = ??
class UAVParticleFilter:
    def __init__(self, system_position=np.array([0, 0, 0]), 
                 system_orientation=np.array([0, 0, 0])):
        self.system_position = system_position
        self.system_orientation = system_orientation
        
    def doa_to_world_coordinates(self, doa_data):
        """Convert DOA angles to world coordinates direction vector"""
        azimuth = np.deg2rad(doa_data[0])
        elevation = np.deg2rad(doa_data[1])
        
        # Convert spherical to Cartesian coordinates
        x = np.cos(elevation) * np.cos(azimuth)
        y = np.cos(elevation) * np.sin(azimuth)
        z = np.sin(elevation)
        
        # Return direction vector (unit vector)
        return np.array([x, y, z])
    
    def initialize_particles_along_doa(self, doa_data, num_particles=1000, max_distance=70):
        """Initialize particles along the DOA direction vector"""
        direction_vector = self.doa_to_world_coordinates(doa_data)
        
        particles = []
        for _ in range(num_particles):
            distance = np.random.uniform(0, max_distance)
            particle_position = self.system_position + direction_vector * distance
            particles.append(particle_position)
        
        return np.array(particles)
    
    def spread_particles_around_doa(self, particles, angle_spread_deg=5):
        """Add angular spread to particles to account for DOA uncertainty"""
        angle_spread_rad = np.deg2rad(angle_spread_deg)
        spread_particles = []
        
        for particle in particles:
            # Random angular perturbation
            theta = np.random.uniform(-angle_spread_rad, angle_spread_rad)
            phi = np.random.uniform(-angle_spread_rad, angle_spread_rad)
            
            # Convert spherical perturbation to Cartesian
            dx = np.sin(theta) * np.cos(phi)
            dy = np.sin(theta) * np.sin(phi)
            dz = np.cos(theta)
            
            perturbed_particle = particle + np.array([dx, dy, dz])
            spread_particles.append(perturbed_particle)
        
        return np.array(spread_particles)
    
    def update_particles(self, doa_data, num_particles=1000, max_distance=70, angle_spread_deg=5):
        """Update particles based on new DOA data"""
        particles = self.initialize_particles_along_doa(doa_data, num_particles, max_distance)
        spread_particles = self.spread_particles_around_doa(particles, angle_spread_deg)
        return particles, spread_particles

def generate_new_doa_angles(previous_doa, step_size=0.5, noise_level=1.0):
    """Generate new DOA angles with movement and noise"""
    new_azimuth = previous_doa[0] + np.random.uniform(-noise_level, noise_level) + step_size
    new_elevation = previous_doa[1] + np.random.uniform(-noise_level, noise_level) + step_size
    return np.array([new_azimuth, new_elevation])

def setup_plot():
    """Setup the 3D plot for visualization"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-80, 80])
    ax.set_ylim([-80, 80])
    ax.set_zlim([0, 80])
    ax.set_xlabel('X axis (meters)')
    ax.set_ylabel('Y axis (meters)')
    ax.set_zlabel('Z axis (meters)')
    ax.set_title('UAV Particle Filter Visualization')
    return fig, ax

def main():
    # Initialize particle filter
    particle_filter = UAVParticleFilter()
    
    # Initial DOA data
    current_doa = np.array([0, 90])
    
    # Setup visualization
    fig, ax = setup_plot()
    
    # Simulation loop
    for i in range(100):
        plt.cla()
        
        # Generate new DOA data
        current_doa = generate_new_doa_angles(current_doa)
        
        # Update particles
        particles, spread_particles = particle_filter.update_particles(current_doa)
        
        # Update plot
        ax.set_xlim([-80, 80])
        ax.set_ylim([-80, 80])
        ax.set_zlim([0, 80])
        ax.set_xlabel('X axis (meters)')
        ax.set_ylabel('Y axis (meters)')
        ax.set_zlabel('Z axis (meters)')
        ax.set_title(f'UAV Particle Filter Visualization - Frame {i+1}')
        
        ax.scatter(particles[:, 0], particles[:, 1], particles[:, 2], 
                  color='r', s=1, label='Particles')
        ax.scatter(spread_particles[:, 0], spread_particles[:, 1], spread_particles[:, 2], 
                  color='g', s=1, label='Spread Particles')
        ax.legend()
        
        plt.pause(0.1)
    
    plt.show()

# Placeholder functions for future implementation
def acoustic_doa_data(aq_data_input):
    """Process acoustic DOA data"""
    return aq_data_input

def point_pillar_uav_data(pp_data_input):
    """Process Point Pillar UAV data"""
    return pp_data_input

def uav_particle_filter_full(doa_data, pp_data):
    """Full particle filter implementation (placeholder)"""
    # This would integrate both DOA and Point Pillar data
    estimated_uav_position = np.array([0, 0, 0])
    estimated_uav_orientation = np.array([0, 0, 0])
    return estimated_uav_position, estimated_uav_orientation

if __name__ == "__main__":
    main()
