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


# things to implement:
# add 3d point form the point pillars as a measurement update to the particle filter
# add code for when the drone is within 20 meters to begin predicting larger steps for aiming and fireing at the drone


# Update rate in Hz
# ADoA_UPDATE_RATE = ??
# PP_UAV_UPDATE_RATE = ??

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

class UAVDynamicModel:
    """Simple dynamic model for DJI Mavic-like UAV"""
    def __init__(self):
        self.max_speed = 15.0
        self.max_acceleration = 5.0
        self.max_vertical_speed = 4.0
        self.max_vertical_acceleration = 2.0
        self.velocity_decay = 0.95
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

def generate_new_doa_angles(true_position, system_position, noise_level=1.0):
    vector = true_position - system_position
    distance = np.linalg.norm(vector)
    
    if distance > 0:
        direction = vector / distance
        azimuth = np.arctan2(direction[1], direction[0])
        elevation = np.arcsin(direction[2])
        
        azimuth += np.random.normal(0, np.deg2rad(noise_level))
        elevation += np.random.normal(0, np.deg2rad(noise_level))
        
        return np.array([np.rad2deg(azimuth), np.rad2deg(elevation)])
    else:
        return np.array([0, 90])

def setup_plot():
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
    particle_filter = UAVParticleFilter(num_particles=2000)
    
    true_position = np.random.uniform(0, 70, 3)
    true_velocity = np.array([2, 1, 0.5])
    
    fig, ax = setup_plot()
    
    keep_running = True
    use_pp_data = False
    
    def on_key_press_q(event):
        nonlocal keep_running
        if event.key in ['q', 'escape']:
            keep_running = False
    
    def on_key_press_l(event):
        nonlocal use_pp_data
        if event.key in ['l']:
            use_pp_data = not use_pp_data
            mode = "PP+DOA" if use_pp_data else "DOA Only"
            print(f"Switched to {mode} mode")
    
    fig.canvas.mpl_connect('key_press_event', on_key_press_q)
    fig.canvas.mpl_connect('key_press_event', on_key_press_l)
    
    true_positions = []
    estimated_positions = []
    
    frame = 0
    while keep_running:
        plt.cla()
        frame += 1
        
        true_position, true_velocity = particle_filter.dynamic_model.update_state(
            true_position, true_velocity
        )
        true_positions.append(true_position.copy())
        
        doa_data = generate_new_doa_angles(true_position, particle_filter.system_position)
        
        if use_pp_data:
            pp_position = true_position + np.random.normal(0, 0.5, 3)
            estimated_position, estimated_velocity = particle_filter.process_measurement_with_pp(
                doa_data, pp_position
            )
        else:
            estimated_position, estimated_velocity = particle_filter.process_measurement_doa_only(doa_data)
        
        estimated_positions.append(estimated_position.copy())
        
        true_trajectory = np.array(true_positions)
        estimated_trajectory = np.array(estimated_positions)
        
        ax.set_xlim([-70, 70])
        ax.set_ylim([-70, 70])
        ax.set_zlim([0, 70])
        
        mode_display = "PP+DOA" if use_pp_data else "DOA Only"
        ax.set_title(f'UAV Tracking - Frame {frame} - Mode: {mode_display}\n(Q: Quit, L: Toggle Mode)')
        
        if particle_filter.particles is not None:
            ax.scatter(particle_filter.particles[:, 0], 
                      particle_filter.particles[:, 1], 
                      particle_filter.particles[:, 2], 
                      color='r', s=1, alpha=0.3, label='Particles')
        
        if len(true_trajectory) > 1:
            ax.plot(true_trajectory[:, 0], true_trajectory[:, 1], true_trajectory[:, 2],
                   'g-', linewidth=2, label='True Trajectory')
            ax.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], estimated_trajectory[:, 2],
                   'b-', linewidth=2, label='Estimated Trajectory')
        
        ax.scatter(*true_position, color='green', s=100, marker='o', label='True Position')
        ax.scatter(*estimated_position, color='blue', s=100, marker='s', label='Estimated Position')
        
        ax.quiver(*true_position, *true_velocity, color='green', length=5, normalize=True, label='True Velocity')
        ax.quiver(*estimated_position, *estimated_velocity, color='blue', length=5, normalize=True, label='Estimated Velocity')
        
        ax.legend()
        plt.pause(0.01)
    
    plt.show()

if __name__ == "__main__":
    main()