#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from geometry_msgs.msg import Point, Vector3
from uav_interfaces.msg import DroneState, DOAData, PointPillarsData, ParticleFilterState
from collections import deque
import threading
import select
import sys
import termios
import tty
import os

class FilterVisualizationNode(Node):
    """Separate node for visualization with keyboard controls"""
    def __init__(self):
        super().__init__('filter_visualization_node')
        
        # Data storage for visualization
        self.true_positions = deque(maxlen=200)
        self.estimated_positions = deque(maxlen=200)
        self.doa_measurements = deque(maxlen=5)
        self.pp_measurements = deque(maxlen=5)
        self.particles_history = deque(maxlen=200)  # Store recent particle sets
        
        # Latest data
        self.latest_true_state = None
        self.latest_estimated_state = None
        self.latest_particles = None
        
        # System position
        self.system_position = np.array([0, 0, 0])
        
        # Visualization toggles
        self.show_particles = True
        self.show_doa = True
        self.show_pp = True
        self.show_true_trajectory = True
        self.show_estimated_trajectory = True
        self.show_true_position = True
        self.show_estimated_position = True
        
        # Subscribers
        self.true_state_sub = self.create_subscription(
            DroneState,
            '/drone/true_state',
            self.true_state_callback,
            10
        )
        
        self.filter_state_sub = self.create_subscription(
            ParticleFilterState,
            '/filter/state',
            self.filter_state_callback,
            10
        )
        
        self.particles_sub = self.create_subscription(
            ParticleFilterState,
            '/filter/particles',
            self.particles_callback,
            10
        )
        
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
        
        # Visualization setup
        self.setup_plot()
        
        # Visualization timer - runs at lower frequency
        self.viz_timer = self.create_timer(0.2, self.update_plot)  # 5 Hz update
        
        # Keyboard input timer
        self.keyboard_timer = self.create_timer(0.1, self.check_keyboard_input)
        
        # Store original terminal settings
        self.old_settings = termios.tcgetattr(sys.stdin)
        
        self.get_logger().info("Filter visualization node started with keyboard controls")
        self.print_controls()

    def print_controls(self):
        """Print keyboard controls to terminal"""
        controls = """
        === VISUALIZATION CONTROLS ===
        [P] - Toggle Particles
        [D] - Toggle DOA Measurements  
        [M] - Toggle PointPillars Measurements
        [T] - Toggle True Trajectory
        [E] - Toggle Estimated Trajectory
        [1] - Toggle True Position
        [2] - Toggle Estimated Position
        [A] - Toggle ALL elements
        [C] - Clear all trajectories
        [H] - Show this help
        [Q] - Quit visualization
        ==============================
        """
        print(controls)

    def setup_plot(self):
        """Initialize 3D plot for visualization"""
        plt.ion()
        self.fig = plt.figure(figsize=(14, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim([-50, 50])
        self.ax.set_ylim([-50, 50])
        self.ax.set_zlim([0, 50])
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('Particle Filter - 3D Tracking (Press H for controls)')

    def setup_keyboard_listening(self):
        """Set up terminal for non-blocking keyboard input"""
        try:
            tty.setraw(sys.stdin.fileno())
        except:
            pass

    def restore_terminal(self):
        """Restore terminal settings"""
        try:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
        except:
            pass

    def check_keyboard_input(self):
        """Check for keyboard input without blocking"""
        try:
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                key = sys.stdin.read(1).lower()
                self.handle_keypress(key)
        except:
            pass

    def handle_keypress(self, key):
        """Handle keyboard input"""
        if key == 'p':
            self.show_particles = not self.show_particles
            self.get_logger().info(f"Particles: {'ON' if self.show_particles else 'OFF'}")
            
        elif key == 'd':
            self.show_doa = not self.show_doa
            self.get_logger().info(f"DOA Measurements: {'ON' if self.show_doa else 'OFF'}")
            
        elif key == 'm':
            self.show_pp = not self.show_pp
            self.get_logger().info(f"PointPillars Measurements: {'ON' if self.show_pp else 'OFF'}")
            
        elif key == 't':
            self.show_true_trajectory = not self.show_true_trajectory
            self.get_logger().info(f"True Trajectory: {'ON' if self.show_true_trajectory else 'OFF'}")
            
        elif key == 'e':
            self.show_estimated_trajectory = not self.show_estimated_trajectory
            self.get_logger().info(f"Estimated Trajectory: {'ON' if self.show_estimated_trajectory else 'OFF'}")
            
        elif key == '1':
            self.show_true_position = not self.show_true_position
            self.get_logger().info(f"True Position: {'ON' if self.show_true_position else 'OFF'}")
            
        elif key == '2':
            self.show_estimated_position = not self.show_estimated_position
            self.get_logger().info(f"Estimated Position: {'ON' if self.show_estimated_position else 'OFF'}")
            
        elif key == 'a':
            # Toggle all elements
            all_on = (self.show_particles and self.show_doa and self.show_pp and 
                     self.show_true_trajectory and self.show_estimated_trajectory and
                     self.show_true_position and self.show_estimated_position)
            
            new_state = not all_on
            self.show_particles = new_state
            self.show_doa = new_state
            self.show_pp = new_state
            self.show_true_trajectory = new_state
            self.show_estimated_trajectory = new_state
            self.show_true_position = new_state
            self.show_estimated_position = new_state
            
            state_str = "ON" if new_state else "OFF"
            self.get_logger().info(f"All elements: {state_str}")
            
        elif key == 'c':
            # Clear trajectories
            self.true_positions.clear()
            self.estimated_positions.clear()
            self.doa_measurements.clear()
            self.pp_measurements.clear()
            self.particles_history.clear()
            self.get_logger().info("All trajectories cleared")
            
        elif key == 'h':
            self.print_controls()
            
        elif key == 'q':
            self.get_logger().info("Quitting visualization...")
            self.restore_terminal()
            plt.close('all')
            raise KeyboardInterrupt

    def true_state_callback(self, msg):
        """Store true state for visualization"""
        if (np.isfinite(msg.true_position.x) and np.isfinite(msg.true_position.y) and 
            np.isfinite(msg.true_position.z)):
            position = np.array([msg.true_position.x, msg.true_position.y, msg.true_position.z])
            self.true_positions.append(position)
            self.latest_true_state = position

    def filter_state_callback(self, msg):
        """Store estimated state for visualization"""
        position = np.array([msg.estimated_position.x, msg.estimated_position.y, msg.estimated_position.z])
        self.estimated_positions.append(position)
        self.latest_estimated_state = position

    def particles_callback(self, msg):
        """Store particles for visualization"""
        # For now, we'll generate some dummy particles around the estimated position
        # In practice, you should create a proper particle message type
        if self.latest_estimated_state is not None:
            # Generate particles around current estimate
            num_particles = 100
            particles = np.random.normal(self.latest_estimated_state, 5.0, (num_particles, 3))
            particles[:, 2] = np.maximum(particles[:, 2], 0.1)  # Keep above ground
            self.latest_particles = particles
            self.particles_history.append(particles.copy())

    def doa_callback(self, msg):
        """Store DOA measurements for visualization"""
        azimuth_rad = np.deg2rad(msg.azimuth)
        elevation_rad = np.deg2rad(msg.elevation)
        x = 30 * np.cos(elevation_rad) * np.cos(azimuth_rad)
        y = 30 * np.cos(elevation_rad) * np.sin(azimuth_rad)
        z = 30 * np.sin(elevation_rad)
        self.doa_measurements.append(np.array([x, y, z]))

    def pp_callback(self, msg):
        """Store PointPillars measurements for visualization"""
        if (np.isfinite(msg.position.x) and np.isfinite(msg.position.y) and np.isfinite(msg.position.z)):
            position = np.array([msg.position.x, msg.position.y, msg.position.z])
            self.pp_measurements.append(position)

    def update_plot(self):
        """Update visualization with toggles"""
        try:
            self.ax.clear()
            
            self.ax.set_xlim([-50, 50])
            self.ax.set_ylim([-50, 50])
            self.ax.set_zlim([0, 50])
            self.ax.set_xlabel('X (m)')
            self.ax.set_ylabel('Y (m)')
            self.ax.set_zlabel('Z (m)')
            
            # Update title with status
            status = []
            if self.show_particles: status.append("Particles")
            if self.show_doa: status.append("DOA")
            if self.show_pp: status.append("PP")
            if self.show_true_trajectory: status.append("True")
            if self.show_estimated_trajectory: status.append("Est")
            
            title = f'Particle Filter - Showing: {", ".join(status) if status else "NOTHING"}'
            self.ax.set_title(title)
            
            # Plot system origin (always visible)
            self.ax.scatter(*self.system_position, c='black', s=100, marker='*', label='System Origin')
            
            # Plot true trajectory
            if self.show_true_trajectory and len(self.true_positions) > 1:
                true_traj = np.array(self.true_positions)
                self.ax.plot(true_traj[:, 0], true_traj[:, 1], true_traj[:, 2], 
                            'g-', linewidth=2, label='True Trajectory')
            
            # Plot estimated trajectory
            if self.show_estimated_trajectory and len(self.estimated_positions) > 1:
                est_traj = np.array(self.estimated_positions)
                self.ax.plot(est_traj[:, 0], est_traj[:, 1], est_traj[:, 2], 
                            'b-', linewidth=2, label='Estimated Trajectory')
            
            # Plot DOA measurements
            if self.show_doa and len(self.doa_measurements) > 0:
                doa_points = np.array(self.doa_measurements)
                self.ax.scatter(doa_points[:, 0], doa_points[:, 1], doa_points[:, 2],
                              c='orange', s=50, alpha=0.5, marker='^', label='DOA Measurements')
            
            # Plot PointPillars measurements
            if self.show_pp and len(self.pp_measurements) > 0:
                pp_points = np.array(self.pp_measurements)
                self.ax.scatter(pp_points[:, 0], pp_points[:, 1], pp_points[:, 2],
                              c='purple', s=50, alpha=0.5, marker='s', label='PP Measurements')
            
            # Plot particles
            if self.show_particles and self.latest_particles is not None:
                self.ax.scatter(self.latest_particles[:, 0], 
                              self.latest_particles[:, 1], 
                              self.latest_particles[:, 2], 
                              c='red', alpha=0.3, s=5, label='Particles')
            
            # Plot current true position
            if self.show_true_position and self.latest_true_state is not None:
                self.ax.scatter(*self.latest_true_state, c='green', s=100, 
                              marker='o', label='True Position')
            
            # Plot current estimate
            if self.show_estimated_position and self.latest_estimated_state is not None:
                self.ax.scatter(*self.latest_estimated_state, c='blue', s=100, 
                              marker='s', label='Current Estimate')
            
            # Only show legend if there are visible elements
            if (self.show_true_trajectory or self.show_estimated_trajectory or 
                self.show_doa or self.show_pp or self.show_particles or
                self.show_true_position or self.show_estimated_position):
                self.ax.legend()
            
            plt.draw()
            plt.pause(0.001)
            
        except Exception as e:
            self.get_logger().warning(f"Visualization update error: {str(e)}")

    def destroy_node(self):
        """Cleanup when node is destroyed"""
        self.restore_terminal()
        super().destroy_node()

def main():
    rclpy.init()
    node = FilterVisualizationNode()
    
    try:
        # Set up non-blocking keyboard input
        node.setup_keyboard_listening()
        
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Filter visualization node shutting down...")
    except Exception as e:
        node.get_logger().error(f"Unexpected error: {str(e)}")
    finally:
        node.restore_terminal()
        node.destroy_node()
        rclpy.shutdown()
        plt.close('all')

if __name__ == "__main__":
    main()
    