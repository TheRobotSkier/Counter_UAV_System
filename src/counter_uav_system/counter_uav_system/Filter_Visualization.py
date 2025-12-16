#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from std_msgs.msg import Header, String
from geometry_msgs.msg import Point
import json
from collections import deque
import select
import sys
import termios
import tty
import os
plot_range = 100 # meters
class FilterVisualizationNode(Node):
    """Visualization node with real particle data from filter"""
    def __init__(self):
        super().__init__('filter_visualization_node')
        
        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('system_x', 0.0),
                ('system_y', 0.0),
                ('system_z', 0.0),
                ('history_length', 50),
                ('max_particles_to_show', 2000),
                ('particle_alpha', 0.3),
                ('trajectory_linewidth', 2)
            ]
        )
        
        # Get parameters
        self.system_position = np.array([
            self.get_parameter('system_x').value,
            self.get_parameter('system_y').value,
            self.get_parameter('system_z').value
        ])
        self.history_length = self.get_parameter('history_length').value
        self.max_particles = self.get_parameter('max_particles_to_show').value
        self.particle_alpha = self.get_parameter('particle_alpha').value
        self.trajectory_linewidth = self.get_parameter('trajectory_linewidth').value
        
        # Data storage for visualization
        self.true_positions = deque(maxlen=self.history_length)
        self.estimated_positions = deque(maxlen=self.history_length)
        self.doa_measurements = deque(maxlen=1)
        self.pp_measurements = deque(maxlen=1)
        self.particles_history = deque(maxlen=1)  # Store recent particle sets
        
        # Latest data
        self.latest_true_state = None
        self.latest_estimated_state = None
        self.latest_particles = None
        
        # Visualization toggles
        self.show_particles = True
        self.show_doa = True
        self.show_pp = True
        self.show_true_trajectory = True
        self.show_estimated_trajectory = True
        self.show_true_position = True
        self.show_estimated_position = True
        
        # Subscribers - all using String messages
        self.true_state_sub = self.create_subscription(
            String,
            '/drone/true_state',
            self.true_state_callback,
            10
        )
        
        self.filter_state_sub = self.create_subscription(
            String,
            '/filter/state',
            self.filter_state_callback,
            10
        )
        
        self.particles_sub = self.create_subscription(
            String,
            '/filter/particles',
            self.particles_callback,
            10
        )
        
        self.doa_sub = self.create_subscription(
            String,
            '/sensors/doa',
            self.doa_callback,
            10
        )
        
        self.pp_sub = self.create_subscription(
            String,
            '/sensors/point_pillars',
            self.pp_callback,
            10
        )
        
        # Visualization setup
        self.setup_plot()
        
        # Visualization timer
        self.viz_timer = self.create_timer(0.2, self.update_plot)  # 5 Hz update
        
        # Keyboard input timer
        self.keyboard_timer = self.create_timer(0.1, self.check_keyboard_input)
        
        # Store original terminal settings
        self.old_settings = termios.tcgetattr(sys.stdin)
        
        self.get_logger().info("Filter visualization node started with real particle data")
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
        self.ax.set_xlim([-plot_range, plot_range])
        self.ax.set_ylim([-plot_range, plot_range])
        self.ax.set_zlim([0, plot_range])
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
        """Store true state from JSON string"""
        try:
            data = json.loads(msg.data)
            position = np.array([
                data['true_position']['x'],
                data['true_position']['y'],
                data['true_position']['z']
            ])
            if np.all(np.isfinite(position)):
                self.true_positions.append(position)
                self.latest_true_state = position
        except (json.JSONDecodeError, KeyError) as e:
            self.get_logger().warning(f"Failed to parse true state: {e}")

    def filter_state_callback(self, msg):
        """Store estimated state from JSON string"""
        try:
            data = json.loads(msg.data)
            position = np.array([
                data['estimated_position']['x'],
                data['estimated_position']['y'],
                data['estimated_position']['z']
            ])
            if np.all(np.isfinite(position)):
                self.estimated_positions.append(position)
                self.latest_estimated_state = position
        except (json.JSONDecodeError, KeyError) as e:
            self.get_logger().warning(f"Failed to parse filter state: {e}")

    def particles_callback(self, msg):
        """Store real particles from particle filter"""
        try:
            data = json.loads(msg.data)
            if 'particles' in data and data['particles']:
                particles = np.array(data['particles'])
                # Ensure particles are valid
                if particles.size > 0 and np.all(np.isfinite(particles)):
                    self.latest_particles = particles
                    self.particles_history.append(particles.copy())
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.get_logger().warning(f"Failed to parse particles: {e}")

    def doa_callback(self, msg):
        """Store DOA measurements from JSON string"""
        try:
            data = json.loads(msg.data)
            azimuth = np.deg2rad(data['azimuth'])
            elevation = np.deg2rad(data['elevation'])
            
            # Create a point in the DOA direction at 30m distance
            distance = 30.0
            x = distance * np.cos(elevation) * np.cos(azimuth)
            y = distance * np.cos(elevation) * np.sin(azimuth)
            z = distance * np.sin(elevation)
            
            doa_point = np.array([x, y, z])
            if np.all(np.isfinite(doa_point)):
                self.doa_measurements.append(doa_point)
        except (json.JSONDecodeError, KeyError) as e:
            self.get_logger().warning(f"Failed to parse DOA data: {e}")

    def pp_callback(self, msg):
        """Store PointPillars measurements from JSON string"""
        try:
            data = json.loads(msg.data)
            position = np.array([
                data['position']['x'],
                data['position']['y'],
                data['position']['z']
            ])
            if np.all(np.isfinite(position)):
                self.pp_measurements.append(position)
        except (json.JSONDecodeError, KeyError) as e:
            self.get_logger().warning(f"Failed to parse PP data: {e}")

    def update_plot(self):
        """Update visualization with toggles"""
        try:
            self.ax.clear()
            
            self.ax.set_xlim([-plot_range, plot_range])
            self.ax.set_ylim([-plot_range, plot_range])
            self.ax.set_zlim([0, plot_range])
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
            self.ax.scatter(*self.system_position, c='black', s=100, marker='*', 
                          label='System Origin')
            
            # Plot true trajectory
            if self.show_true_trajectory and len(self.true_positions) > 1:
                true_traj = np.array(self.true_positions)
                self.ax.plot(true_traj[:, 0], true_traj[:, 1], true_traj[:, 2], 
                            'g-', linewidth=self.trajectory_linewidth, 
                            label='True Trajectory')
            
            # Plot estimated trajectory
            if self.show_estimated_trajectory and len(self.estimated_positions) > 1:
                est_traj = np.array(self.estimated_positions)
                self.ax.plot(est_traj[:, 0], est_traj[:, 1], est_traj[:, 2], 
                            'b-', linewidth=self.trajectory_linewidth, 
                            label='Estimated Trajectory')
            
            # Plot DOA measurements
            if self.show_doa and len(self.doa_measurements) > 0:
                doa_points = np.array(self.doa_measurements)
                self.ax.scatter(doa_points[:, 0], doa_points[:, 1], doa_points[:, 2],
                              c='orange', s=50, alpha=0.5, marker='^', 
                              label='DOA Measurements')
            
            # Plot PointPillars measurements
            if self.show_pp and len(self.pp_measurements) > 0:
                pp_points = np.array(self.pp_measurements)
                self.ax.scatter(pp_points[:, 0], pp_points[:, 1], pp_points[:, 2],
                              c='purple', s=50, alpha=0.5, marker='s', 
                              label='PP Measurements')
            
            # Plot REAL particles from filter
            if self.show_particles and self.latest_particles is not None:
                # Sample particles if there are too many
                if len(self.latest_particles) > self.max_particles:
                    indices = np.random.choice(len(self.latest_particles), 
                                             self.max_particles, replace=False)
                    particles_to_show = self.latest_particles[indices]
                else:
                    particles_to_show = self.latest_particles
                
                self.ax.scatter(particles_to_show[:, 0], 
                              particles_to_show[:, 1], 
                              particles_to_show[:, 2], 
                              c='red', alpha=self.particle_alpha, s=5, 
                              label='Particles')
            
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
                self.ax.legend(loc='upper left')
            
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