#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt
from std_msgs.msg import String
import json
from collections import deque
import sys

class SensorRangeVisualization(Node):
    """Visualization showing different sensor ranges and detection status"""
    def __init__(self):
        super().__init__('sensor_range_viz')
        
        # Store data
        self.true_positions = deque(maxlen=100)
        self.estimated_positions = deque(maxlen=100)
        self.particles = None
        
        # Sensor parameters
        self.doa_range = 90.0    # DOA range in meters
        self.pp_range = 70.0     # PointPillars range in meters
        
        # Detection flags
        self.doa_detected = False
        self.pp_detected = False
        
        # Last known distances
        self.last_distance = 0.0
        
        # Setup plot
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(14, 10), subplot_kw={'projection': '3d'})
        
        # Set plot limits
        plot_limit = 100  # Show up to 150m for spawn at 100m
        self.ax.set_xlim([-plot_limit, plot_limit])
        self.ax.set_ylim([-plot_limit, plot_limit])
        self.ax.set_zlim([0, plot_limit])
        
        self.ax.set_xlabel('X (m)', fontsize=12)
        self.ax.set_ylabel('Y (m)', fontsize=12)
        self.ax.set_zlabel('Z (m)', fontsize=12)
        
        # Draw sensor ranges
        self.draw_sensor_ranges()
        
        # Subscribe to topics
        self.create_subscription(String, '/drone/true_state', self.true_cb, 10)
        self.create_subscription(String, '/filter/state', self.est_cb, 10)
        self.create_subscription(String, '/filter/particles', self.particles_cb, 10)
        self.create_subscription(String, '/sensors/doa', self.doa_status_cb, 10)
        self.create_subscription(String, '/sensors/point_pillars', self.pp_status_cb, 10)
        
        # Update plot at 5Hz
        self.create_timer(0.2, self.update_plot)
        
        print("\n" + "="*60)
        print("SENSOR RANGE VISUALIZATION")
        print(f"  DOA Sensor Range: {self.doa_range}m (Blue)")
        print(f"  PointPillars Range: {self.pp_range}m (Red)")
        print("="*60)
        print("Drone Color Legend:")
        print("  RED: >90m (No detection)")
        print("  YELLOW: 70-90m (DOA only)")
        print("  GREEN: ≤70m (DOA + PointPillars)")
        print("="*60)
        print("Close window to stop.\n")

    def draw_sensor_ranges(self):
        """Draw transparent spheres for both sensor ranges"""
        # DOA range sphere (90m) - Blue
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        
        x = self.doa_range * np.outer(np.cos(u), np.sin(v))
        y = self.doa_range * np.outer(np.sin(u), np.sin(v))
        z = self.doa_range * np.outer(np.ones(np.size(u)), np.cos(v))
        
        self.ax.plot_wireframe(x, y, z, color='blue', alpha=0.15, linewidth=0.8, 
                              label=f'DOA Range ({self.doa_range}m)')
        
        # PointPillars range sphere (70m) - Red
        x_pp = self.pp_range * np.outer(np.cos(u), np.sin(v))
        y_pp = self.pp_range * np.outer(np.sin(u), np.sin(v))
        z_pp = self.pp_range * np.outer(np.ones(np.size(u)), np.cos(v))
        
        self.ax.plot_wireframe(x_pp, y_pp, z_pp, color='red', alpha=0.15, linewidth=0.8,
                              label=f'PP Range ({self.pp_range}m)')

    def get_detection_status(self, distance):
        """Determine detection status based on distance"""
        self.doa_detected = distance <= self.doa_range
        self.pp_detected = distance <= self.pp_range
        
        if distance > self.doa_range:
            return "OUT OF RANGE", "red", 100
        elif distance > self.pp_range:
            return "DOA ONLY", "yellow", 120
        else:
            return "BOTH SENSORS", "green", 150

    def true_cb(self, msg):
        """Store true position"""
        try:
            data = json.loads(msg.data)
            pos = data['true_position']
            self.true_positions.append([pos['x'], pos['y'], pos['z']])
            
            # Update distance
            position = np.array([pos['x'], pos['y'], pos['z']])
            self.last_distance = np.linalg.norm(position)
        except:
            pass

    def est_cb(self, msg):
        """Store estimated position"""
        try:
            data = json.loads(msg.data)
            pos = data['estimated_position']
            self.estimated_positions.append([pos['x'], pos['y'], pos['z']])
        except:
            pass

    def particles_cb(self, msg):
        """Store particles"""
        try:
            data = json.loads(msg.data)
            self.particles = np.array(data['particles'])
        except:
            pass

    def doa_status_cb(self, msg):
        """Update DOA detection status"""
        try:
            data = json.loads(msg.data)
            self.doa_detected = data.get('in_range', False)
        except:
            pass

    def pp_status_cb(self, msg):
        """Update PointPillars detection status"""
        try:
            data = json.loads(msg.data)
            self.pp_detected = data.get('in_range', False)
        except:
            pass

    def update_plot(self):
        """Update the 3D plot"""
        try:
            # Check if window is still open
            if not plt.fignum_exists(self.fig.number):
                raise KeyboardInterrupt
            
            self.ax.clear()
            
            # Set plot limits
            plot_limit = 150
            self.ax.set_xlim([-plot_limit, plot_limit])
            self.ax.set_ylim([-plot_limit, plot_limit])
            self.ax.set_zlim([0, plot_limit])
            self.ax.set_xlabel('X (m)', fontsize=12)
            self.ax.set_ylabel('Y (m)', fontsize=12)
            self.ax.set_zlabel('Z (m)', fontsize=12)
            
            # Redraw sensor ranges
            self.draw_sensor_ranges()
            
            # Plot sensor position at origin
            self.ax.scatter(0, 0, 0, c='black', s=400, marker='*', 
                          label='Sensor System', alpha=0.9, edgecolors='white', linewidth=2)
            
            # Plot true path if available
            if len(self.true_positions) > 1:
                true_array = np.array(self.true_positions)
                
                # Plot trajectory
                self.ax.plot(true_array[:, 0], true_array[:, 1], true_array[:, 2], 
                            'gray', linewidth=1.5, label='Drone Path', alpha=0.5, linestyle='--')
                
                # Plot current drone position with color based on range
                if len(true_array) > 0:
                    last_pos = true_array[-1]
                    
                    # Get status based on distance
                    status, color, size = self.get_detection_status(self.last_distance)
                    
                    # Plot drone
                    self.ax.scatter(last_pos[0], last_pos[1], last_pos[2], 
                                  c=color, s=size, marker='o', 
                                  label=f'Drone ({status})', alpha=0.9,
                                  edgecolors='black', linewidth=2)
            
            # Plot estimated path and position
            if len(self.estimated_positions) > 1:
                est_array = np.array(self.estimated_positions)
                
                # Plot estimate trajectory
                self.ax.plot(est_array[:, 0], est_array[:, 1], est_array[:, 2], 
                            'blue', linewidth=2, label='Filter Estimate', alpha=0.7)
                
                # Plot current estimate
                if len(est_array) > 0:
                    last_est = est_array[-1]
                    self.ax.scatter(last_est[0], last_est[1], last_est[2], 
                                  c='cyan', s=80, marker='s', 
                                  label='Current Estimate', alpha=0.8,
                                  edgecolors='black', linewidth=1)
            
            # Plot particles (if any)
            if self.particles is not None and len(self.particles) > 0:
                # Sample particles for performance
                n_particles = min(300, len(self.particles))
                indices = np.random.choice(len(self.particles), n_particles, replace=False)
                sampled_particles = self.particles[indices]
                
                # Color particles based on detection status
                if self.pp_detected:
                    particle_color = 'red'  # Both sensors active
                elif self.doa_detected:
                    particle_color = 'orange'  # Only DOA active
                else:
                    particle_color = 'gray'  # No sensors active
                
                self.ax.scatter(sampled_particles[:, 0], sampled_particles[:, 1], 
                              sampled_particles[:, 2], c=particle_color, 
                              alpha=0.3, s=10, label='Particles')
            
            # Add status box with text
            status_text = f"Distance: {self.last_distance:.1f}m\n"
            
            if self.last_distance > self.doa_range:
                status_text += "Status: NO DETECTION"
                status_color = "red"
            elif self.last_distance > self.pp_range:
                status_text += "Status: DOA ONLY"
                status_color = "yellow"
            else:
                status_text += "Status: BOTH SENSORS"
                status_color = "green"
            
            status_text += f"\nDOA Active: {'✓' if self.doa_detected else '✗'}"
            status_text += f"\nPP Active: {'✓' if self.pp_detected else '✗'}"
            
            # Add text box
            props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=status_color)
            self.ax.text2D(0.02, 0.98, status_text, transform=self.ax.transAxes,
                          fontsize=11, verticalalignment='top', bbox=props,
                          color='black')
            
            # Add title
            if self.last_distance > self.doa_range:
                title = "Drone: OUT OF RANGE - Waiting for detection..."
                title_color = "red"
            elif self.last_distance > self.pp_range:
                title = "Drone: DOA DETECTED - Course tracking only"
                title_color = "orange"
            else:
                title = "Drone: BOTH SENSORS ACTIVE - Precise tracking"
                title_color = "green"
            
            self.ax.set_title(title, fontsize=14, color=title_color, fontweight='bold')
            
            # Add legend
            self.ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
            
            # Set view angle
            self.ax.view_init(elev=25, azim=45)
            
            plt.draw()
            plt.pause(0.001)
            
        except Exception as e:
            self.get_logger().warning(f"Plot error: {e}")

def main():
    rclpy.init()
    node = SensorRangeVisualization()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n" + "="*60)
        print("Visualization stopped")
        print("="*60)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        plt.close('all')
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()