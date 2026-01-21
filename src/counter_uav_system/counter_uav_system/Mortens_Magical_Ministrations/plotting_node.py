#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import json
import matplotlib.pyplot as plt
from collections import deque
import time
from std_msgs.msg import String
import threading
from datetime import datetime
import os
import sys
import select

class SimpleTrajectoryPlotter(Node):
    """Simple trajectory plotter that starts on run and stops with 'q' key"""
    
    def __init__(self):
        super().__init__('simple_trajectory_plotter')
        
        # Parameters
        self.max_history = 5000
        self.save_plots = True
        self.plots_dir = 'trajectory_plots'
        self.error_threshold = 10.0  # meters
        
        # Create plots directory if needed
        if self.save_plots:
            os.makedirs(self.plots_dir, exist_ok=True)
        
        # Data storage
        self.true_positions = deque(maxlen=self.max_history)
        self.estimated_positions = deque(maxlen=self.max_history)
        self.timestamps = deque(maxlen=self.max_history)
        self.errors = deque(maxlen=self.max_history)
        
        # Per-axis error storage
        self.x_errors = deque(maxlen=self.max_history)
        self.y_errors = deque(maxlen=self.max_history)
        self.z_errors = deque(maxlen=self.max_history)
        
        # Control flags
        self.running = True
        self.plotting_active = True
        
        # Statistics
        self.start_time = time.time()
        self.data_count = 0
        
        # Subscribers
        self.true_state_sub = self.create_subscription(
            String, '/drone/true_state', self.true_state_callback, 10)
        
        self.filter_state_sub = self.create_subscription(
            String, '/filter/state', self.filter_state_callback, 10)
        
        # Setup keyboard monitoring thread
        self.keyboard_thread = threading.Thread(target=self.monitor_keyboard)
        self.keyboard_thread.daemon = True
        self.keyboard_thread.start()
        
        # Initialize plot
        self.init_plot()
        
        # Timer for updating plot
        self.plot_timer = self.create_timer(0.5, self.update_plot)
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("SIMPLE TRAJECTORY PLOTTER STARTED")
        self.get_logger().info("Press 'q' in terminal to stop recording")
        self.get_logger().info("Plot will remain open for inspection")
        self.get_logger().info("=" * 60)

    def init_plot(self):
        """Initialize the matplotlib figure"""
        plt.ion()  # Interactive mode
        self.fig, self.axes = plt.subplots(2, 3, figsize=(16, 10))
        self.fig.suptitle('Drone Tracking - Live Plot (Press q to stop)', fontsize=14)
        
        # XY Trajectory Plot
        self.ax_xy = self.axes[0, 0]
        self.ax_xy.set_title('XY Trajectory')
        self.ax_xy.set_xlabel('X (m)')
        self.ax_xy.set_ylabel('Y (m)')
        self.ax_xy.grid(True, alpha=0.3)
        self.true_line_xy, = self.ax_xy.plot([], [], 'g-', label='True', linewidth=2, alpha=0.8)
        self.est_line_xy, = self.ax_xy.plot([], [], 'b--', label='Estimated', linewidth=2, alpha=0.8)
        self.ax_xy.legend()
        
        # Error vs Time Plot
        self.ax_error = self.axes[0, 1]
        self.ax_error.set_title('Position Error vs Time')
        self.ax_error.set_xlabel('Time (s)')
        self.ax_error.set_ylabel('Error (m)')
        self.ax_error.grid(True, alpha=0.3)
        self.error_line, = self.ax_error.plot([], [], 'r-', label='Total Error', linewidth=1.5)
        self.ax_error.legend()
        
        # Per-Axis Error Scatter Plot (like your screenshot)
        self.ax_scatter = self.axes[0, 2]
        self.ax_scatter.set_title('Error Distribution')
        self.ax_scatter.set_xlabel('X Error (m)')
        self.ax_scatter.set_ylabel('Y Error (m)')
        self.ax_scatter.grid(True, alpha=0.3)
        self.ax_scatter.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
        self.ax_scatter.axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
        
        # Scatter plot for X,Y errors
        self.scatter_plot = self.ax_scatter.scatter([], [], s=20, alpha=0.5, c='blue')
        
        # Add error ellipses if we have enough data
        self.error_ellipse = None
        
        # Per-Axis Error Histograms (bottom row)
        self.ax_x_hist = self.axes[1, 0]
        self.ax_x_hist.set_title('X Error Distribution')
        self.ax_x_hist.set_xlabel('X Error (m)')
        self.ax_x_hist.set_ylabel('Frequency')
        self.ax_x_hist.grid(True, alpha=0.3)
        
        self.ax_y_hist = self.axes[1, 1]
        self.ax_y_hist.set_title('Y Error Distribution')
        self.ax_y_hist.set_xlabel('Y Error (m)')
        self.ax_y_hist.set_ylabel('Frequency')
        self.ax_y_hist.grid(True, alpha=0.3)
        
        self.ax_z_hist = self.axes[1, 2]
        self.ax_z_hist.set_title('Z Error Distribution')
        self.ax_z_hist.set_xlabel('Z Error (m)')
        self.ax_z_hist.set_ylabel('Frequency')
        self.ax_z_hist.grid(True, alpha=0.3)
        
        # Initialize histogram bars
        self.x_hist_bars = None
        self.y_hist_bars = None
        self.z_hist_bars = None
        
        plt.tight_layout()
        plt.draw()
        
        # Connect close event
        self.fig.canvas.mpl_connect('close_event', self.on_close)

    def true_state_callback(self, msg):
        """Store true drone state"""
        if not self.plotting_active:
            return
            
        try:
            data = json.loads(msg.data)
            position = np.array([
                data['true_position']['x'],
                data['true_position']['y'],
                data['true_position']['z']
            ])
            
            current_time = time.time() - self.start_time
            
            self.true_positions.append(position)
            self.timestamps.append(current_time)
            self.data_count += 1
            
        except (json.JSONDecodeError, KeyError) as e:
            self.get_logger().warn(f"Failed to parse true state: {e}")

    def filter_state_callback(self, msg):
        """Store filter estimates and calculate per-axis errors"""
        if not self.plotting_active:
            return
            
        try:
            data = json.loads(msg.data)
            position = np.array([
                data['estimated_position']['x'],
                data['estimated_position']['y'],
                data['estimated_position']['z']
            ])
            
            self.estimated_positions.append(position)
            
            # Calculate errors if we have corresponding true position
            if len(self.true_positions) == len(self.estimated_positions):
                current_true = self.true_positions[-1]
                
                # Calculate per-axis errors
                x_error = position[0] - current_true[0]
                y_error = position[1] - current_true[1]
                z_error = position[2] - current_true[2]
                
                # Store per-axis errors
                self.x_errors.append(x_error)
                self.y_errors.append(y_error)
                self.z_errors.append(z_error)
                
                # Calculate total error
                total_error = np.linalg.norm(position - current_true)
                self.errors.append(total_error)
                
        except (json.JSONDecodeError, KeyError) as e:
            self.get_logger().warn(f"Failed to parse filter state: {e}")

    def monitor_keyboard(self):
        """Monitor keyboard input for 'q' key press"""
        self.get_logger().info("Keyboard monitor started. Press 'q' then Enter to stop recording.")
        
        while self.running:
            try:
                # Check if 'q' is pressed (non-blocking)
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.read(1)
                    if key == 'q' or key == 'Q':
                        self.stop_recording()
                        break
                time.sleep(0.1)
            except Exception as e:
                self.get_logger().error(f"Keyboard monitor error: {e}")
                break

    def stop_recording(self):
        """Stop recording data"""
        self.plotting_active = False
        self.get_logger().info("=" * 60)
        self.get_logger().info("RECORDING STOPPED BY USER")
        self.get_logger().info("Plot will remain open for inspection")
        self.get_logger().info("=" * 60)
        
        # Save the data
        if self.save_plots:
            self.save_data()
        
        # Update plot title to indicate recording stopped
        self.fig.suptitle('Drone Tracking - RECORDING STOPPED (Plot remains open)', 
                         fontsize=14, color='red')
        plt.draw()

    def update_plot(self):
        """Update the plot with new data"""
        if not self.plotting_active or self.data_count == 0:
            return
        
        try:
            # Convert deques to numpy arrays
            true_array = np.array(self.true_positions)
            est_array = np.array(self.estimated_positions)
            time_array = np.array(self.timestamps)
            error_array = np.array(self.errors)
            
            # Update XY plot
            if len(true_array) > 0:
                self.true_line_xy.set_data(true_array[:, 0], true_array[:, 1])
                
            if len(est_array) > 0:
                self.est_line_xy.set_data(est_array[:, 0], est_array[:, 1])
            
            # Update error plot
            if len(error_array) > 0 and len(time_array) >= len(error_array):
                self.error_line.set_data(time_array[-len(error_array):], error_array)
            
            # Update scatter plot with X,Y errors (like your screenshot)
            if len(self.x_errors) > 0 and len(self.y_errors) > 0:
                x_err_array = np.array(self.x_errors)
                y_err_array = np.array(self.y_errors)
                
                # Update scatter plot
                self.scatter_plot.set_offsets(np.column_stack([x_err_array, y_err_array]))
                
                # Update scatter plot colors based on recency
                colors = plt.cm.viridis(np.linspace(0.3, 1, len(x_err_array)))
                self.scatter_plot.set_color(colors)
                
                # Update scatter plot limits
                if len(x_err_array) > 1:
                    x_range = max(np.abs(x_err_array).max(), 0.1) * 1.2
                    y_range = max(np.abs(y_err_array).max(), 0.1) * 1.2
                    max_range = max(x_range, y_range)
                    
                    self.ax_scatter.set_xlim(-max_range, max_range)
                    self.ax_scatter.set_ylim(-max_range, max_range)
                    
                    # Add error ellipse if we have enough data
                    if len(x_err_array) > 10:
                        self.update_error_ellipse(x_err_array, y_err_array)
            
            # Update histograms
            self.update_histograms()
            
            # Adjust plot limits
            if len(true_array) > 1 or len(est_array) > 1:
                self.adjust_plot_limits(true_array, est_array, time_array)
            
            # Redraw the plot
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            self.get_logger().error(f"Error updating plot: {e}")

    def update_error_ellipse(self, x_errors, y_errors):
        """Update error ellipse on scatter plot"""
        from matplotlib.patches import Ellipse
        import matplotlib.transforms as transforms
        
        # Calculate covariance matrix
        cov = np.cov(x_errors, y_errors)
        
        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Get the largest eigenvalue and corresponding eigenvector
        max_eigenvalue = np.max(eigenvalues)
        max_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]
        
        # Calculate angle of ellipse
        angle = np.degrees(np.arctan2(max_eigenvector[1], max_eigenvector[0]))
        
        # Calculate 95% confidence ellipse (2 sigma)
        width = 2 * np.sqrt(5.991 * eigenvalues[0])  # 95% CI for 2D
        height = 2 * np.sqrt(5.991 * eigenvalues[1])
        
        # Remove old ellipse if it exists
        if self.error_ellipse:
            self.error_ellipse.remove()
        
        # Create new ellipse
        self.error_ellipse = Ellipse(
            xy=(np.mean(x_errors), np.mean(y_errors)),
            width=width, height=height,
            angle=angle, edgecolor='red', facecolor='none',
            linestyle='--', linewidth=1.5, alpha=0.7,
            label='95% Confidence'
        )
        
        self.ax_scatter.add_patch(self.error_ellipse)
        
        # Update legend
        handles, labels = self.ax_scatter.get_legend_handles_labels()
        if '95% Confidence' not in labels:
            self.ax_scatter.legend(handles=[self.scatter_plot, self.error_ellipse], 
                                 labels=['Errors', '95% Confidence'])

    def update_histograms(self):
        """Update per-axis error histograms"""
        # X error histogram
        if len(self.x_errors) > 10:
            x_err_array = np.array(self.x_errors)
            self.ax_x_hist.clear()
            self.ax_x_hist.set_title(f'X Error Distribution (μ={np.mean(x_err_array):.3f}, σ={np.std(x_err_array):.3f})')
            self.ax_x_hist.set_xlabel('X Error (m)')
            self.ax_x_hist.set_ylabel('Frequency')
            self.ax_x_hist.grid(True, alpha=0.3)
            
            # Create histogram
            bins = min(20, len(x_err_array) // 5)
            bins = max(5, bins)
            
            n, bins, patches = self.ax_x_hist.hist(x_err_array, bins=bins, 
                                                  alpha=0.7, color='skyblue', 
                                                  edgecolor='black')
            
            # Add vertical line at mean
            self.ax_x_hist.axvline(x=np.mean(x_err_array), color='red', 
                                  linestyle='--', linewidth=2, 
                                  label=f'Mean: {np.mean(x_err_array):.3f}m')
            self.ax_x_hist.legend()
        
        # Y error histogram
        if len(self.y_errors) > 10:
            y_err_array = np.array(self.y_errors)
            self.ax_y_hist.clear()
            self.ax_y_hist.set_title(f'Y Error Distribution (μ={np.mean(y_err_array):.3f}, σ={np.std(y_err_array):.3f})')
            self.ax_y_hist.set_xlabel('Y Error (m)')
            self.ax_y_hist.set_ylabel('Frequency')
            self.ax_y_hist.grid(True, alpha=0.3)
            
            bins = min(20, len(y_err_array) // 5)
            bins = max(5, bins)
            
            n, bins, patches = self.ax_y_hist.hist(y_err_array, bins=bins, 
                                                  alpha=0.7, color='lightgreen', 
                                                  edgecolor='black')
            
            self.ax_y_hist.axvline(x=np.mean(y_err_array), color='red', 
                                  linestyle='--', linewidth=2, 
                                  label=f'Mean: {np.mean(y_err_array):.3f}m')
            self.ax_y_hist.legend()
        
        # Z error histogram
        if len(self.z_errors) > 10:
            z_err_array = np.array(self.z_errors)
            self.ax_z_hist.clear()
            self.ax_z_hist.set_title(f'Z Error Distribution (μ={np.mean(z_err_array):.3f}, σ={np.std(z_err_array):.3f})')
            self.ax_z_hist.set_xlabel('Z Error (m)')
            self.ax_z_hist.set_ylabel('Frequency')
            self.ax_z_hist.grid(True, alpha=0.3)
            
            bins = min(20, len(z_err_array) // 5)
            bins = max(5, bins)
            
            n, bins, patches = self.ax_z_hist.hist(z_err_array, bins=bins, 
                                                  alpha=0.7, color='lightcoral', 
                                                  edgecolor='black')
            
            self.ax_z_hist.axvline(x=np.mean(z_err_array), color='red', 
                                  linestyle='--', linewidth=2, 
                                  label=f'Mean: {np.mean(z_err_array):.3f}m')
            self.ax_z_hist.legend()

    def adjust_plot_limits(self, true_array, est_array, time_array):
        """Adjust plot limits based on data"""
        # XY plot limits
        if len(true_array) > 0 or len(est_array) > 0:
            all_xy = []
            if len(true_array) > 0:
                all_xy.append(true_array[:, :2])
            if len(est_array) > 0:
                all_xy.append(est_array[:, :2])
            
            if all_xy:
                all_xy = np.vstack(all_xy)
                min_xy = np.min(all_xy, axis=0)
                max_xy = np.max(all_xy, axis=0)
                center = (min_xy + max_xy) / 2
                range_xy = max_xy - min_xy
                
                # Add 20% padding
                padding = np.max(range_xy) * 0.2
                self.ax_xy.set_xlim(center[0] - padding, center[0] + padding)
                self.ax_xy.set_ylim(center[1] - padding, center[1] + padding)
        
        # Error plot limits
        if len(self.errors) > 0 and len(time_array) >= len(self.errors):
            error_times = time_array[-len(self.errors):]
            self.ax_error.set_xlim(0, max(error_times) * 1.05)
            
            errors = np.array(self.errors)
            max_error = np.max(errors) if len(errors) > 0 else 1.0
            self.ax_error.set_ylim(0, max_error * 1.2)

    def save_data(self):
        """Save collected data to file with statistics like your table"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trajectory_data_{timestamp}"
            
            # Calculate statistics
            stats = self.calculate_statistics()
            
            # Save CSV file with errors
            csv_path = os.path.join(self.plots_dir, f"{filename}.csv")
            with open(csv_path, 'w') as f:
                # Write header
                f.write("timestamp,x_true,y_true,z_true,x_est,y_est,z_est,"
                       "x_error,y_error,z_error,total_error\n")
                
                # Align data by timestamp
                min_len = min(len(self.timestamps), len(self.true_positions), 
                            len(self.estimated_positions), len(self.x_errors))
                
                for i in range(min_len):
                    t = self.timestamps[i]
                    true_pos = self.true_positions[i]
                    est_pos = self.estimated_positions[i] if i < len(self.estimated_positions) else np.zeros(3)
                    x_err = self.x_errors[i] if i < len(self.x_errors) else 0
                    y_err = self.y_errors[i] if i < len(self.y_errors) else 0
                    z_err = self.z_errors[i] if i < len(self.z_errors) else 0
                    total_err = self.errors[i] if i < len(self.errors) else 0
                    
                    f.write(f"{t:.3f},"
                           f"{true_pos[0]:.3f},{true_pos[1]:.3f},{true_pos[2]:.3f},"
                           f"{est_pos[0]:.3f},{est_pos[1]:.3f},{est_pos[2]:.3f},"
                           f"{x_err:.3f},{y_err:.3f},{z_err:.3f},{total_err:.3f}\n")
            
            # Save statistics to separate file
            stats_path = os.path.join(self.plots_dir, f"{filename}_stats.txt")
            with open(stats_path, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("ERROR STATISTICS SUMMARY\n")
                f.write("=" * 60 + "\n\n")
                
                f.write("Mean Errors:\n")
                f.write(f"  X: {stats['mean_x']:.3f} m\n")
                f.write(f"  Y: {stats['mean_y']:.3f} m\n")
                f.write(f"  Z: {stats['mean_z']:.3f} m\n\n")
                
                f.write("Standard Deviations:\n")
                f.write(f"  X: {stats['std_x']:.3f} m\n")
                f.write(f"  Y: {stats['std_y']:.3f} m\n")
                f.write(f"  Z: {stats['std_z']:.3f} m\n\n")
                
                f.write("Maximum Errors:\n")
                f.write(f"  X: {stats['max_x']:.3f} m\n")
                f.write(f"  Y: {stats['max_y']:.3f} m\n")
                f.write(f"  Z: {stats['max_z']:.3f} m\n")
                f.write(f"  Total: {stats['max_total']:.3f} m\n\n")
                
                f.write("Other Statistics:\n")
                f.write(f"  RMSE: {stats['rmse']:.3f} m\n")
                f.write(f"  Mean Distance: {stats['mean_dist']:.3f} m\n")
                f.write(f"  Data Points: {stats['num_points']}\n")
                f.write(f"  Recording Time: {stats['recording_time']:.1f} s\n")
            
            # Save plot image
            plot_path = os.path.join(self.plots_dir, f"{filename}.png")
            self.fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            
            self.get_logger().info(f"Data saved to:\n  {csv_path}\n  {stats_path}\n  {plot_path}")
            
            # Print statistics summary
            self.print_statistics_summary(stats)
            
        except Exception as e:
            self.get_logger().error(f"Failed to save data: {e}")

    def calculate_statistics(self):
        """Calculate error statistics like in your table"""
        if len(self.x_errors) == 0:
            return {}
        
        x_err_array = np.array(self.x_errors)
        y_err_array = np.array(self.y_errors)
        z_err_array = np.array(self.z_errors)
        total_err_array = np.array(self.errors)
        
        stats = {
            'mean_x': float(np.mean(x_err_array)),
            'mean_y': float(np.mean(y_err_array)),
            'mean_z': float(np.mean(z_err_array)),
            'std_x': float(np.std(x_err_array)),
            'std_y': float(np.std(y_err_array)),
            'std_z': float(np.std(z_err_array)),
            'max_x': float(np.max(np.abs(x_err_array))),
            'max_y': float(np.max(np.abs(y_err_array))),
            'max_z': float(np.max(np.abs(z_err_array))),
            'max_total': float(np.max(total_err_array)),
            'rmse': float(np.sqrt(np.mean(total_err_array**2))),
            'mean_dist': float(np.mean(total_err_array)),
            'num_points': len(x_err_array),
            'recording_time': float(self.timestamps[-1]) if len(self.timestamps) > 0 else 0
        }
        
        return stats

    def print_statistics_summary(self, stats):
        """Print statistics in a table format similar to your screenshot"""
        self.get_logger().info("=" * 60)
        self.get_logger().info("ERROR STATISTICS SUMMARY")
        self.get_logger().info("=" * 60)
        
        # Create table header
        header = f"{'Metric':<15} {'X':<10} {'Y':<10} {'Z':<10} {'Total':<10}"
        self.get_logger().info(header)
        self.get_logger().info("-" * 60)
        
        # Mean errors
        self.get_logger().info(f"{'Mean Error (m)':<15} "
                              f"{stats['mean_x']:<10.3f} "
                              f"{stats['mean_y']:<10.3f} "
                              f"{stats['mean_z']:<10.3f} "
                              f"{stats['mean_dist']:<10.3f}")
        
        # Standard deviations
        self.get_logger().info(f"{'STD (m)':<15} "
                              f"{stats['std_x']:<10.3f} "
                              f"{stats['std_y']:<10.3f} "
                              f"{stats['std_z']:<10.3f} "
                              f"{'-':<10}")
        
        # Maximum errors
        self.get_logger().info(f"{'Max Error (m)':<15} "
                              f"{stats['max_x']:<10.3f} "
                              f"{stats['max_y']:<10.3f} "
                              f"{stats['max_z']:<10.3f} "
                              f"{stats['max_total']:<10.3f}")
        
        # Additional statistics
        self.get_logger().info(f"{'RMSE (m)':<15} {'-':<10} {'-':<10} {'-':<10} "
                              f"{stats['rmse']:<10.3f}")
        
        self.get_logger().info("=" * 60)

    def on_close(self, event):
        """Handle plot window close event"""
        self.get_logger().info("Plot window closed")
        self.running = False
        if self.plotting_active and self.save_plots:
            self.save_data()

    def destroy_node(self):
        """Cleanup on shutdown"""
        self.running = False
        if self.plotting_active and self.save_plots:
            self.save_data()
        
        # Keep the plot open
        if plt.fignum_exists(self.fig.number):
            plt.ioff()  # Turn off interactive mode
            plt.show()  # Keep plot window open
        
        super().destroy_node()

def main():
    rclpy.init()
    node = SimpleTrajectoryPlotter()
    
    try:
        # Start spinning in a separate thread
        import threading
        spin_thread = threading.Thread(target=rclpy.spin, args=(node,))
        spin_thread.daemon = True
        spin_thread.start()
        
        # Keep main thread alive for keyboard input
        while node.running:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
