#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Point
from uav_interfaces.msg import DroneState, DOAData, PointPillarsData, ParticleFilterState
from visualization_msgs.msg import Marker
import csv
from datetime import datetime
import math

# ================= RTK CSV Processing Parameters =================
CSV_FILE_PATH = "/home/mort/Counter_UAV_System/RTK_Data/test_pp8_rtk_log_in_local_frame_20251211_145341.csv"

# This is the where you find the ros bag https://aaudk.sharepoint.com/:f:/r/sites/a_P7_Mobile_Robots/Delte%20dokumenter/General/Tests/Test_Bags/park_t8?csf=1&web=1&e=ODcb37

# Time Configuration
ROSBAG_START_UNIX = 1765546524.901434980
                    
RTK_TIME_OFFSET = -3600.0  
MANUAL_TIME_SHIFT = 0.0    

# Manual Translations
MANUAL_TRANS_X = 0.0   
MANUAL_TRANS_Y = 0.0
MANUAL_TRANS_Z = 0.0

# Manual Rotations (Set to 0.0 as not specified in current config)
MANUAL_ROT_X = 0.0
MANUAL_ROT_Y = 0.0
MANUAL_ROT_Z = 0.0

# Axis Adjustments
INVERT_X = True
INVERT_Y = True
SWAP_XY = False

# Filter Configuration
MAX_ERROR_THRESHOLD = 10.0  

# Time Trimming
IGNORE_START_SECONDS = 15.0  
IGNORE_END_SECONDS = 8.0  
# =================================================

class RTKDataProcessor():
    def __init__(self, logger):
        self.logger = logger
        self.rtk_data = self.load_csv(CSV_FILE_PATH)

        self.logger.info(f"Loaded {len(self.rtk_data)} RTK points from {CSV_FILE_PATH}")
        if self.rtk_data:
            self.logger.info(f"First RTK point: {self.rtk_data[0]}")

    def load_csv(self, filepath):
        data = []
        try:
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        dt = datetime.fromisoformat(row['timestamp_utc'].replace('Z', '+00:00'))
                        unix_time = dt.timestamp() + RTK_TIME_OFFSET
                        
                        raw_x = float(row['local_x_m'])
                        raw_y = float(row['local_y_m'])
                        raw_z = float(row['local_z_m'])

                        if SWAP_XY: raw_x, raw_y = raw_y, raw_x
                        final_x = -raw_x if INVERT_X else raw_x
                        final_y = -raw_y if INVERT_Y else raw_y
                        
                        # 1. Apply Rotation
                        rot_x, rot_y, rot_z = self.apply_rotation(final_x, final_y, raw_z)

                        # 2. Apply Translation
                        final_x = rot_x + MANUAL_TRANS_X
                        final_y = rot_y + MANUAL_TRANS_Y
                        final_z = rot_z + MANUAL_TRANS_Z

                        entry = {
                            'timestamp': unix_time,
                            'x': final_x,
                            'y': final_y,
                            'z': final_z
                        }
                        data.append(entry)
                    except ValueError:
                        continue
            return data    
        except FileNotFoundError:
            self.logger.error(f"File not found: {filepath}")

    def apply_rotation(self, x, y, z):
        """Applies 3D rotation (Euler angles in degrees) to a point."""
        # Convert to radians
        roll = math.radians(MANUAL_ROT_X)
        pitch = math.radians(MANUAL_ROT_Y)
        yaw = math.radians(MANUAL_ROT_Z)

        # 1. Rotate around X (Roll)
        y1 = y * math.cos(roll) - z * math.sin(roll)
        z1 = y * math.sin(roll) + z * math.cos(roll)
        x1 = x

        # 2. Rotate around Y (Pitch)
        x2 = x1 * math.cos(pitch) + z1 * math.sin(pitch)
        z2 = -x1 * math.sin(pitch) + z1 * math.cos(pitch)
        y2 = y1

        # 3. Rotate around Z (Yaw)
        x3 = x2 * math.cos(yaw) - y2 * math.sin(yaw)
        y3 = x2 * math.sin(yaw) + y2 * math.cos(yaw)
        z3 = z2

        return x3, y3, z3

    def get_interpolated_rtk(self, query_time):
            low = 0
            high = len(self.rtk_data) - 1
            while low <= high:
                mid = (low + high) // 2
                if self.rtk_data[mid]['timestamp'] < query_time:
                    low = mid + 1
                else:
                    high = mid - 1     
            candidates = []
            if low < len(self.rtk_data): candidates.append(self.rtk_data[low])
            if low > 0: candidates.append(self.rtk_data[low - 1])
            if not candidates: return None
            closest_entry = min(candidates, key=lambda x: abs(x['timestamp'] - query_time))
            if abs(closest_entry['timestamp'] - query_time) > 0.5:
                return None
            return closest_entry

class ParticleFilter:
    def __init__(self, clock, num_particles=1000):
        self.clock = clock
        self.prevtime = None

        # Initialization parameters
        self.num_particles = num_particles
        self.particles = None
        self.velocities = None
        self.weights = None 
        self.doa_range = 50.0

        # Doa simulation parameters
        self.doa_true_std = 14 * (math.pi / 180)  # From the worksheet DOA test. Optimistic estimate of total angular std in degrees, converted to radians.

        # Prediction parameters
        self.max_speed = 30.0 
        self.velocity_decay = 0.95
        self.position_noise_std = 0.1
        self.velocity_noise_std = 0.1

        # Measurement parameters 
        self.pp_std = 1 # From the worksheet pointpillars test. About 1 meter standard deviation on all axis.
        self.doa_std = 15 * (math.pi / 180) 

    def doa_simulation(self, rtk_point):
        #  Ground truth direction vector
        v = np.array([
            rtk_point['x'],
            rtk_point['y'],
            rtk_point['z']
        ])
        # normalize
        v = v / (np.linalg.norm(v) + 1e-9)

        # angular noise (radians)
        noise = np.random.normal(0.0, self.doa_true_std, 3)
        # perturb direction and renormalize
        v_noisy = v + noise
        v_noisy /= np.linalg.norm(v_noisy) + 1e-9

        return v_noisy
         
    def initialize_from_doa(self, doa_measure):
        # Initialize particles around the doa axis
        positions = []
        velocities = []
        for _ in range(self.num_particles):
            # Sample position along the DOA axis with noise
            doa_axis_pos = np.random.uniform(0, self.doa_range, 1) # How far along the doa axis 
            pos_noise = np.random.normal(0, 10.0, 3) # noise around the doa axis
            pos = doa_measure * doa_axis_pos + pos_noise   
            positions.append(pos)

            # Initial random velocity
            vel = np.random.uniform(-1, 1, 3)
            velocities.append(vel)

        # Apply initialization
        self.particles = np.array(positions)
        self.velocities = np.array(velocities)
        self.weights = np.ones(self.num_particles) / self.num_particles
    
    def predict(self):
        now = self.clock.now().nanoseconds * 1e-9
        if self.prevtime is None:
            dt = 0.05
        else:
            dt = now - self.prevtime

        # Velocity update with noise and decay
        Acc_noise = np.random.normal(0, self.velocity_noise_std, (self.num_particles, 3))
        self.velocities += Acc_noise * dt
        self.velocities *= self.velocity_decay 
        self.velocities = np.clip(self.velocities, -self.max_speed, self.max_speed)
  
        # Position update with noise
        Pos_noise = np.random.normal(0, self.position_noise_std, (self.num_particles, 3))
        self.particles += (self.velocities * dt) + Pos_noise

        self.prevtime = now

    def update_weights(self, pp_position, doa_measure, use_pp, use_doa):
        
        # Pointpillars probabillity 
        if use_pp:
            dists = np.linalg.norm(self.particles - pp_position, axis=1)
            pp_likelihoods = np.exp(-0.5 * (dists / self.pp_std) ** 2)
        else:
            pp_likelihoods = np.ones(self.num_particles)
    
        # DOA probability
        if use_doa:
            doa_vectors = self.particles / (np.linalg.norm(self.particles, axis=1, keepdims=True) + 1e-9)
            dot_products = np.clip(np.dot(doa_vectors, doa_measure), -1.0, 1.0)
            angles = np.arccos(dot_products)
            doa_likelihoods = np.exp(-0.5 * (angles / self.doa_std) ** 2)
        else:
            doa_likelihoods = np.ones(self.num_particles)

        # Update weights based on combined likelihoods
        self.weights *= pp_likelihoods * doa_likelihoods
        self.weights += 1.e-300 
        self.weights /= np.sum(self.weights)

    def resample(self):
        # Effective sample size
        E_ff = 1.0 / np.sum(self.weights ** 2)
        if E_ff > self.num_particles / 2:
            return  # No resampling needed

        # Systematic resampling
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0
        step = 1.0 / self.num_particles
        start = np.random.uniform(0, step)
        positions = np.arange(self.num_particles) * step + start
        
        indices = np.zeros(self.num_particles, dtype=int)
        i, j = 0, 0
        while i < self.num_particles:
            if positions[i] < cumulative_sum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1
        
        self.particles = self.particles[indices]
        self.velocities = self.velocities[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate_position(self):
        if self.particles is not None:
            return np.average(self.particles, weights=self.weights, axis=0)
        return np.array([0., 0., 0.])
    
    def estimate_velocity(self):
        if self.velocities is not None:
            return np.average(self.velocities, weights=self.weights, axis=0)
        return np.array([0., 0., 0.])

class ParticleFilterNode(Node):
    def __init__(self):
        super().__init__('particle_filter_node')
        
        
        # Class initializations
        self.rtk_processor = RTKDataProcessor(self.get_logger())
        self.particle_filter = ParticleFilter(self.get_clock())

        # Simulation activation and frame parameters
        self.declare_parameter('simulate', True)
        self.declare_parameter('global_frame', 'world')
        self.global_frame = self.get_parameter('global_frame').get_parameter_value().string_value        

        # Messurement Data Holders
        self.latest_pp_data = None
        self.latest_DOA_data = None
        self.pp_Measure: bool = False
        self.doa_Measure: bool = False
        self.start_time_bag = None
        self.start_time_true = None

        # Subscribers
        self.pp_sub = self.create_subscription(
            PointPillarsData, '/pointpillars_bbox', self.pp_callback, 10)
        
        # Publishers
        self.vis_pub = self.create_publisher(Marker, '/filter/visualization_marker', 10)
        self.aiming_pub = self.create_publisher(Point, '/cmd_point', 10) 

        # DOA Measurement Timer
        self.timer_rtk = self.create_timer(0.1, self.doa_measurement)             
        
        # Particle filter update timer
        self.timer = self.create_timer(0.05, self.process_update)

        self.get_logger().info(f"Particle Filter Node started")

    def pp_callback(self, msg):
        
        # Initialization of time
        self.get_logger().info("PointPillars measurement received")
        if self.start_time_bag is None:
            self.start_time_bag = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.start_time_true = self.get_clock().now()

        self.latest_pp_data = np.array([msg.position.x, msg.position.y, msg.position.z])
        self.pp_Measure = True

    def doa_measurement(self):  

        # Skip if bag time not initialized
        if self.start_time_bag is None:
            return
            
        # Fetch current RTK ground truth based on ROS time
        time_diff = self.get_clock().now() - self.start_time_true

        current_unix_time = self.start_time_bag + time_diff.nanoseconds * 1e-9 + MANUAL_TIME_SHIFT
        rtk_point = self.rtk_processor.get_interpolated_rtk(current_unix_time)

        if rtk_point is None:
            self.get_logger().warn(
                f"RTK lookup failed at t={current_unix_time:.3f}"
            )
            return

        # Simulate DOA measurement based on RTK ground truth
        self.latest_DOA_data = self.particle_filter.doa_simulation(rtk_point) 
        self.doa_Measure = True

    def process_update(self):
        
        # Initialization of Particles 
        if self.particle_filter.particles is None:
            if self.doa_Measure:
                self.particle_filter.initialize_from_doa(self.latest_DOA_data)
                self.doa_Measure = False
            else:
                return

        # Prediction 
        self.particle_filter.predict() 

        # Measurement updates (only runs when sensor gives !!new!! data)
        if self.pp_Measure or self.doa_Measure:
            self.particle_filter.update_weights(self.latest_pp_data, self.latest_DOA_data, self.pp_Measure, self.doa_Measure)
            self.particle_filter.resample()
            self.pp_Measure = False
            self.doa_Measure = False

        estimated_position = self.particle_filter.estimate_position()
        
        # 1. Publish Aiming Command
        aiming_msg = Point()
        aiming_msg.x = float(estimated_position[0])
        aiming_msg.y = float(estimated_position[1])
        aiming_msg.z = float(estimated_position[2])
        self.aiming_pub.publish(aiming_msg)
        
        # 2. Publish Visualization
        self.publish_markers(estimated_position)

    def publish_markers(self, estimated_pos):
        marker = Marker()
        marker.header.frame_id = self.global_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "filter_estimate"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(estimated_pos[0])
        marker.pose.position.y = float(estimated_pos[1])
        marker.pose.position.z = float(estimated_pos[2])
        marker.scale.x = 0.5 
        marker.scale.y = 0.5
        marker.scale.z = 0.5
        
        # Green color
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0 
        
        self.vis_pub.publish(marker)

def main():
    rclpy.init()
    node = ParticleFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()