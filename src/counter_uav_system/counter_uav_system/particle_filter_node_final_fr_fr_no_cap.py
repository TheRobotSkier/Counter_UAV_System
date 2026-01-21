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
# Path for 'test_pp5'
CSV_FILE_PATH = "Counter_UAV_System-development/firmware/rtk_calibration_2/rtk_data_in_local_frame/day3/test_pp8.csv"

# Time Configuration
ROSBAG_START_UNIX = 1765461212.783921965
RTK_TIME_OFFSET = -3600.0  
MANUAL_TIME_SHIFT = 0.0    

# Manual Translations
MANUAL_TRANS_X = 0.0   
MANUAL_TRANS_Y = 0.0
MANUAL_TRANS_Z = -0.5

# Manual Rotations (Set to 0.0 as not specified in current config)
MANUAL_ROT_X = 0.0
MANUAL_ROT_Y = 0.0
MANUAL_ROT_Z = 15.0

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

class RTKDataProcessor:
    def __init__(self):
        self.rtk_data = self.load_csv(CSV_FILE_PATH)

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
        except FileNotFoundError:
            self.get_logger().error(f"File not found: {filepath}")

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
        self.num_particles = num_particles
        self.particles = None
        self.velocities = None
        self.weights = None 
        self.max_speed = 50.0 
        self.velocity_decay = 0.9
        self.position_noise_std = 0.1
        self.position_std = 1.0
        
    def initialize_particles(self, initial_positions, initial_velocities):
        self.particles = np.array(initial_positions)
        self.velocities = np.array(initial_velocities)
        self.weights = np.ones(self.num_particles) / self.num_particles
        
    def predict(self):
        now = self.clock.now().nanoseconds * 1e-9
        if self.prevtime is None:
            dt = 0.05
        else:
            dt = now - self.prevtime

        # Velocity update with noise and decay
        Acc_noise = np.random.normal(0, 0.1, (self.num_particles, 3))
        self.velocities += Acc_noise * dt
        self.velocities = self.constrain_velocity(self.velocities * self.velocity_decay)

        # Position update with noise
        Pos_noise = np.random.normal(0, 0.1, (self.num_particles, 3))
        self.particles += (self.velocities * dt) + Pos_noise

        self.prevtime = now

    def constrain_velocity(self, velocity):
        return np.clip(velocity, -self.max_speed, self.max_speed)
            
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
    
    def initialize_from_pp(self, pp_position):
        positions = []
        velocities = []
        for _ in range(self.num_particles):
            pos = pp_position + np.random.normal(0, 2.0, 3)
            vel = np.random.uniform(-1, 1, 3)
            positions.append(pos)
            velocities.append(vel)
        self.initialize_particles(positions, velocities)

    def update_weights(self, pp_position):
        dists = np.linalg.norm(self.particles - pp_position, axis=1)
        self.weights *= np.exp(-0.5 * (dists / self.position_std) ** 2)
        self.weights += 1.e-300 
        self.weights /= np.sum(self.weights)


class ParticleFilterNode(Node):
    def __init__(self):
        super().__init__('particle_filter_node')
        
        # Load RTK data
        self.rtk_processor = RTKDataProcessor()

        self.declare_parameter('global_frame', 'world')
        self.global_frame = self.get_parameter('global_frame').get_parameter_value().string_value
        
        self.particle_filter = ParticleFilter(self.get_clock())
        self.latest_pp_data = None
        self.Pp_Measure: bool = False
        self.Pp_Doa: bool = False
        
        # Subscribers
        self.pp_sub = self.create_subscription(
            PointPillarsData, '/sensors/point_pillars', self.pp_callback, 10)
        
        # Publishers
        self.vis_pub = self.create_publisher(Marker, '/filter/visualization_marker', 10)
        self.aiming_pub = self.create_publisher(Point, '/cmd_point', 10) # <--- RESTORED THIS

        # Initialization
        while True:
            if self.particle_filter.particles is None and self.Pp_Measure == True:
                    self.particle_filter.initialize_from_pp(self.latest_pp_data)
                    break
        
        self.timer = self.create_timer(0.05, self.process_update)
        self.get_logger().info(f"Particle Filter Node started in frame: {self.global_frame}")

    def pp_callback(self, msg):
        self.latest_pp_data = np.array([msg.position.x, msg.position.y, msg.position.z])
        self.Pp_Measure = True

    #def Doa_callback(self, msg):
    #    self.latest_pp_data = np.array([msg.position.x, msg.position.y, msg.position.z])
    #    self.Pp_Measure = True

    def process_update(self):

        # The ground truth from RTK CSV


        # Initialization
        #if self.particle_filter.particles is None and self.Pp_Measure == True:
        #        self.particle_filter.initialize_from_pp(self.latest_pp_data)
        #elif self.particle_filter.particles is None:
        #      return  # Wait until we have initial data to initialize particles
        
        # Prediction step
        self.particle_filter.predict() 

        # Measurement updates (only runs when sensor gives !!new!! data)
        if self.Pp_Measure:
            self.particle_filter.update_weights(self.latest_pp_data)
            self.particle_filter.resample()
            self.Pp_Measure = False

        estimated_position = self.particle_filter.estimate_position()
        
        # 1. Publish Aiming Command (RESTORED)
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