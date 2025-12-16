#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Pose, Vector3
from std_msgs.msg import Header, ColorRGBA, String
from std_srvs.srv import Empty
import json
from collections import deque
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

class FilterVisualizationNode(Node):
    """RViz2-based visualization node for particle filter"""
    
    def __init__(self):
        super().__init__('filter_visualization_node')
        
        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('history_length', 50),
                ('max_particles_to_show', 2000),
                ('particle_alpha', 0.3),
                ('trajectory_linewidth', 0.05),
                ('particle_size', 0.1),
                ('true_color_r', 0.0),
                ('true_color_g', 1.0),
                ('true_color_b', 0.0),
                ('estimate_color_r', 0.0),
                ('estimate_color_g', 0.0),
                ('estimate_color_b', 1.0),
                ('particle_color_r', 1.0),
                ('particle_color_g', 0.0),
                ('particle_color_b', 0.0),
                ('frame_id', 'map')
            ]
        )
        
        # Get parameters
        self.history_length = self.get_parameter('history_length').value
        self.max_particles = self.get_parameter('max_particles_to_show').value
        self.particle_alpha = self.get_parameter('particle_alpha').value
        self.trajectory_linewidth = self.get_parameter('trajectory_linewidth').value
        self.particle_size = self.get_parameter('particle_size').value
        self.frame_id = self.get_parameter('frame_id').value
        
        # Colors
        self.true_color = ColorRGBA(
            r=self.get_parameter('true_color_r').value,
            g=self.get_parameter('true_color_g').value,
            b=self.get_parameter('true_color_b').value,
            a=1.0
        )
        
        self.estimate_color = ColorRGBA(
            r=self.get_parameter('estimate_color_r').value,
            g=self.get_parameter('estimate_color_g').value,
            b=self.get_parameter('estimate_color_b').value,
            a=1.0
        )
        
        self.particle_color = ColorRGBA(
            r=self.get_parameter('particle_color_r').value,
            g=self.get_parameter('particle_color_g').value,
            b=self.get_parameter('particle_color_b').value,
            a=self.particle_alpha
        )
        
        # Data storage
        self.true_positions = deque(maxlen=self.history_length)
        self.estimated_positions = deque(maxlen=self.history_length)
        self.latest_particles = None
        
        # RViz2 markers publisher with compatible QoS
        # RViz2 typically uses RELIABLE and VOLATILE for MarkerArray topics
        marker_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST
        )
        
        self.marker_pub = self.create_publisher(MarkerArray, '/visualization_markers', marker_qos)
        
        # Subscribers with BEST_EFFORT for sensor data (common practice)
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )
        
        self.true_state_sub = self.create_subscription(
            String,
            '/drone/true_state',
            self.true_state_callback,
            sensor_qos
        )
        
        self.filter_state_sub = self.create_subscription(
            String,
            '/filter/state',
            self.filter_state_callback,
            sensor_qos
        )
        
        self.particles_sub = self.create_subscription(
            String,
            '/filter/particles',
            self.particles_callback,
            sensor_qos
        )
        
        # Service to clear markers
        self.clear_srv = self.create_service(Empty, 'clear_visualization', self.clear_callback)
        
        # Visualization timer - publish at 10 Hz
        self.viz_timer = self.create_timer(0.1, self.publish_markers)
        
        self.get_logger().info(f"RViz2 visualization node started (frame: {self.frame_id})")
        self.get_logger().info("Add a 'MarkerArray' display in RViz2 and set topic to '/visualization_markers'")

    def clear_callback(self, request, response):
        """Clear all visualization markers"""
        self.true_positions.clear()
        self.estimated_positions.clear()
        self.latest_particles = None
        
        # Publish empty marker array to clear
        marker_array = MarkerArray()
        delete_marker = Marker()
        delete_marker.header.frame_id = self.frame_id
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        self.marker_pub.publish(marker_array)
        
        self.get_logger().info("Visualization cleared")
        return response

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
        except (json.JSONDecodeError, KeyError) as e:
            self.get_logger().warning(f"Failed to parse filter state: {e}")

    def particles_callback(self, msg):
        """Store real particles from particle filter"""
        try:
            data = json.loads(msg.data)
            if 'particles' in data and data['particles']:
                particles = np.array(data['particles'])
                if particles.size > 0 and np.all(np.isfinite(particles)):
                    # Sample particles if there are too many
                    if len(particles) > self.max_particles:
                        indices = np.random.choice(len(particles), 
                                                 self.max_particles, 
                                                 replace=False)
                        self.latest_particles = particles[indices]
                    else:
                        self.latest_particles = particles
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.get_logger().warning(f"Failed to parse particles: {e}")

    def create_marker(self, marker_type, ns, id, pose=None, points=None, 
                     scale=None, color=None):
        """Create a marker with common properties"""
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = ns
        marker.id = id
        marker.type = marker_type
        marker.action = Marker.ADD
        
        if pose:
            marker.pose = pose
        else:
            marker.pose.orientation.w = 1.0
            
        if scale:
            marker.scale = scale
            
        if color:
            marker.color = color
            
        if points:
            marker.points = points
            
        # Set lifetime to auto-delete old markers (2 seconds)
        marker.lifetime.sec = 2
            
        return marker

    def publish_markers(self):
        """Publish all markers to RViz2"""
        marker_array = MarkerArray()
        marker_id = 0
        
        # Publish true trajectory (LINE_STRIP)
        if len(self.true_positions) > 1:
            points = []
            for pos in self.true_positions:
                point = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
                points.append(point)
            
            marker = self.create_marker(
                marker_type=Marker.LINE_STRIP,
                ns='true_trajectory',
                id=marker_id,
                points=points,
                scale=Vector3(x=self.trajectory_linewidth),
                color=self.true_color
            )
            marker_array.markers.append(marker)
            marker_id += 1
        
        # Publish estimated trajectory (LINE_STRIP)
        if len(self.estimated_positions) > 1:
            points = []
            for pos in self.estimated_positions:
                point = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
                points.append(point)
            
            marker = self.create_marker(
                marker_type=Marker.LINE_STRIP,
                ns='estimated_trajectory',
                id=marker_id,
                points=points,
                scale=Vector3(x=self.trajectory_linewidth),
                color=self.estimate_color
            )
            marker_array.markers.append(marker)
            marker_id += 1
        
        # Publish particles (POINTS)
        if self.latest_particles is not None:
            points = []
            for particle in self.latest_particles:
                point = Point(x=float(particle[0]), y=float(particle[1]), z=float(particle[2]))
                points.append(point)
            
            marker = self.create_marker(
                marker_type=Marker.POINTS,
                ns='particles',
                id=marker_id,
                points=points,
                scale=Vector3(x=self.particle_size, y=self.particle_size),
                color=self.particle_color
            )
            marker_array.markers.append(marker)
            marker_id += 1
        
        # Publish current true position (SPHERE)
        if len(self.true_positions) > 0:
            latest_true = self.true_positions[-1]
            pose = Pose()
            pose.position = Point(x=float(latest_true[0]), 
                                 y=float(latest_true[1]), 
                                 z=float(latest_true[2]))
            pose.orientation.w = 1.0
            
            marker = self.create_marker(
                marker_type=Marker.SPHERE,
                ns='true_position',
                id=marker_id,
                pose=pose,
                scale=Vector3(x=0.3, y=0.3, z=0.3),
                color=self.true_color
            )
            marker_array.markers.append(marker)
            marker_id += 1
        
        # Publish current estimate (CUBE)
        if len(self.estimated_positions) > 0:
            latest_est = self.estimated_positions[-1]
            pose = Pose()
            pose.position = Point(x=float(latest_est[0]), 
                                 y=float(latest_est[1]), 
                                 z=float(latest_est[2]))
            pose.orientation.w = 1.0
            
            marker = self.create_marker(
                marker_type=Marker.CUBE,
                ns='estimated_position',
                id=marker_id,
                pose=pose,
                scale=Vector3(x=0.3, y=0.3, z=0.3),
                color=self.estimate_color
            )
            marker_array.markers.append(marker)
            marker_id += 1
        
        # Publish all markers if we have any
        if marker_array.markers:
            try:
                self.marker_pub.publish(marker_array)
            except Exception as e:
                self.get_logger().warning(f"Failed to publish markers: {e}")

def main():
    rclpy.init()
    node = FilterVisualizationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Visualization node shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()