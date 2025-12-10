# launch_all_nodes.py
#!/usr/bin/env python3
"""
Launch all nodes for the particle filter simulation system.
"""

import rclpy
from rclpy.executors import MultiThreadedExecutor
import threading
import time
import sys
import os

# Import nodes (they will be imported when needed)
# This avoids import errors if some nodes are not available

def main():
    print("Starting Particle Filter Simulation System...")
    print("=" * 60)
    print("Nodes to launch:")
    print("  1. Drone Simulation Node")
    print("  2. Sensor Simulation Node")
    print("  3. Particle Filter Node")
    print("  4. Visualization Node")
    print("=" * 60)
    
    # Initialize ROS
    rclpy.init()
    
    # Create executor
    executor = MultiThreadedExecutor()
    
    nodes = []
    
    try:
        # 1. Drone Simulation Node
        from drone_sim_node import DroneSimNode
        drone_node = DroneSimNode()
        executor.add_node(drone_node)
        nodes.append(drone_node)
        print("✓ Drone simulation node started")
        
        # Give drone node time to initialize
        time.sleep(1)
        
        # 2. Sensor Simulation Node
        from sensor_sim_node import SensorSimNode
        sensor_node = SensorSimNode()
        executor.add_node(sensor_node)
        nodes.append(sensor_node)
        print("✓ Sensor simulation node started")
        
        # 3. Particle Filter Node
        from Particle_Filter import ParticleFilterNode
        pf_node = ParticleFilterNode()
        executor.add_node(pf_node)
        nodes.append(pf_node)
        print("✓ Particle filter node started")
        
        # 4. Visualization Node
        from Filter_Visualization import FilterVisualization
        viz_node = FilterVisualization()
        executor.add_node(viz_node)
        nodes.append(viz_node)
        print("✓ Visualization node started")
        
        print("\n" + "=" * 60)
        print("All nodes started successfully!")
        print("Press Ctrl+C to stop all nodes")
        print("=" * 60)
        
        # Run all nodes
        executor.spin()
        
    except ImportError as e:
        print(f"Error importing node: {e}")
        print("Make sure all node files are in the same directory")
        return 1
    except KeyboardInterrupt:
        print("\nShutting down all nodes...")
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1
    finally:
        # Shutdown all nodes
        for node in nodes:
            node.destroy_node()
        rclpy.shutdown()
        print("All nodes shut down successfully")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())