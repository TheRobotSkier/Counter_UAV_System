import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable
from launch_ros.actions import Node

def generate_launch_description():
    
    # --- File Paths ---
    pkg_dir = get_package_share_directory('drone_detector_sim')
    world_file = os.path.join(pkg_dir, 'worlds', 'detector_world.sdf')
    rviz_config_file = os.path.join(pkg_dir, 'rviz', 'config.rviz')
    
    # --- Environment Variable ---
    # !! IMPORTANT !!
    # Use os.path.expanduser to handle the '~' correctly.
    rgl_patterns_dir = os.path.expanduser('~/simulations/RGLGazeboPlugin-0.2.0-fortress/lidar_patterns')
    
    set_env_var = SetEnvironmentVariable(
        name='RGL_PATTERNS_DIR',
        value=rgl_patterns_dir
    )

    # --- Gazebo ---
    start_gazebo = ExecuteProcess(
        cmd=['ign', 'gazebo', '-r', world_file],
        output='screen'
    )

    # --- Bridges ---
    
    # Lidar Bridge (GZ -> ROS)
    lidar_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/lidar/avia@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked'],
        parameters=[{'use_sim_time': True}],
        output='screen'
    )
    
    # Pose Bridge (GZ -> ROS)
    pose_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[f'/world/drone_world/pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V'],
        parameters=[{'use_sim_time': True}],
        output='screen'
    )
    
    # --- FIX: Drone Control Bridge (ROS -> GZ) ---
    # We will use a unique ROS topic to avoid ambiguity.
    # ROS Topic: /drone/cmd_vel
    # Gazebo Topic: /model/target_drone/cmd_vel (This is the default for the plugin)
    control_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/drone/cmd_vel@geometry_msgs/msg/Twist[gz.msgs.Twist'],
        parameters=[{'use_sim_time': True}],
        remappings=[
            ('/drone/cmd_vel', '/model/target_drone/cmd_vel')
        ],
        output='screen'
    )

    # --- Nodes ---
    
    # Ground Truth TF Node
    ground_truth_tf_node = Node(
        package='drone_detector_sim',
        executable='ground_truth_tf_node',
        parameters=[{'use_sim_time': True}],
        output='screen'
    )
    
    # Ground Truth BBox Node
    ground_truth_bbox_node = Node(
        package='drone_detector_sim',
        executable='ground_truth_bbox_node',
        parameters=[{'use_sim_time': True}],
        output='screen'
    )
    
    # Drone Controller Node
    drone_controller_node = Node(
        package='drone_detector_sim',
        executable='drone_controller_node',
        parameters=[{'use_sim_time': True}],
        output='screen'
    )
    
    # RViz
    start_rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_file],
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    # --- Launch Description ---
    return LaunchDescription([
        set_env_var,
        start_gazebo,
        lidar_bridge,
        pose_bridge,
        control_bridge,
        ground_truth_tf_node,
        ground_truth_bbox_node,
        drone_controller_node,
        start_rviz
    ])