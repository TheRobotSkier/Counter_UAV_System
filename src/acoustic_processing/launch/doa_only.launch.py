from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    return LaunchDescription([
        Node(
            package='acoustic_processing',
            executable='uav_doa_node',
            name='acoustic_doa',
            output='screen',
            emulate_tty=True
        )
    ])
