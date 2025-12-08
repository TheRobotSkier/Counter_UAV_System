from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='cuav_serial',
            executable='speed_of_sound_server',
            output='screen'
        ),
        Node(
            package='cuav_acoustic',
            executable='doa_logging_node',
            output='screen',
            parameters=[
                {'speed_of_sound_mode': 'service'},
                {'doa_topic': '/acoustic_doa'},
            ]
        )
    ])
