from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='cuav_serial',
            executable='speed_of_sound_server',
            name='speed_of_sound_server',
            parameters=[
                {'port': '/dev/ttyACM0'},
                {'baud': 115200},
            ],
            output='screen',
        ),
        Node(
            package='cuav_acoustic',
            executable='doa_node',
            name='doa_node',
            parameters=[
                {'speed_of_sound_mode': 'service'},
                {'speed_of_sound_service_name': '/get_speed_of_sound'},
                {'doa_topic': '/acoustic_doa'},
            ],
            output='screen',
        ),
    ])
