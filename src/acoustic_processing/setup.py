from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'acoustic_processing'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],  # inside acoustic_processing_pkg/
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),

        ('share/' + package_name, ['package.xml']),

        ('share/' + package_name + '/launch',
         glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ehb',
    maintainer_email='TheRobotSkier@users.noreply.github.com',
    description='Real-time acoustic DOA processing with ROS2 + JACK',
    license='MIT',
    entry_points={
        'console_scripts': [
            'uav_doa_node = acoustic_processing.uav_doa_node:main',
        ],
    },
)
