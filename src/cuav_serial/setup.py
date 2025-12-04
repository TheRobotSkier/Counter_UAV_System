from setuptools import setup

package_name = 'cuav_serial'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ehb',
    maintainer_email='121627019+TheRobotSkier@users.noreply.github.com',
    description='Serial interface and SpeedOfSound service node for counter-UAV system.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # ros2 run cuav_serial speed_of_sound_server
            'speed_of_sound_server = cuav_serial.speed_of_sound_server_node:main',
        ],
    },
)
