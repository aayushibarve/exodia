from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    rviz_and_imu_node = Node(
        package='wit_ros2_imu',
        executable='wit_ros2_imu',
        name='imu',
        remappings=[('/wit/imu', '/imu/data')],
        parameters=[{'port': '/dev/ttyUSB0'},
                    {"baud": 115200}],
        output="screen"

    )
    
    thermal_node = Node(
        package='thermal_camera',
        executable='thermal_camera_node',
        name='thermal',
        parameters=[{'device1':'/dev/video0',
                    'camera1':'nv',
                    'pixel_format1':'UYVY',
                    'width1':512,
                    'height1':192,
                    'publish_rate':30.0}],
        output="screen"
    )

    rviz_display_node = Node(
        package='rviz2',
        executable="rviz2",
        output="screen"
    )

    return LaunchDescription(
        [
            rviz_and_imu_node,
            thermal_node,
            #rviz_display_node
        ]
    )
