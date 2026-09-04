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
        parameters=[{'device1':'/dev/video4',
                    'camera1':'brick2',
                    'pixel_format1':'UYVY',
                    'width1':256,
                    'height1':192,
                    'stereo':True,
                    'device2':'/dev/video6',
                    'camera2':'nv',
                    'pixel_format2':'UYVY',
                    'width2':512,
                    'height2':192,                  
                    'publish_rate':30.0}],
        output="screen"
    )

    rgb_node = Node(
        package='rgb_camera',
        executable='rgb_camera_node',
        name='rgb',
        parameters=[{'device1':'/dev/video0',
                    'device2':'/dev/video2',
                    'stereo':True,
                    'pixel_format1':'MJPG',
                    'pixel_format2':'MJPG',
                    'width':1920,
                    'height':1080,
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
            rgb_node,
            #rviz_display_node
        ]
    )
