#!/usr/bin/env python3

import os
import sys
import csv
import threading
import termios
import tty

import cv2
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from cv_bridge import CvBridge


class DataCaptureNode(Node):

    def __init__(self):
        super().__init__('data_capture_node')

        self.base_dir = os.path.expanduser(
            '~/data_capture/experiments'
        )

        os.makedirs(self.base_dir, exist_ok=True)

        experiment_number = 1

        while True:
            experiment_name = f'experiment_{experiment_number:03d}'
            self.output_dir = os.path.join(
                self.base_dir,
                experiment_name
            )

            if not os.path.exists(self.output_dir):
                break

            experiment_number += 1

        self.image_dir = os.path.join(
            self.output_dir,
            'images'
        )

        os.makedirs(self.image_dir, exist_ok=True)

        self.imu_file_path = os.path.join(
            self.output_dir,
            'imu.csv'
        )

        self.timestamp_file_path = os.path.join(
            self.output_dir,
            'image_timestamps.csv'
        )


        self.bridge = CvBridge()

        # Latest image received from each topic
        self.latest_images = {
            'rgb1': None,
            'rgb2': None,
            'thermal1_gray': None,
            'thermal2_gray': None
        }

        # ROS timestamp corresponding to each latest image
        self.image_timestamps = {
            'rgb1': None,
            'rgb2': None,
            'thermal1_gray': None,
            'thermal2_gray': None
        }

        # Protect image data from simultaneous access by
        # ROS callbacks and the capture function.
        self.image_lock = threading.Lock()

        self.capture_count = 0

        # ============================================================
        # IMU CSV
        # ============================================================

        self.imu_file = open(
            self.imu_file_path,
            'w',
            newline='',
            buffering=1
        )

        self.imu_writer = csv.writer(self.imu_file)

        self.imu_writer.writerow([
            'timestamp',
            'ax',
            'ay',
            'az',
            'gx',
            'gy',
            'gz',
            'qx',
            'qy',
            'qz',
            'qw'
        ])

        # ============================================================
        # Image timestamp CSV
        # ============================================================

        self.timestamp_file = open(
            self.timestamp_file_path,
            'w',
            newline=''
        )

        self.timestamp_writer = csv.writer(
            self.timestamp_file
        )

        self.timestamp_writer.writerow([
            'capture_id',
            'rgb1_timestamp',
            'rgb2_timestamp',
            'thermal1_gray_timestamp',
            'thermal2_gray_timestamp'
        ])

        # ============================================================
        # Image subscribers
        # ============================================================

        self.create_subscription(
            Image,
            '/rgb1/image_raw',
            self.rgb1_callback,
            10
        )

        self.create_subscription(
            Image,
            '/rgb2/image_raw',
            self.rgb2_callback,
            10
        )

        self.create_subscription(
            Image,
            '/thermal1/image_gray',
            self.thermal1_gray_callback,
            10
        )

        self.create_subscription(
            Image,
            '/thermal2/image_gray',
            self.thermal2_gray_callback,
            10
        )

        # ============================================================
        # IMU subscriber
        # ============================================================

        self.create_subscription(
            Imu,
            '/imu/data_raw',
            self.imu_callback,
            1000
        )

        # ============================================================
        # Status
        # ============================================================

        self.get_logger().info(
            '=============================================='
        )

        self.get_logger().info(
            'Data capture node started'
        )

        self.get_logger().info(
            f'Saving to: {self.output_dir}'
        )

        self.get_logger().info(
            'IMU recording continuously'
        )

        self.get_logger().info(
            '=============================================='
        )

    # ================================================================
    # Utility
    # ================================================================

    @staticmethod
    def ros_time_to_seconds(stamp):
        return (
            stamp.sec +
            stamp.nanosec * 1e-9
        )

    # ================================================================
    # Image processing
    # ================================================================

    def process_image(self, msg, name):

        try:

            image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='passthrough'
            )

            timestamp = self.ros_time_to_seconds(
                msg.header.stamp
            )

            with self.image_lock:

                self.latest_images[name] = image

                self.image_timestamps[name] = timestamp

        except Exception as e:

            self.get_logger().error(
                f'Error processing {name}: {e}'
            )

    # ================================================================
    # Image callbacks
    # ================================================================

    def rgb1_callback(self, msg):
        self.process_image(msg, 'rgb1')

    def rgb2_callback(self, msg):
        self.process_image(msg, 'rgb2')

    def thermal1_gray_callback(self, msg):
        self.process_image(msg, 'thermal1_gray')

    def thermal2_gray_callback(self, msg):
        self.process_image(msg, 'thermal2_gray')

    # ================================================================
    # IMU callback
    # ================================================================

    def imu_callback(self, msg):

        timestamp = self.ros_time_to_seconds(
            msg.header.stamp
        )

        self.imu_writer.writerow([
            timestamp,

            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,

            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,

            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ])

    # ================================================================
    # Capture images
    # ================================================================

    def capture_images(self):

        # ------------------------------------------------------------
        # Copy the latest images and timestamps while holding the lock
        # ------------------------------------------------------------

        with self.image_lock:

            images = {}

            for name, image in self.latest_images.items():

                if image is not None:
                    images[name] = image.copy()
                else:
                    images[name] = None

            timestamps = dict(
                self.image_timestamps
            )

        # ------------------------------------------------------------
        # Create capture directory
        # ------------------------------------------------------------

        self.capture_count += 1

        capture_id = self.capture_count

        capture_folder = os.path.join(
            self.image_dir,
            f'{capture_id:06d}'
        )

        os.makedirs(
            capture_folder,
            exist_ok=True
        )

        # ------------------------------------------------------------
        # File names
        # ------------------------------------------------------------

        filenames = {
            'rgb1': 'rgb1.png',
            'rgb2': 'rgb2.png',
            'thermal1_gray': 'thermal1_gray.png',
            'thermal2_gray': 'thermal2_gray.png'
        }

        # ------------------------------------------------------------
        # Save images
        # ------------------------------------------------------------

        for name, filename in filenames.items():

            image = images[name]

            if image is None:

                self.get_logger().warning(
                    f'{name}: no image received yet'
                )

                continue

            path = os.path.join(
                capture_folder,
                filename
            )

            success = cv2.imwrite(
                path,
                image
            )

            if not success:

                self.get_logger().error(
                    f'Failed to save {path}'
                )

        # ------------------------------------------------------------
        # Save timestamps
        # ------------------------------------------------------------

        self.timestamp_writer.writerow([
            capture_id,
            timestamps['rgb1'],
            timestamps['rgb2'],
            timestamps['thermal1_gray'],
            timestamps['thermal2_gray']
        ])

        self.timestamp_file.flush()

        # ------------------------------------------------------------
        # Print information
        # ------------------------------------------------------------

        self.get_logger().info(
            f'CAPTURE {capture_id:06d} saved'
        )

        for name in timestamps:

            timestamp = timestamps[name]

            if timestamp is not None:

                self.get_logger().info(
                    f'  {name}: {timestamp:.9f}'
                )

            else:

                self.get_logger().warning(
                    f'  {name}: NO IMAGE'
                )

    # ================================================================
    # Close files
    # ================================================================

    def close_files(self):

        self.get_logger().info(
            'Closing files...'
        )

        if not self.imu_file.closed:

            self.imu_file.flush()
            self.imu_file.close()

        if not self.timestamp_file.closed:

            self.timestamp_file.flush()
            self.timestamp_file.close()

        self.get_logger().info(
            'Data saved successfully.'
        )

        self.get_logger().info(
            f'Experiment directory: {self.output_dir}'
        )


# ====================================================================
# Keyboard handling
# ====================================================================

def keyboard_loop(node):

    print()
    print('==============================================')
    print('          DATA CAPTURE RUNNING')
    print('==============================================')
    print("Press 'c' to capture images")
    print("Press 'q' to quit")
    print('==============================================')
    print()

    # Save current terminal settings
    old_settings = termios.tcgetattr(sys.stdin)

    try:

        # Put terminal into character-at-a-time mode
        tty.setcbreak(sys.stdin.fileno())

        while rclpy.ok():

            key = sys.stdin.read(1)

            if key == 'c':

                node.capture_images()

            elif key == 'q':

                print('\nStopping...')
                rclpy.shutdown()
                break

    finally:

        # Restore normal terminal behaviour
        termios.tcsetattr(
            sys.stdin,
            termios.TCSADRAIN,
            old_settings
        )


# ====================================================================
# Main
# ====================================================================

def main(args=None):

    rclpy.init(args=args)

    node = DataCaptureNode()

    # Keyboard runs independently from ROS callbacks
    keyboard_thread = threading.Thread(
        target=keyboard_loop,
        args=(node,),
        daemon=True
    )

    keyboard_thread.start()

    try:

        # ROS callbacks run continuously here
        rclpy.spin(node)

    except KeyboardInterrupt:

        print('\nCtrl+C detected. Stopping...')

    finally:

        # Close CSV files
        node.close_files()

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()