import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, Imu


class TimestampMonitor(Node):

    def __init__(self):
        super().__init__('timestamp_monitor')

        self.rgb1_time = None
        self.rgb2_time = None
        self.thermal_raw_time = None
        self.thermal_gray_time = None
        self.imu_time = None

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
            '/thermal/image_raw',
            self.thermal_callback1,
            10
        )

        self.create_subscription(
            Image,
            '/thermal/image_gray',
            self.thermal_callback2,
            10
        )

        self.create_subscription(
            Imu,
            '/imu/data_raw',
            self.imu_callback,
            100
        )

        # Print at 30 Hz
        self.timer = self.create_timer(
            1.0 / 5.0,
            self.print_timestamps
        )

    def stamp_to_sec(self, stamp):
        return stamp.sec + stamp.nanosec * 1e-9

    def rgb1_callback(self, msg):
        self.rgb1_time = self.stamp_to_sec(msg.header.stamp)

    def rgb2_callback(self, msg):
        self.rgb2_time = self.stamp_to_sec(msg.header.stamp)

    def thermal_callback1(self, msg):
        self.thermal_raw_time = self.stamp_to_sec(msg.header.stamp)

    def thermal_callback2(self, msg):
        self.thermal_gray_time = self.stamp_to_sec(msg.header.stamp)

    def imu_callback(self, msg):
        self.imu_time = self.stamp_to_sec(msg.header.stamp)

    def print_timestamps(self):

        if self.rgb1_time is None:
            return

        print("\n-----------------------------")

        if self.rgb1_time:
            print(f"RGB1         : {self.rgb1_time:.9f}")

        if self.rgb2_time:
            print(f"RGB2         : {self.rgb2_time:.9f}")

        if self.thermal_raw_time:
            print(f"Thermal raw  : {self.thermal_raw_time:.9f}")

        if self.thermal_gray_time:
            print(f"Thermal gray : {self.thermal_gray_time:.9f}")

        if self.imu_time:
            print(f"IMU          : {self.imu_time:.9f}")

        if self.rgb1_time and self.thermal_raw_time:
            diff = self.thermal_raw_time - self.rgb1_time
            print(f"Thermal-RGB1: {diff*1000:.3f} ms")

        if self.rgb1_time and self.rgb2_time:
            diff = self.rgb2_time - self.rgb1_time
            print(f"RGB2-RGB1: {diff*1000:.3f} ms")


def main(args=None):
    rclpy.init(args=args)

    node = TimestampMonitor()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
