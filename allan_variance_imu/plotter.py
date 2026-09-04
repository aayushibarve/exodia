import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu

import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import math


def quat_to_euler(x, y, z, w):
    """
    Convert quaternion -> roll, pitch, yaw (in radians)
    """

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll*180/np.pi, pitch*180/np.pi, yaw*180/np.pi


class IMULivePlotter(Node):
    def __init__(self):
        super().__init__('imu_live_plotter')

        self.sub = self.create_subscription(
            Imu,
            '/imu/data_raw',
            self.callback,
            10
        )

        self.N = 200
        self.t = deque(maxlen=self.N)
        self.i = 0

        # accel
        self.ax = deque(maxlen=self.N)
        self.ay = deque(maxlen=self.N)
        self.az = deque(maxlen=self.N)

        # gyro
        self.gx = deque(maxlen=self.N)
        self.gy = deque(maxlen=self.N)
        self.gz = deque(maxlen=self.N)

        # orientation (Euler)
        self.roll = deque(maxlen=self.N)
        self.pitch = deque(maxlen=self.N)
        self.yaw = deque(maxlen=self.N)

        # matplotlib setup
        plt.ion()
        self.fig, axes = plt.subplots(3, 1, figsize=(8, 8))

        self.acc_ax, self.gyro_ax, self.orient_ax = axes

        # accel lines
        self.lax, = self.acc_ax.plot([], [], label='ax')
        self.lay, = self.acc_ax.plot([], [], label='ay')
        self.laz, = self.acc_ax.plot([], [], label='az')

        # gyro lines
        self.lgx, = self.gyro_ax.plot([], [], label='gx')
        self.lgy, = self.gyro_ax.plot([], [], label='gy')
        self.lgz, = self.gyro_ax.plot([], [], label='gz')

        # orientation lines
        self.lroll, = self.orient_ax.plot([], [], label='roll')
        self.lpitch, = self.orient_ax.plot([], [], label='pitch')
        self.lyaw, = self.orient_ax.plot([], [], label='yaw')

        self.acc_ax.set_title("Linear Acceleration")
        self.gyro_ax.set_title("Angular Velocity")
        self.orient_ax.set_title("Orientation (Euler)")

        for ax in axes:
            ax.legend()

        plt.show()

        self.create_timer(0.1, self.update_plot)

    def callback(self, msg: Imu):
        # accel
        self.ax.append(msg.linear_acceleration.x)
        self.ay.append(msg.linear_acceleration.y)
        self.az.append(msg.linear_acceleration.z)

        # gyro
        self.gx.append(msg.angular_velocity.x)
        self.gy.append(msg.angular_velocity.y)
        self.gz.append(msg.angular_velocity.z)

        # quaternion → Euler
        q = msg.orientation
        r, p, y = quat_to_euler(q.x, q.y, q.z, q.w)

        self.roll.append(r)
        self.pitch.append(p)
        self.yaw.append(y)

        self.t.append(self.i)
        self.i += 1

    def update_plot(self):
        if len(self.t) < 2:
            return

        # accel
        self.lax.set_data(self.t, self.ax)
        self.lay.set_data(self.t, self.ay)
        self.laz.set_data(self.t, self.az)

        # gyro
        self.lgx.set_data(self.t, self.gx)
        self.lgy.set_data(self.t, self.gy)
        self.lgz.set_data(self.t, self.gz)

        # orientation
        self.lroll.set_data(self.t, self.roll)
        self.lpitch.set_data(self.t, self.pitch)
        self.lyaw.set_data(self.t, self.yaw)

        # autoscale
        for ax in [self.acc_ax, self.gyro_ax, self.orient_ax]:
            ax.relim()
            ax.autoscale_view()

        plt.pause(0.001)


def main():
    rclpy.init()
    node = IMULivePlotter()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()