import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np
import subprocess
import threading


class ThermalCamera(Node):

    def __init__(self):
        super().__init__('thermal_camera')

        self.declare_parameter('device1', '/dev/video0')
        self.declare_parameter('device2', '')
        self.declare_parameter('stereo', False)

        self.declare_parameter('camera1', 'nv')
        self.declare_parameter('camera2', 'brick2')

        self.declare_parameter('pixel_format1', 'UYVY')
        self.declare_parameter('pixel_format2', 'UYVY')

        self.declare_parameter('width1', 512)
        self.declare_parameter('height1', 192)

        self.declare_parameter('width2', 256)
        self.declare_parameter('height2', 192)

        self.declare_parameter('publish_rate', 30.0)

        self.device1 = self.get_parameter('device1').value
        self.device2 = self.get_parameter('device2').value
        self.stereo = self.get_parameter('stereo').value

        self.camera1 = self.get_parameter('camera1').value
        self.camera2 = self.get_parameter('camera2').value

        self.pixel_format1 = self.get_parameter('pixel_format1').value
        self.pixel_format2 = self.get_parameter('pixel_format2').value

        self.width1 = self.get_parameter('width1').value
        self.height1 = self.get_parameter('height1').value

        self.width2 = self.get_parameter('width2').value
        self.height2 = self.get_parameter('height2').value

        self.publish_rate = self.get_parameter('publish_rate').value

        self.frame1 = None
        self.frame2 = None

        self.running = True
        self.lock = threading.Lock()

        self.bridge = CvBridge()

        self.fourcc_map = {
            "Y16": cv2.VideoWriter.fourcc('Y','1','6',' '),
            "UYVY": cv2.VideoWriter.fourcc('U','Y','V','Y'),
            "YUYV": cv2.VideoWriter.fourcc('Y','U','Y','V'),
            "GREY": cv2.VideoWriter.fourcc('G','R','E','Y'),
            "RGBP": cv2.VideoWriter.fourcc('R','G','B','P'),
            "BGR3": cv2.VideoWriter.fourcc('B','G','R','3')
        }

        self.publisher_gray1 = self.create_publisher(Image, '/thermal1/image_gray', 10)
        self.publisher_raw1 = self.create_publisher(Image, '/thermal1/image_raw', 10)

        if self.stereo:
            self.publisher_gray2 = self.create_publisher(Image, '/thermal2/image_gray', 10)
            self.publisher_raw2 = self.create_publisher(Image, '/thermal2/image_raw', 10)

        self.setup_camera1()

        if self.stereo:
            self.setup_camera2()

        self.thread1 = threading.Thread(target=self.capture_camera1, daemon=True)
        self.thread1.start()

        if self.stereo:
            self.thread2 = threading.Thread(target=self.capture_camera2, daemon=True)
            self.thread2.start()

        self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_frame)

        self.get_logger().info("Thermal camera node started")


    def setup_camera1(self):

        subprocess.run([
            "v4l2-ctl",
            f"--device={self.device1}",
            f"--set-fmt-video=width={self.width1},height={self.height1},pixelformat={self.pixel_format1}"
        ], check=False)

        self.cap1 = cv2.VideoCapture(self.device1, cv2.CAP_V4L2)

        if not self.cap1.isOpened():
            raise RuntimeError("Could not open thermal camera 1")

        self.cap1.set(cv2.CAP_PROP_FOURCC, self.fourcc_map[self.pixel_format1])
        self.cap1.set(cv2.CAP_PROP_FRAME_WIDTH, self.width1)
        self.cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height1)
        self.cap1.set(cv2.CAP_PROP_CONVERT_RGB, 0)


    def setup_camera2(self):

        subprocess.run([
            "v4l2-ctl",
            f"--device={self.device2}",
            f"--set-fmt-video=width={self.width2},height={self.height2},pixelformat={self.pixel_format2}"
        ], check=False)

        self.cap2 = cv2.VideoCapture(self.device2, cv2.CAP_V4L2)

        if not self.cap2.isOpened():
            raise RuntimeError("Could not open thermal camera 2")

        self.cap2.set(cv2.CAP_PROP_FOURCC, self.fourcc_map[self.pixel_format2])
        self.cap2.set(cv2.CAP_PROP_FRAME_WIDTH, self.width2)
        self.cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height2)
        self.cap2.set(cv2.CAP_PROP_CONVERT_RGB, 0)


    def capture_camera1(self):

        while self.running:
            ret, frame = self.cap1.read()

            if ret:
                with self.lock:
                    self.frame1 = frame


    def capture_camera2(self):

        while self.running:
            ret, frame = self.cap2.read()

            if ret:
                with self.lock:
                    self.frame2 = frame


    def process_frame(self, raw_frame, pixel_format, camera, width, height):

        if raw_frame is None:
            return None, None

        if pixel_format == "Y16":
            gray16 = raw_frame.astype(np.uint16)
            display = cv2.normalize(gray16, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            return raw_frame, display

        elif pixel_format == "GREY":
            return raw_frame, raw_frame

        elif pixel_format == "UYVY":
            display = cv2.cvtColor(raw_frame, cv2.COLOR_YUV2BGR_UYVY)

            if camera == "nv":
                display = display[:, 256:]

            gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)

            return raw_frame, gray

        elif pixel_format == "YUYV":
            display = cv2.cvtColor(raw_frame, cv2.COLOR_YUV2BGR_YUYV)
            gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)

            return raw_frame, gray

        elif pixel_format == "RGBP":
            raw8 = np.frombuffer(raw_frame, dtype=np.uint8)
            frame = raw8.reshape((height, width, 2))
            bgr = cv2.cvtColor(frame, cv2.COLOR_BGR5652BGR)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            return frame, gray

        elif pixel_format == "BGR3":
            gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
            return raw_frame, gray


    def get_raw_encoding(self, pixel_format):

        if pixel_format == "Y16":
            return "mono16"

        elif pixel_format == "GREY":
            return "mono8"

        elif pixel_format == "BGR3":
            return "bgr8"

        return "8UC2"


    def publish_frame(self):

        with self.lock:
            if self.frame1 is None:
                return

            frame1 = self.frame1.copy()
            timestamp1 = self.get_clock().now().to_msg()

            if self.stereo:
                if self.frame2 is None:
                    return

                frame2 = self.frame2.copy()
                timestamp2 = self.get_clock().now().to_msg()

        raw1, gray1 = self.process_frame(
            frame1,
            self.pixel_format1,
            self.camera1,
            self.width1,
            self.height1
        )

        msg_gray1 = self.bridge.cv2_to_imgmsg(gray1, encoding="mono8")
        msg_gray1.header.stamp = timestamp1
        msg_gray1.header.frame_id = "thermal_camera1"

        msg_raw1 = self.bridge.cv2_to_imgmsg(raw1, encoding=self.get_raw_encoding(self.pixel_format1))
        msg_raw1.header.stamp = timestamp1
        msg_raw1.header.frame_id = "thermal_camera1"

        self.publisher_gray1.publish(msg_gray1)
        self.publisher_raw1.publish(msg_raw1)


        if self.stereo:

            raw2, gray2 = self.process_frame(
                frame2,
                self.pixel_format2,
                self.camera2,
                self.width2,
                self.height2
            )

            msg_gray2 = self.bridge.cv2_to_imgmsg(gray2, encoding="mono8")
            msg_gray2.header.stamp = timestamp2
            msg_gray2.header.frame_id = "thermal_camera2"

            msg_raw2 = self.bridge.cv2_to_imgmsg(raw2, encoding=self.get_raw_encoding(self.pixel_format2))
            msg_raw2.header.stamp = timestamp2
            msg_raw2.header.frame_id = "thermal_camera2"

            self.publisher_gray2.publish(msg_gray2)
            self.publisher_raw2.publish(msg_raw2)


    def destroy_node(self):

        self.running = False

        self.cap1.release()

        if self.stereo:
            self.cap2.release()

        super().destroy_node()



def main(args=None):

    rclpy.init(args=args)

    node = ThermalCamera()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
