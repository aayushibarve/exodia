import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np
import subprocess
import threading


class RGBCamera(Node):

    def __init__(self):
        super().__init__('rgb_camera')
        self.declare_parameter('device1', '/dev/video0')
        self.declare_parameter('device2', '')
        self.declare_parameter('stereo', False)
        self.declare_parameter('pixel_format1', 'MJPG')
        self.declare_parameter('pixel_format2', 'MJPG')
        self.declare_parameter('width', 1920)
        self.declare_parameter('height', 1080)
        self.declare_parameter('publish_rate', 30.0)
        
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.pixel_format1 = self.get_parameter('pixel_format1').value
        self.pixel_format2 = self.get_parameter('pixel_format2').value
        self.device1 = self.get_parameter('device1').value
        self.device2 = self.get_parameter('device2').value
        self.stereo = self.get_parameter('stereo').value
        self.publish_rate = self.get_parameter('publish_rate').value
        
        self.convert_rgb1 = 1
        self.convert_rgb2 = 1

        self.running=True
        self.lock = threading.Lock()

        if self.pixel_format1=='YUYV':
            self.convert_rgb1=0
        if self.pixel_format2=='YUYV':
            self.convert_rgb2=0


        self.publisher1 = self.create_publisher(
            Image,
            '/rgb1/image_raw',
            10
        )
            
        self.bridge = CvBridge()

        subprocess.run([
            "v4l2-ctl",
            f"--device={self.device1}",
            f"--set-fmt-video=width={self.width},height={self.height},pixelformat={self.pixel_format1}"
        ],
        check=False)         

        self.cap1 = cv2.VideoCapture(
            self.device1,
            cv2.CAP_V4L2
        )

        if not self.cap1.isOpened():
            self.get_logger().error(
                "Could not open RGB1 camera"
            )
            raise RuntimeError(
                "Camera open failed"
            )
        
        fourcc_map = {
            "MJPG": cv2.VideoWriter.fourcc(*"MJPG"),
            "YUYV": cv2.VideoWriter.fourcc('Y','U','Y','V')
        }
        fourcc1 = fourcc_map[self.pixel_format1]

        self.cap1.set(cv2.CAP_PROP_FOURCC,fourcc1)
        self.cap1.set(cv2.CAP_PROP_FRAME_WIDTH,self.width)
        self.cap1.set(cv2.CAP_PROP_FRAME_HEIGHT,self.height)
        self.cap1.set(cv2.CAP_PROP_CONVERT_RGB,self.convert_rgb1)

        if self.stereo:
            self.publisher2 = self.create_publisher(
                Image,
                '/rgb2/image_raw',
                10
            )
            subprocess.run([
            "v4l2-ctl",
            f"--device={self.device2}",
            f"--set-fmt-video=width={self.width},height={self.height},pixelformat={self.pixel_format2}"
            ],
            check=False)
            self.cap2 = cv2.VideoCapture(
                self.device2,
                cv2.CAP_V4L2
            )
            if not self.cap2.isOpened():
                self.get_logger().error(
                    "Could not open RGB2 camera"
                )
                raise RuntimeError(
                    "Camera open failed"
                )
            fourcc2 = fourcc_map[self.pixel_format2]

            self.cap2.set(cv2.CAP_PROP_FOURCC,fourcc2)
            self.cap2.set(cv2.CAP_PROP_FRAME_WIDTH,self.width)
            self.cap2.set(cv2.CAP_PROP_FRAME_HEIGHT,self.height)
            self.cap2.set(cv2.CAP_PROP_CONVERT_RGB,self.convert_rgb2)
        
        self.frame1=None
        self.frame2=None

        self.thread1 = threading.Thread(
            target=self.capture_camera1,
            daemon=True
        )
        self.thread1.start()

        if self.stereo:
            self.thread2 = threading.Thread(
                target=self.capture_camera2,
                daemon=True
            )
            self.thread2.start()

        self.timer = self.create_timer(
            1.0 / self.publish_rate,
            self.publish_frame
        )


        self.get_logger().info(
            "RGB camera node started"
        )

    def capture_camera1(self):
        while self.running:
            ret, frame = self.cap1.read()
            if ret:
                with self.lock:
                    self.frame1=frame
    
    def capture_camera2(self):
        while self.running:
            ret, frame = self.cap2.read()
            if ret:
                with self.lock:
                    self.frame2 = frame 

    def process_frame(self, raw_frame, pixel_format):
        if raw_frame is None:
            return None

        if pixel_format=='MJPG':
            return raw_frame
        
        elif pixel_format == "YUYV":
            display_frame = cv2.cvtColor(raw_frame, cv2.COLOR_YUV2BGR_YUYV)
            return display_frame

        
    def publish_frame(self):

        with self.lock:
            if self.frame1 is None:
                return
            frame_raw1 = self.frame1.copy()
        timestamp1 = self.get_clock().now().to_msg()
        rgb_frame1 = self.process_frame(frame_raw1, self.pixel_format1)

        if self.stereo:
            with self.lock:
                if self.frame2 is None:
                    return
                frame_raw2 = self.frame2.copy()
            timestamp2 = self.get_clock().now().to_msg()
            rgb_frame2 = self.process_frame(frame_raw2, self.pixel_format2)

        msg1 = self.bridge.cv2_to_imgmsg(
            rgb_frame1,
            encoding="bgr8"
        )

        msg1.header.stamp = timestamp1
        msg1.header.frame_id = "rgb_camera1"
        if self.stereo:
            msg2 = self.bridge.cv2_to_imgmsg(
            rgb_frame2,
            encoding="bgr8"
            )

            msg2.header.stamp = timestamp2
            msg2.header.frame_id = "rgb_camera2"
            self.publisher2.publish(msg2)

        self.publisher1.publish(msg1)




    def destroy_node(self):
        self.running=False    
        self.cap1.release()
        if self.stereo:
            self.cap2.release()
        super().destroy_node()



def main(args=None):

    rclpy.init(args=args)

    node = RGBCamera()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()



if __name__ == '__main__':
    main()
