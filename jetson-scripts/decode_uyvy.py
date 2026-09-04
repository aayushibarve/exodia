import rclpy
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
from rclpy.serialization import deserialize_message

camera='brick1'
bag_path = "/home/disal/uyvy_test"

output_dir = "decoded_frames"
os.makedirs(output_dir, exist_ok=True)


reader = SequentialReader()

reader.open(
    StorageOptions(
        uri=bag_path,
        storage_id="mcap"
    ),
    ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr"
    )
)


bridge = CvBridge()

count = 0

while reader.has_next():

    topic, data, timestamp = reader.read_next()

    if topic != "/thermal/image_raw":
        continue

    msg = deserialize_message(
        data,
        Image
    )


    uyvy = bridge.imgmsg_to_cv2(
        msg,
        desired_encoding="passthrough"
    )

    bgr = cv2.cvtColor(
        uyvy,
        cv2.COLOR_YUV2BGR_UYVY
    )
    if camera=='brick1':
        bgr=bgr[:, 256:, :]

    success = cv2.imwrite(
        f"{output_dir}/frame_{count:04d}.png",
        bgr
    )


    count += 1

print("Decoded", count, "frames")