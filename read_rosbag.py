import os
import csv
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, Imu
from rclpy.serialization import deserialize_message

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory


# ----------- SETTINGS ------------

bag_path = "./depth_test/experiment_01/experiment_01_0.mcap"

output_dirs = {
    "/rgb1/image_raw": "./depth_test/experiment_01/rgb1",
    "/rgb2/image_raw": "./depth_test/experiment_01/rgb2",
    "/thermal1/image_gray": "./depth_test/experiment_01/thermal1",
    "/thermal2/image_gray": "./depth_test/experiment_01/thermal2",
}

imu_topic = "/imu/data_raw"
imu_csv_path = "./depth_test/experiment_01/imu.csv"

# ---------------------------------


# Create image output directories
for folder in output_dirs.values():
    os.makedirs(folder, exist_ok=True)

# Create output directory for IMU CSV
os.makedirs(os.path.dirname(imu_csv_path), exist_ok=True)

bridge = CvBridge()
decoder_factory = DecoderFactory()

counts = {topic: 0 for topic in output_dirs}
imu_count = 0


# Open IMU CSV file
with open(imu_csv_path, "w", newline="") as imu_file:

    imu_writer = csv.writer(imu_file)

    # CSV header
    imu_writer.writerow([
        "timestamp_sec",
        "timestamp_nanosec",
        "linear_acceleration_x",
        "linear_acceleration_y",
        "linear_acceleration_z",
        "angular_velocity_x",
        "angular_velocity_y",
        "angular_velocity_z",
        "orientation_x",
        "orientation_y",
        "orientation_z",
        "orientation_w"
    ])

    # Open rosbag
    with open(bag_path, "rb") as f:

        reader = make_reader(
            f,
            decoder_factories=[decoder_factory]
        )

        for schema, channel, message, ros_msg in reader.iter_decoded_messages():

            topic = channel.topic

            # ---------------- IMAGE TOPICS ----------------

            if topic in output_dirs:

                img_msg = ros_msg

                cv_img = bridge.imgmsg_to_cv2(
                    img_msg,
                    desired_encoding="passthrough"
                )

                timestamp = (
                    f"{img_msg.header.stamp.sec}_"
                    f"{img_msg.header.stamp.nanosec:09d}"
                )

                filename = os.path.join(
                    output_dirs[topic],
                    timestamp + ".png"
                )

                cv2.imwrite(filename, cv_img)

                counts[topic] += 1

            # ---------------- IMU TOPIC ----------------

            elif topic == imu_topic:

                imu_msg = ros_msg

                timestamp = imu_msg.header.stamp

                imu_writer.writerow([
                    timestamp.sec,
                    timestamp.nanosec,

                    imu_msg.linear_acceleration.x,
                    imu_msg.linear_acceleration.y,
                    imu_msg.linear_acceleration.z,

                    imu_msg.angular_velocity.x,
                    imu_msg.angular_velocity.y,
                    imu_msg.angular_velocity.z,

                    imu_msg.orientation.x,
                    imu_msg.orientation.y,
                    imu_msg.orientation.z,
                    imu_msg.orientation.w
                ])

                imu_count += 1


print("\nExtraction complete")

for topic, count in counts.items():
    print(topic, count)

print(f"{imu_topic}: {imu_count} messages")
print(f"IMU CSV: {imu_csv_path}")