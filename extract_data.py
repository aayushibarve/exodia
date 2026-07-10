import rosbag2_py
import pandas as pd
import numpy as np

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from sensor_msgs.msg import Imu


BAG_PATH = "rosbag_raw_vals_4h_set2"   
TOPIC = "/imu/data_raw"
OUTPUT_CSV = "imu_extracted_raw2.csv"


def open_reader(bag_path):
    storage_options = rosbag2_py.StorageOptions(
        uri=bag_path,
        storage_id="sqlite3"
    )

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr"
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    return reader


def main():
    reader = open_reader(BAG_PATH)

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}

    imu_type = get_message(type_map[TOPIC])

    data = []

    while reader.has_next():
        topic, msg, t = reader.read_next()

        if topic != TOPIC:
            continue

        imu = deserialize_message(msg, imu_type)

        stamp_ns = imu.header.stamp.sec * 10**9 + imu.header.stamp.nanosec
        stamp_sec = stamp_ns * 1e-9

        data.append([
            stamp_ns,
            stamp_sec,

            imu.angular_velocity.x,
            imu.angular_velocity.y,
            imu.angular_velocity.z,

            imu.linear_acceleration.x,
            imu.linear_acceleration.y,
            imu.linear_acceleration.z,

            imu.orientation.x,
            imu.orientation.y,
            imu.orientation.z,
            imu.orientation.w,
        ])

    df = pd.DataFrame(data, columns=[
        "timestamp_ns",
        "timestamp_sec",

        "gyro_x",
        "gyro_y",
        "gyro_z",

        "accel_x",
        "accel_y",
        "accel_z",

        "quat_x",
        "quat_y",
        "quat_z",
        "quat_w",
    ])

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} samples to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()