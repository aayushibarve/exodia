import rosbag2_py
import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import Imu


BAG_PATH = "rosbag_test_4h_final"
TOPIC = "/imu/data_raw"


def open_bag(path):
    storage_options = rosbag2_py.StorageOptions(
        uri=path,
        storage_id="sqlite3"
    )

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr"
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    return reader


def stats(name, arr):
    arr = np.array(arr)

    print(f"\n--- {name} ---")
    print(f"count: {len(arr)}")
    print(f"mean : {np.mean(arr):.6e}")
    print(f"std  : {np.std(arr):.6e}")
    print(f"min  : {np.min(arr):.6e}")
    print(f"max  : {np.max(arr):.6e}")

    print(f"NaN  : {np.isnan(arr).sum()}")
    print(f"Inf  : {np.isinf(arr).sum()}")
    if name=="gyro_z":
        print(np.unique(arr))


def main():
    reader = open_bag(BAG_PATH)

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}

    imu_type = get_message(type_map[TOPIC])

    gx, gy, gz = [], [], []
    ax, ay, az = [], [], []

    t_prev = None
    dt_list = []

    while reader.has_next():
        topic, msg, t = reader.read_next()

        if topic != TOPIC:
            continue

        imu = deserialize_message(msg, imu_type)

        gx.append(imu.angular_velocity.x)
        gy.append(imu.angular_velocity.y)
        gz.append(imu.angular_velocity.z)

        ax.append(imu.linear_acceleration.x)
        ay.append(imu.linear_acceleration.y)
        az.append(imu.linear_acceleration.z)

        if t_prev is not None:
            dt_list.append((t - t_prev) * 1e-9)
        t_prev = t

    dt_list = np.array(dt_list)

    print(f"Samples: {len(gx)}")

    if len(dt_list) > 0:
        fs = 1.0 / np.mean(dt_list)
        print(f"Estimated sampling rate: {fs:.2f} Hz")
        print(f"dt std deviation: {np.std(dt_list):.6e} s")

    print("\n================ GYROSCOPE =======================")

    stats("gyro_x", gx)
    stats("gyro_y", gy)
    stats("gyro_z", gz)

    print("\n================ ACCELEROMETER ===================")

    stats("accel_x", ax)
    stats("accel_y", ay)
    stats("accel_z", az)
    ax=np.array(ax)
    ay=np.array(ay)
    az=np.array(az)
    print(np.average(ax**2+ay**2+az**2))
    print("\n==================================================")

if __name__ == "__main__":
    main()