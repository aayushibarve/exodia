"""
Usage:
    python3 extract_mocap_csv.py /path/to/bag_dir_or_mcap_file [--topic /macortex_bridge/exodia/pose] [--out mocap_aligned_trajectory.csv]
"""

import argparse
import os
from pathlib import Path
import numpy as np

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

DEFAULT_TOPIC = "/macortex_bridge/exodia/pose"
DEFAULT_OUT = "mocap_aligned_trajectory.csv"


# ============================================================
# QUATERNION / TRANSFORM HELPERS  (ROS quat convention: x, y, z, w)
# ============================================================

def quat_to_R(qx, qy, qz, qw):
    """Unit quaternion (x,y,z,w) -> 3x3 rotation matrix."""
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    R = np.array([
        [1 - 2 * (qy**2 + qz**2),     2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [    2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2),     2 * (qy * qz - qx * qw)],
        [    2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
    ])
    return R


def R_to_quat(R):
    """3x3 rotation matrix -> quaternion (x,y,z,w)."""
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return np.array([qx, qy, qz, qw])


def align_to_first_pose(positions, quats):
    """
    positions: (N,3) world-frame translations t_wb(t)
    quats:     (N,4) world-frame orientations (x,y,z,w), R_wb(t)

    Returns positions_rel, quats_rel such that pose[0] -> (0,0,0), identity quat.
    New world frame = body frame at t=0.

        R_rel(t) = R(0)^T @ R(t)
        t_rel(t) = R(0)^T @ (t(t) - t(0))
    """
    R0 = quat_to_R(*quats[0])
    t0 = positions[0]
    R0_T = R0.T

    positions_rel = (positions - t0) @ R0_T.T  # equivalent to R0_T @ (t - t0) per row
    quats_rel = np.zeros_like(quats)
    for i in range(len(quats)):
        Ri = quat_to_R(*quats[i])
        R_rel = R0_T @ Ri
        quats_rel[i] = R_to_quat(R_rel)
    return positions_rel, quats_rel


# ============================================================
# BAG READING
# ============================================================

def read_pose_topic(bag_path, topic):
    typestore = get_typestore(Stores.ROS2_JAZZY)
    stamps, positions, quats = [], [], []

    with AnyReader([Path(bag_path)], default_typestore=typestore) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            available = sorted({c.topic for c in reader.connections})
            raise RuntimeError(
                f"Topic '{topic}' not found in bag. Available topics:\n  " + "\n  ".join(available)
            )
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            p = msg.pose.position
            q = msg.pose.orientation
            stamps.append(timestamp * 1e-9)  # ns -> s
            positions.append([p.x, p.y, p.z])
            quats.append([q.x, q.y, q.z, q.w])

    order = np.argsort(stamps)
    stamps = np.array(stamps)[order]
    positions = np.array(positions)[order]
    quats = np.array(quats)[order]
    stamps = stamps - stamps[0]
    return stamps, positions, quats


def save_csv(stamps, positions, quats, out_path):
    header = "t_sec,x,y,z,qx,qy,qz,qw"
    data = np.column_stack([stamps, positions, quats])
    np.savetxt(out_path, data, delimiter=",", header=header, comments="", fmt="%.9f")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag_path", help="Path to rosbag2 directory (containing metadata.yaml) or .mcap file")
    ap.add_argument("--topic", default=DEFAULT_TOPIC)
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output CSV path")
    args = ap.parse_args()

    stamps, positions, quats = read_pose_topic(args.bag_path, args.topic)
    print(f"Loaded {len(stamps)} poses from '{args.topic}', duration {stamps[-1]:.2f} s")

    positions_rel, quats_rel = align_to_first_pose(positions, quats)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    save_csv(stamps, positions_rel, quats_rel, args.out)
    print(f"Saved aligned trajectory -> {args.out}")

    total_path_len = np.sum(np.linalg.norm(np.diff(positions_rel, axis=0), axis=1))
    net_disp = np.linalg.norm(positions_rel[-1])
    print(f"Path length: {total_path_len:.3f} m | net displacement: {net_disp:.3f} m")


if __name__ == "__main__":
    main()
