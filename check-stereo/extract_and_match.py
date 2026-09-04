"""
Extract /rgb1/image_raw and /rgb2/image_raw from a ROS2 bag
Pairing rule :
- If both topics have the same number of frames -> pair by arrival order
  (i.e. the order the messages were recorded in, index-for-index).
- If counts differ -> pair by nearest-timestamp, greedy one-to-one
  assignment (smallest |t1 - t2| wins first, each frame used at most once).
  Frames that don't get a partner are reported and dropped.

Usage:
    python extract_and_match.py \
        --bag /path/to/bag_dir \
        --out /path/to/extracted \
        --left-topic /rgb1/image_raw \
        --right-topic /rgb2/image_raw \
        --max-dt 0.05

Output:
    <out>/rgb1/000000_<t>.png, <out>/rgb2/000000_<t>.png, ...
    <out>/manifest.json   -- list of matched pairs with paths + timestamps
    <out>/unmatched.json  -- frames that couldn't be paired (if any)
"""
import argparse
import json
import os
import yaml

import numpy as np
import cv2

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def detect_storage_id(bag_path):
    """Read metadata.yaml to figure out sqlite3 vs mcap, fall back to sqlite3."""
    meta_path = os.path.join(bag_path, "metadata.yaml")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = yaml.safe_load(f)
        try:
            return meta["rosbag2_bagfile_information"]["storage_identifier"]
        except (KeyError, TypeError):
            pass
    return "sqlite3"


def open_reader(bag_path):
    storage_id = detect_storage_id(bag_path)
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr", output_serialization_format="cdr"
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    return reader


def imgmsg_to_bgr(msg):
    """Convert sensor_msgs/Image to a BGR (or mono) numpy array without cv_bridge."""
    h, w, enc = msg.height, msg.width, msg.encoding
    buf = np.frombuffer(msg.data, dtype=np.uint8)

    if enc in ("bgr8", "rgb8"):
        img = buf.reshape(h, w, 3)
        if enc == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    if enc in ("mono8", "8UC1"):
        return buf.reshape(h, w)
    if enc in ("mono16", "16UC1"):
        return buf.view(np.uint16).reshape(h, w)
    if enc == "bayer_rggb8":
        raw = buf.reshape(h, w)
        return cv2.cvtColor(raw, cv2.COLOR_BayerRGGB2BGR)
    if enc == "yuv422" or enc == "uyvy":
        raw = buf.reshape(h, w, 2)
        return cv2.cvtColor(raw, cv2.COLOR_YUV2BGR_UYVY)

    raise ValueError(
        f"Unhandled image encoding '{enc}'. Add a case in imgmsg_to_bgr(), "
        f"or install cv_bridge and swap this function for CvBridge().imgmsg_to_cv2()."
    )


def extract_topic(bag_path, topic_name, out_dir):
    reader = open_reader(bag_path)
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic_name not in type_map:
        raise RuntimeError(
            f"Topic '{topic_name}' not found in bag. Available: {list(type_map)}"
        )
    msg_type = get_message(type_map[topic_name])

    os.makedirs(out_dir, exist_ok=True)
    frames = []  # list of dicts: index, t (float sec), path

    storage_filter = rosbag2_py.StorageFilter(topics=[topic_name])
    reader.set_filter(storage_filter)

    idx = 0
    while reader.has_next():
        topic, data, _bag_time_ns = reader.read_next()
        msg = deserialize_message(data, msg_type)
        img = imgmsg_to_bgr(msg)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        fname = f"{idx:06d}_{t:.6f}.png"
        path = os.path.join(out_dir, fname)
        cv2.imwrite(path, img)
        frames.append({"index": idx, "t": t, "path": path})
        idx += 1

    return frames


def match_by_order(frames_a, frames_b):
    """Equal counts: pair index-for-index in recorded order."""
    assert len(frames_a) == len(frames_b)
    return [
        (a["index"], b["index"], abs(a["t"] - b["t"]))
        for a, b in zip(frames_a, frames_b)
    ]


def match_by_timestamp(frames_a, frames_b, max_dt=None):
    """Unequal counts: greedy nearest-timestamp, one-to-one, smallest diff first."""
    candidates = []
    for i, a in enumerate(frames_a):
        for j, b in enumerate(frames_b):
            d = abs(a["t"] - b["t"])
            if max_dt is None or d <= max_dt:
                candidates.append((d, i, j))
    candidates.sort(key=lambda x: x[0])

    used_a, used_b = set(), set()
    matches = []
    for d, i, j in candidates:
        if i in used_a or j in used_b:
            continue
        matches.append((frames_a[i]["index"], frames_b[j]["index"], d))
        used_a.add(i)
        used_b.add(j)

    unmatched_a = [frames_a[i]["index"] for i in range(len(frames_a)) if i not in used_a]
    unmatched_b = [frames_b[j]["index"] for j in range(len(frames_b)) if j not in used_b]
    matches.sort(key=lambda m: min(
        [f["t"] for f in frames_a if f["index"] == m[0]][0],
        [f["t"] for f in frames_b if f["index"] == m[1]][0],
    ))
    return matches, unmatched_a, unmatched_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, help="Path to ROS2 bag directory")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--left-topic", default="/rgb1/image_raw")
    ap.add_argument("--right-topic", default="/rgb2/image_raw")
    ap.add_argument(
        "--max-dt", type=float, default=None,
        help="Optional max |t1-t2| (sec) allowed when timestamp-matching unequal counts. "
             "Leave unset to allow any distance (all frames get a best-effort match).",
    )
    args = ap.parse_args()

    left_dir = os.path.join(args.out, "rgb1")
    right_dir = os.path.join(args.out, "rgb2")

    print(f"Extracting {args.left_topic} ...")
    left_frames = extract_topic(args.bag, args.left_topic, left_dir)
    print(f"  {len(left_frames)} frames -> {left_dir}")

    print(f"Extracting {args.right_topic} ...")
    right_frames = extract_topic(args.bag, args.right_topic, right_dir)
    print(f"  {len(right_frames)} frames -> {right_dir}")

    unmatched_a, unmatched_b = [], []
    if len(left_frames) == len(right_frames):
        print("Equal counts -> pairing by recorded order.")
        raw_matches = match_by_order(left_frames, right_frames)
    else:
        print(f"Unequal counts ({len(left_frames)} vs {len(right_frames)}) "
              f"-> pairing by nearest timestamp.")
        raw_matches, unmatched_a, unmatched_b = match_by_timestamp(
            left_frames, right_frames, max_dt=args.max_dt
        )

    left_by_idx = {f["index"]: f for f in left_frames}
    right_by_idx = {f["index"]: f for f in right_frames}

    manifest = []
    for pair_idx, (li, ri, dt) in enumerate(raw_matches):
        manifest.append({
            "pair_index": pair_idx,
            "rgb1_path": left_by_idx[li]["path"],
            "rgb2_path": right_by_idx[ri]["path"],
            "t1": left_by_idx[li]["t"],
            "t2": right_by_idx[ri]["t"],
            "dt": dt,
        })

    manifest_path = os.path.join(args.out, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {len(manifest)} matched pairs -> {manifest_path}")

    if unmatched_a or unmatched_b:
        unmatched_path = os.path.join(args.out, "unmatched.json")
        with open(unmatched_path, "w") as f:
            json.dump({"rgb1_unmatched": unmatched_a, "rgb2_unmatched": unmatched_b}, f, indent=2)
        print(f"{len(unmatched_a)} rgb1 + {len(unmatched_b)} rgb2 frames left unmatched "
              f"-> {unmatched_path}")


if __name__ == "__main__":
    main()
