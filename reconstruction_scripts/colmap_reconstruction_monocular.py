#!/usr/bin/env python3
import argparse
import csv
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

BAG_DIR = "/home/barve/exodia/mocap_thermal_rgb2"
CACHE_DIR = "/home/barve/exodia/vo_cache"

RGB_LEFT_TOPIC = "/rgb1/image_raw"
RGB_RIGHT_TOPIC = "/rgb2/image_raw"
THERMAL_LEFT_TOPIC = "/thermal1/image_gray"
THERMAL_RIGHT_TOPIC = "/thermal2/image_gray"
MOCAP_TOPIC = "/macortex_bridge/exodia/pose"
IMU_TOPIC = "/imu/data_raw"

COLMAP_SUBSAMPLE = 5
COLMAP_MAX_IMAGE_SIZE = 1600
SEQUENTIAL_OVERLAP = 10
ALIGN_MAX_ERROR = 0.05

COLMAP_WORKDIR = str(Path(CACHE_DIR).parent / (Path(CACHE_DIR).name + "_colmap"))
IMAGE_DIR = str(Path(COLMAP_WORKDIR) / "colmap_images")
DATABASE_PATH = str(Path(COLMAP_WORKDIR) / "database.db")
SPARSE_DIR = str(Path(COLMAP_WORKDIR) / "sparse")
ALIGNED_DIR = str(Path(COLMAP_WORKDIR) / "sparse_metric")
DENSE_DIR = str(Path(COLMAP_WORKDIR) / "dense")
REF_IMAGES_PATH = str(Path(COLMAP_WORKDIR) / "ref_images.txt")

IMG_SIZE = (1920, 1080)
LEFT_INTR = dict(fx=595.5131125646116, fy=595.4347842307814, cx=937.1438588828621, cy=459.52906005794046,
                  skew=0.0, k1=-0.006053561997736177, k2=-0.010116448442781898,
                  p1=-0.0007332457524041055, p2=-0.0017822070415447345)
RIGHT_INTR = dict(fx=604.5403230213659, fy=604.3848715029433, cx=967.638578208722, cy=424.0641291666773,
                   skew=0.0, k1=0.03459723604219442, k2=-0.030767694481543133,
                   p1=-0.002550466204814176, p2=0.004007846658771188)
R_RGB_KALIBR = np.array([
    [0.9999960692456313, 5.6045239682392934e-05, 0.0028032752643997677],
    [-0.00010995513721470303, 0.9998149922261677, 0.019234584207786273],
    [-0.002801678629801559, -0.019234816835876807, 0.9998110683614859],
])
T_RGB_KALIBR = np.array([-0.14982557125354823, -0.00011735766719689479, 0.006025489182246096]).reshape(3, 1)


def decode_image_msg(msg):
    h, w = msg.height, msg.width
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    enc = msg.encoding.lower()
    if enc == "mono8":
        img = buf.reshape(h, w)
    elif enc == "mono16":
        img = buf.view(np.uint16).reshape(h, w)
    elif enc == "rgb8":
        img = buf.reshape(h, w, 3)[:, :, ::-1]
    elif enc == "bgr8":
        img = buf.reshape(h, w, 3)
    elif enc == "rgba8":
        img = cv2.cvtColor(buf.reshape(h, w, 4), cv2.COLOR_RGBA2BGR)
    elif enc == "bgra8":
        img = cv2.cvtColor(buf.reshape(h, w, 4), cv2.COLOR_BGRA2BGR)
    else:
        raise ValueError(f"Unhandled image encoding: {msg.encoding}")
    return np.ascontiguousarray(img)


def ingest_bag(bag_path, cache_dir):
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    cache_dir = Path(cache_dir)
    topics_of_interest = {
        RGB_LEFT_TOPIC: "rgb_left", RGB_RIGHT_TOPIC: "rgb_right",
        THERMAL_LEFT_TOPIC: "thermal_left", THERMAL_RIGHT_TOPIC: "thermal_right",
    }
    img_dirs = {}
    for topic, slug in topics_of_interest.items():
        d = cache_dir / "images" / slug
        d.mkdir(parents=True, exist_ok=True)
        img_dirs[topic] = d

    manifest_path = cache_dir / "image_manifest.csv"
    mocap_path = cache_dir / "mocap_raw.csv"
    imu_path = cache_dir / "imu_raw.csv"

    if manifest_path.exists() and mocap_path.exists() and imu_path.exists():
        print("Cache already populated -- skipping bag read.")
        return manifest_path, mocap_path, imu_path

    typestore = get_typestore(Stores.ROS2_JAZZY)
    manifest_rows, mocap_rows, imu_rows = [], [], []
    frame_counters = {slug: 0 for slug in topics_of_interest.values()}

    with AnyReader([Path(bag_path)], default_typestore=typestore) as reader:
        wanted_topics = set(topics_of_interest) | {MOCAP_TOPIC, IMU_TOPIC}
        connections = [c for c in reader.connections if c.topic in wanted_topics]
        missing = wanted_topics - {c.topic for c in connections}
        if missing:
            print(f"WARNING: topics not found in bag: {missing}")

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            t_sec = timestamp * 1e-9
            topic = connection.topic

            if topic == MOCAP_TOPIC:
                p, q = msg.pose.position, msg.pose.orientation
                mocap_rows.append((t_sec, p.x, p.y, p.z, q.x, q.y, q.z, q.w))
            elif topic == IMU_TOPIC:
                w, a = msg.angular_velocity, msg.linear_acceleration
                imu_rows.append((t_sec, w.x, w.y, w.z, a.x, a.y, a.z))
            elif topic in topics_of_interest:
                slug = topics_of_interest[topic]
                img = decode_image_msg(msg)
                idx = frame_counters[slug]
                fname = img_dirs[topic] / f"{idx:05d}.png"
                cv2.imwrite(str(fname), img)
                manifest_rows.append((topic, slug, idx, t_sec, str(fname)))
                frame_counters[slug] += 1

    all_t = [r[0] for r in mocap_rows] + [r[0] for r in imu_rows] + [r[3] for r in manifest_rows]
    assert all_t, "No messages read from any of the requested topics -- check topic names / bag path."
    t0 = min(all_t)
    print(f"Global time origin (bag epoch t0): {t0:.6f}")

    mocap_rows = [("t_sec", "x", "y", "z", "qx", "qy", "qz", "qw")] + \
                 [(t - t0, x, y, z, qx, qy, qz, qw) for (t, x, y, z, qx, qy, qz, qw) in mocap_rows]
    imu_rows = [("t_sec", "wx", "wy", "wz", "ax", "ay", "az")] + \
               [(t - t0, wx, wy, wz, ax, ay, az) for (t, wx, wy, wz, ax, ay, az) in imu_rows]
    manifest_rows = [("topic", "slug", "frame_idx", "t_sec", "filepath")] + \
                    [(topic, slug, idx, t - t0, fp) for (topic, slug, idx, t, fp) in manifest_rows]

    for path, rows in [(manifest_path, manifest_rows), (mocap_path, mocap_rows), (imu_path, imu_rows)]:
        with open(path, "w", newline="") as f:
            csv.writer(f).writerows(rows)

    print("Ingested frame counts:", frame_counters)
    return manifest_path, mocap_path, imu_path


def build_K(i):
    return np.array([[i["fx"], i["skew"], i["cx"]], [0, i["fy"], i["cy"]], [0, 0, 1]])


def build_dist(i):
    return np.array([i["k1"], i["k2"], i.get("p1", 0.0), i.get("p2", 0.0), 0.0])


def build_calibration():
    K_rgb_left, D_rgb_left = build_K(LEFT_INTR), build_dist(LEFT_INTR)
    K_rgb_right, D_rgb_right = build_K(RIGHT_INTR), build_dist(RIGHT_INTR)
    R1_rgb, R2_rgb, P1_rgb, P2_rgb, Q_rgb, _, _ = cv2.stereoRectify(
        K_rgb_left, D_rgb_left, K_rgb_right, D_rgb_right, IMG_SIZE, R_RGB_KALIBR, T_RGB_KALIBR,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    map1x_rgb, map1y_rgb = cv2.initUndistortRectifyMap(K_rgb_left, D_rgb_left, R1_rgb, P1_rgb, IMG_SIZE, cv2.CV_32FC1)
    map2x_rgb, map2y_rgb = cv2.initUndistortRectifyMap(K_rgb_right, D_rgb_right, R2_rgb, P2_rgb, IMG_SIZE, cv2.CV_32FC1)
    baseline_rgb = float(np.linalg.norm(T_RGB_KALIBR))
    print(f"RGB stereo baseline: {baseline_rgb:.4f} m")

    return dict(
        R1_rgb=R1_rgb, P1_rgb=P1_rgb, Q_rgb=Q_rgb, map1x_rgb=map1x_rgb, map1y_rgb=map1y_rgb,
        map2x_rgb=map2x_rgb, map2y_rgb=map2y_rgb, baseline_rgb=baseline_rgb,
    )


def get_topic_frames(manifest, slug):
    return manifest[manifest["slug"] == slug].sort_values("frame_idx").reset_index(drop=True)


def unique_nearest_match(ref_times, other_times, max_dt=0.05):
    ref_times = np.asarray(ref_times)
    other_times = np.asarray(other_times)
    n_other = len(other_times)

    candidates = []
    for ref_idx, rt in enumerate(ref_times):
        idx = np.searchsorted(other_times, rt)
        for cand in (idx - 1, idx):
            if 0 <= cand < n_other:
                dt = abs(rt - other_times[cand])
                if dt <= max_dt:
                    candidates.append((dt, ref_idx, cand))
    candidates.sort(key=lambda c: c[0])

    result = np.full(len(ref_times), -1, dtype=int)
    used_other, used_ref = set(), set()
    for dt, ref_idx, other_idx in candidates:
        if ref_idx in used_ref or other_idx in used_other:
            continue
        result[ref_idx] = other_idx
        used_ref.add(ref_idx)
        used_other.add(other_idx)
    return result


def select_frames(manifest_path, subsample):
    manifest = pd.read_csv(manifest_path)
    rgb_left_frames = get_topic_frames(manifest, "rgb_left")
    rgb_right_frames = get_topic_frames(manifest, "rgb_right")

    ref_t = rgb_left_frames["t_sec"].to_numpy()
    idx_rgb_right = unique_nearest_match(ref_t, rgb_right_frames["t_sec"].to_numpy())

    valid = idx_rgb_right >= 0
    print(f"Synchronized RGB stereo pairs: {valid.sum()} / {len(ref_t)}")
    sync_indices = np.where(valid)[0]

    frame_selection = sync_indices[::subsample]
    print(f"Using {len(frame_selection)} of {len(sync_indices)} synchronized frames (SUBSAMPLE={subsample})")

    frames = []
    for i in frame_selection:
        frames.append(dict(
            t_sec=float(ref_t[i]),
            rgb_left=rgb_left_frames.loc[i, "filepath"],
            rgb_right=rgb_right_frames.loc[idx_rgb_right[i], "filepath"],
        ))
    print(f"First frame t={frames[0]['t_sec']:.3f}s, last frame t={frames[-1]['t_sec']:.3f}s")
    return frames


def run(cmd):
    print("+", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True)


def run_optional(cmd, note):
    print("+", " ".join(str(c) for c in cmd))
    result = subprocess.run([str(c) for c in cmd])
    if result.returncode != 0:
        print(f"NOTE: {note}")
    return result.returncode == 0


def check_colmap():
    if shutil.which("colmap") is None:
        raise RuntimeError(
            "`colmap` not found on PATH. See https://colmap.github.io/install.html"
        )
    result = subprocess.run(["colmap", "-h"], capture_output=True, text=True)
    first_line = result.stdout.splitlines()[0] if result.stdout else "(no output)"
    print("Found colmap:", first_line)


def supported_options(subcommand):
    result = subprocess.run(["colmap", subcommand, "-h"], capture_output=True, text=True)
    text = result.stdout + result.stderr
    return {tok.split()[0] for line in text.splitlines()
            for tok in [line.strip()] if tok.startswith("--")}


def resolve_flag(available, candidates):
    for c in candidates:
        if c in available:
            return c
    return None


def add_opt(cmd, available, candidates, value, note=None):
    flag = resolve_flag(available, candidates)
    if flag is not None:
        cmd += [flag, str(value)]
    elif note:
        print(f"NOTE: {note} (tried {candidates}, none recognized by this colmap build)")
    return cmd


def export_rectified_frames(manifest_path, calib, subsample, image_dir):
    frames = select_frames(manifest_path, subsample)
    image_dir = Path(image_dir)
    if image_dir.exists():
        shutil.rmtree(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    for k, fr in enumerate(frames):
        img = cv2.imread(fr["rgb_left"])
        rect = cv2.remap(img, calib["map1x_rgb"], calib["map1y_rgb"], cv2.INTER_LINEAR)
        cv2.imwrite(str(image_dir / f"{k:06d}.png"), rect)
    n_on_disk = len(list(image_dir.glob("*.png")))
    print(f"Exported {len(frames)} rectified frames -> {image_dir} ({n_on_disk} files on disk)")
    assert n_on_disk == len(frames), (
        f"Mismatch: exported {len(frames)} frames but {n_on_disk} files ended up on disk."
    )
    return frames


def build_feature_extractor_cmd(calib, use_gpu):
    K = calib["P1_rgb"][:3, :3]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    cmd = [
        "colmap", "feature_extractor",
        "--database_path", DATABASE_PATH,
        "--image_path", IMAGE_DIR,
        "--ImageReader.camera_model", "PINHOLE",
        "--ImageReader.camera_params", f"{fx},{fy},{cx},{cy}",
        "--ImageReader.single_camera", "1",
    ]
    available = supported_options("feature_extractor")
    cmd = add_opt(cmd, available, ["--FeatureExtraction.use_gpu", "--SiftExtraction.use_gpu"],
                  1 if use_gpu else 0, "couldn't set GPU on/off for feature extraction")
    cmd = add_opt(cmd, available, ["--FeatureExtraction.max_image_size", "--SiftExtraction.max_image_size"],
                  COLMAP_MAX_IMAGE_SIZE, "couldn't cap feature-extraction image size")
    return cmd


def build_sequential_matcher_cmd(use_gpu):
    cmd = [
        "colmap", "sequential_matcher",
        "--database_path", DATABASE_PATH,
    ]
    available = supported_options("sequential_matcher")
    cmd = add_opt(cmd, available, ["--SequentialMatching.overlap"], SEQUENTIAL_OVERLAP,
                  "couldn't set the sequential-matching temporal overlap window")
    cmd = add_opt(cmd, available, ["--FeatureMatching.use_gpu", "--SiftMatching.use_gpu"],
                  1 if use_gpu else 0, "couldn't set GPU on/off for matching")
    return cmd


def build_mapper_cmd():
    Path(SPARSE_DIR).mkdir(parents=True, exist_ok=True)
    cmd = [
        "colmap", "mapper",
        "--database_path", DATABASE_PATH,
        "--image_path", IMAGE_DIR,
        "--output_path", SPARSE_DIR,
    ]
    available = supported_options("mapper")
    cmd = add_opt(cmd, available, ["--Mapper.ba_refine_focal_length"], 0,
                  "couldn't lock focal length during bundle adjustment")
    cmd = add_opt(cmd, available, ["--Mapper.ba_refine_principal_point"], 0,
                  "couldn't lock principal point during bundle adjustment")
    cmd = add_opt(cmd, available, ["--Mapper.ba_refine_extra_params"], 0,
                  "couldn't lock distortion params during bundle adjustment")
    return cmd


def load_mocap_positions(mocap_path):
    return pd.read_csv(mocap_path)


def resample_positions_to_frames(mocap_df, frames):
    mocap_t = mocap_df["t_sec"].to_numpy()
    frame_t = np.array([fr["t_sec"] for fr in frames])
    out_of_range = (frame_t < mocap_t.min()) | (frame_t > mocap_t.max())
    if out_of_range.any():
        print(f"WARNING: {out_of_range.sum()}/{len(frame_t)} frame timestamps fall outside "
              f"mocap's covered range -- those will be clipped to the nearest edge.")
    query_t = np.clip(frame_t, mocap_t.min(), mocap_t.max())
    positions = np.stack([
        np.interp(query_t, mocap_t, mocap_df["x"].to_numpy()),
        np.interp(query_t, mocap_t, mocap_df["y"].to_numpy()),
        np.interp(query_t, mocap_t, mocap_df["z"].to_numpy()),
    ], axis=1)
    return positions


def write_ref_images_file(frames, positions, out_path):
    lines = [f"{k:06d}.png {x:.6f} {y:.6f} {z:.6f}" for k, (x, y, z) in enumerate(positions)]
    Path(out_path).write_text("\n".join(lines))
    return out_path


def build_model_aligner_cmd(ref_images_path, input_path, output_path, max_error):
    cmd = [
        "colmap", "model_aligner",
        "--input_path", input_path,
        "--output_path", output_path,
        "--ref_images_path", ref_images_path,
        "--ref_is_gps", "0",
        "--alignment_type", "custom",
    ]
    available = supported_options("model_aligner")
    cmd = add_opt(cmd, available, ["--estimate_scale"], 1,
                  "couldn't explicitly request scale estimation (should already default on)")
    cmd = add_opt(cmd, available, ["--alignment_max_error", "--robust_alignment_max_error"], max_error,
                  "couldn't set the alignment RANSAC error threshold")
    return cmd


def align_to_mocap(frames, mocap_path, sparse_model_path, aligned_output_path, max_error):
    mocap_df = load_mocap_positions(mocap_path)
    positions = resample_positions_to_frames(mocap_df, frames)
    ref_path = write_ref_images_file(frames, positions, REF_IMAGES_PATH)
    Path(aligned_output_path).mkdir(parents=True, exist_ok=True)
    cmd = build_model_aligner_cmd(ref_path, sparse_model_path, aligned_output_path, max_error)
    ok = run_optional(cmd, "model_aligner failed or this colmap build rejected the flags -- "
                            "continuing with the original arbitrary-scale sparse model")
    if ok and (Path(aligned_output_path) / "cameras.bin").exists():
        print(f"Metric-scale model written to {aligned_output_path}")
        return aligned_output_path
    print("Alignment did not produce a model -- falling back to the unaligned sparse model.")
    return sparse_model_path


def build_dense_cmds(sparse_model):
    undistort_cmd = [
        "colmap", "image_undistorter",
        "--image_path", IMAGE_DIR,
        "--input_path", sparse_model,
        "--output_path", DENSE_DIR,
    ]
    undistort_available = supported_options("image_undistorter")
    undistort_cmd = add_opt(undistort_cmd, undistort_available, ["--output_type"], "COLMAP",
                             "couldn't set undistorter output type")

    pmvs_cmd = [
        "colmap", "patch_match_stereo",
        "--workspace_path", DENSE_DIR,
    ]
    pmvs_available = supported_options("patch_match_stereo")
    pmvs_cmd = add_opt(pmvs_cmd, pmvs_available, ["--workspace_format"], "COLMAP",
                        "couldn't set PatchMatch workspace format")

    fusion_cmd = [
        "colmap", "stereo_fusion",
        "--workspace_path", DENSE_DIR,
        "--output_path", str(Path(DENSE_DIR) / "fused.ply"),
    ]
    fusion_available = supported_options("stereo_fusion")
    fusion_cmd = add_opt(fusion_cmd, fusion_available, ["--workspace_format"], "COLMAP",
                          "couldn't set fusion workspace format")

    return undistort_cmd, pmvs_cmd, fusion_cmd


def main():
    global COLMAP_MAX_IMAGE_SIZE, SEQUENTIAL_OVERLAP
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample", type=int, default=COLMAP_SUBSAMPLE,
                         help="Use every Nth synchronized frame (default: %(default)s)")
    parser.add_argument("--max-image-size", type=int, default=COLMAP_MAX_IMAGE_SIZE,
                         help="Cap SIFT/matching working resolution (default: %(default)s)")
    parser.add_argument("--overlap", type=int, default=SEQUENTIAL_OVERLAP,
                         help="Sequential-matcher temporal window size (default: %(default)s)")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU SIFT/matching")
    parser.add_argument("--sparse-only", action="store_true", help="Skip the dense MVS stage")
    parser.add_argument("--no-align", action="store_true",
                         help="Skip mocap-based metric-scale alignment (model_aligner)")
    parser.add_argument("--align-max-error", type=float, default=ALIGN_MAX_ERROR,
                         help="RANSAC error threshold (meters) for model_aligner (default: %(default)s)")
    parser.add_argument("--keep-database", action="store_true",
                         help="Don't delete an existing database.db before running")
    args = parser.parse_args()

    COLMAP_MAX_IMAGE_SIZE = args.max_image_size
    SEQUENTIAL_OVERLAP = args.overlap
    use_gpu = not args.no_gpu

    check_colmap()
    Path(COLMAP_WORKDIR).mkdir(parents=True, exist_ok=True)

    if Path(DATABASE_PATH).exists():
        if args.keep_database:
            print(f"--keep-database: reusing existing {DATABASE_PATH} as-is.")
        else:
            Path(DATABASE_PATH).unlink()
            print(f"Removed existing {DATABASE_PATH}.")

    print("=" * 60, "\n1. Ingesting rosbag\n", "=" * 60)
    manifest_path, mocap_raw_path, imu_raw_path = ingest_bag(BAG_DIR, COLMAP_WORKDIR)

    print("=" * 60, "\n2. Building calibration/rectification\n", "=" * 60)
    calib = build_calibration()

    print("=" * 60, f"\n3. Exporting rectified frames (subsample={args.subsample})\n", "=" * 60)
    frames = export_rectified_frames(manifest_path, calib, args.subsample, IMAGE_DIR)

    print("=" * 60, "\n4. Feature extraction (SIFT, known calibration)\n", "=" * 60)
    run(build_feature_extractor_cmd(calib, use_gpu))

    print("=" * 60, "\n5. Sequential matching\n", "=" * 60)
    run(build_sequential_matcher_cmd(use_gpu))

    print("=" * 60, "\n6. Incremental mapping (sparse reconstruction, fixed intrinsics)\n", "=" * 60)
    run(build_mapper_cmd())
    sparse_model = str(Path(SPARSE_DIR) / "0")
    print(f"Sparse model written under {sparse_model}")

    if not args.no_align:
        print("=" * 60, "\n7. Aligning to mocap for metric scale (model_aligner)\n", "=" * 60)
        sparse_model = align_to_mocap(frames, mocap_raw_path, sparse_model, ALIGNED_DIR, args.align_max_error)

    if args.sparse_only:
        print("\n--sparse-only: skipping dense MVS. Done.")
        print(f"Inspect the sparse model with `colmap gui`, pointing it at {sparse_model}.")
        return

    print("=" * 60, "\n8. Dense reconstruction (undistort -> PatchMatch stereo -> fusion)\n", "=" * 60)
    undistort_cmd, pmvs_cmd, fusion_cmd = build_dense_cmds(sparse_model)
    run(undistort_cmd)
    run(pmvs_cmd)
    run(fusion_cmd)

    out_ply = Path(DENSE_DIR) / "fused.ply"
    print(f"\nDone. Dense point cloud: {out_ply}")
    print(f"View with:  python3 visualize_ply.py {out_ply}")


if __name__ == "__main__":
    main()