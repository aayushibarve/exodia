#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

BAG_DIR = "/home/barve/exodia/mocap_thermal_rgb5"
CACHE_DIR = "/home/barve/exodia/vo_cache3"

RGB_LEFT_TOPIC = "/rgb1/image_raw"
RGB_RIGHT_TOPIC = "/rgb2/image_raw"
THERMAL_LEFT_TOPIC = "/thermal1/image_gray"
THERMAL_RIGHT_TOPIC = "/thermal2/image_gray"
MOCAP_TOPIC = "/macortex_bridge/exodia/pose"
IMU_TOPIC = "/imu/data_raw"

COLMAP_SUBSAMPLE = 5
THERMAL_MATCH_TOL = 0.05

CAMERA_MODEL = "OPENCV"
RGB1_CAMERA_PARAMS = "595.607,594.714,941.101,458.698,-0.00257185,-0.0145025,0,0"
RGB2_CAMERA_PARAMS = "598.517,597.363,970.896,424.176,0.00464308,-0.0195245,0,0"
MAX_NUM_FEATURES = 8192

RGB_PAIR_RVEC_RAW = np.array([0.0171, -0.0259, 3.1381])
RGB_PAIR_TVEC_RAW = np.array([-0.1482, 0.0009, 0.0015]).reshape(3, 1)


def undo_normalize_es(R, t, angle_threshold_deg=90.0):
    rvec, _ = cv2.Rodrigues(R)
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    if np.degrees(np.linalg.norm(rvec)) < angle_threshold_deg:
        return np.ascontiguousarray(R), np.ascontiguousarray(t)
    Rz_pi, _ = cv2.Rodrigues(np.array([0.0, 0.0, np.pi]))
    return np.ascontiguousarray(R @ Rz_pi), np.ascontiguousarray(-t)


def invert_transform(R, t):
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    return np.ascontiguousarray(R.T), np.ascontiguousarray(-R.T @ t)


def R_to_quat_wxyz(R):
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw, qx, qy, qz = 0.25 * S, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw, qx, qy, qz = (R[2, 1] - R[1, 2]) / S, 0.25 * S, (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw, qx, qy, qz = (R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S, 0.25 * S, (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw, qx, qy, qz = (R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S, 0.25 * S
    return [qw, qx, qy, qz]


def compute_rgb2_from_rgb1():
    R_raw, _ = cv2.Rodrigues(RGB_PAIR_RVEC_RAW)
    R_pose, t_pose = undo_normalize_es(R_raw, RGB_PAIR_TVEC_RAW)
    R_rgb2_from_rig, t_rgb2_from_rig = invert_transform(R_pose, t_pose)
    return R_rgb2_from_rig, t_rgb2_from_rig


def run(cmd):
    print("+", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True)


def check_colmap():
    if shutil.which("colmap") is None:
        raise RuntimeError("`colmap` not found on PATH. See https://colmap.github.io/install.html")
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


def run_optional(cmd, note):
    print("+", " ".join(str(c) for c in cmd))
    result = subprocess.run([str(c) for c in cmd])
    if result.returncode != 0:
        print(f"NOTE: {note}")
    return result.returncode == 0


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


def get_topic_frames(manifest, slug):
    return manifest[manifest["slug"] == slug].sort_values("frame_idx").reset_index(drop=True)


def unique_nearest_match(ref_times, other_times, max_dt):
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


def select_and_match_frames(manifest_path, subsample, thermal_tol):
    manifest = pd.read_csv(manifest_path)
    rgb_left = get_topic_frames(manifest, "rgb_left")
    rgb_right = get_topic_frames(manifest, "rgb_right")
    thermal_left = get_topic_frames(manifest, "thermal_left")
    thermal_right = get_topic_frames(manifest, "thermal_right")

    ref_t = rgb_left["t_sec"].to_numpy()
    idx_right = unique_nearest_match(ref_t, rgb_right["t_sec"].to_numpy(), max_dt=0.05)
    valid = idx_right >= 0
    print(f"Synchronized rgb_left/rgb_right pairs: {valid.sum()} / {len(ref_t)}")
    sync_indices = np.where(valid)[0]
    selection = sync_indices[::subsample]
    print(f"Using {len(selection)} of {len(sync_indices)} synchronized frames (SUBSAMPLE={subsample})")

    sel_t = ref_t[selection]
    idx_th_left = unique_nearest_match(sel_t, thermal_left["t_sec"].to_numpy(), max_dt=thermal_tol)
    idx_th_right = unique_nearest_match(sel_t, thermal_right["t_sec"].to_numpy(), max_dt=thermal_tol)
    n_th_left, n_th_right = (idx_th_left >= 0).sum(), (idx_th_right >= 0).sum()
    print(f"Thermal matches within {thermal_tol}s: left {n_th_left}/{len(selection)}, "
          f"right {n_th_right}/{len(selection)}")

    frames = []
    for j, i in enumerate(selection):
        frames.append(dict(
            t_sec=float(ref_t[i]),
            rgb_left=rgb_left.loc[i, "filepath"],
            rgb_right=rgb_right.loc[idx_right[i], "filepath"],
            thermal_left=thermal_left.loc[idx_th_left[j], "filepath"] if idx_th_left[j] >= 0 else None,
            thermal_right=thermal_right.loc[idx_th_right[j], "filepath"] if idx_th_right[j] >= 0 else None,
        ))
    return frames


def export_dataset(frames, images_dir, frame_dataset_dir):
    images_dir, frame_dataset_dir = Path(images_dir), Path(frame_dataset_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    rgb1_names, rgb2_names = [], []
    n_thermal = 0
    for k, fr in enumerate(frames):
        frame_id = f"{k:06d}"
        rgb1_name, rgb2_name = f"rgb1_{frame_id}.png", f"rgb2_{frame_id}.png"
        shutil.copy2(fr["rgb_left"], images_dir / rgb1_name)
        shutil.copy2(fr["rgb_right"], images_dir / rgb2_name)
        rgb1_names.append(rgb1_name)
        rgb2_names.append(rgb2_name)

        fdir = frame_dataset_dir / frame_id
        fdir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fr["rgb_left"], fdir / "rgb1.png")
        shutil.copy2(fr["rgb_right"], fdir / "rgb2.png")
        if fr["thermal_left"] is not None:
            shutil.copy2(fr["thermal_left"], fdir / "thermal1_gray.png")
            n_thermal += 1
        if fr["thermal_right"] is not None:
            shutil.copy2(fr["thermal_right"], fdir / "thermal2_gray.png")

    print(f"Exported {len(frames)} frames -> {images_dir} (COLMAP input) and "
          f"{frame_dataset_dir} (per-frame folders, {n_thermal} with thermal coverage)")
    return rgb1_names, rgb2_names


def write_image_list(names, out_path):
    Path(out_path).write_text("\n".join(names))
    return out_path


def build_feature_extractor_cmd(database_path, images_dir, image_list_path, camera_params, use_gpu):
    cmd = [
        "colmap", "feature_extractor",
        "--database_path", database_path,
        "--image_path", images_dir,
        "--image_list_path", str(image_list_path),
        "--camera_mode", "1",
        "--ImageReader.camera_model", CAMERA_MODEL,
        "--ImageReader.camera_params", camera_params,
        "--SiftExtraction.max_num_features", str(MAX_NUM_FEATURES),
        "--SiftExtraction.estimate_affine_shape", "1",
        "--SiftExtraction.domain_size_pooling", "1",
    ]
    available = supported_options("feature_extractor")
    cmd = add_opt(cmd, available, ["--FeatureExtraction.use_gpu", "--SiftExtraction.use_gpu"],
                  1 if use_gpu else 0, "couldn't set GPU on/off for feature extraction")
    return cmd


def build_rig_config(out_path):
    R_rgb2_from_rig, t_rgb2_from_rig = compute_rgb2_from_rgb1()
    qvec_wxyz = R_to_quat_wxyz(R_rgb2_from_rig)
    config = [{
        "cameras": [
            {"image_prefix": "rgb1_", "ref_sensor": True},
            {"image_prefix": "rgb2_",
             "cam_from_rig_rotation": qvec_wxyz,
             "cam_from_rig_translation": t_rgb2_from_rig.flatten().tolist()},
        ],
    }]
    Path(out_path).write_text(json.dumps(config, indent=2))
    return out_path


def build_exhaustive_matcher_cmd(database_path, use_gpu, guided_matching, max_ratio,
                                  min_num_inliers, min_inlier_ratio):
    cmd = ["colmap", "exhaustive_matcher", "--database_path", database_path]
    available = supported_options("exhaustive_matcher")
    cmd = add_opt(cmd, available, ["--FeatureMatching.use_gpu", "--SiftMatching.use_gpu"],
                  1 if use_gpu else 0, "couldn't set GPU on/off for matching")
    cmd = add_opt(cmd, available, ["--FeatureMatching.guided_matching"], 1 if guided_matching else 0,
                  "couldn't enable guided matching")
    cmd = add_opt(cmd, available, ["--SiftMatching.max_ratio"], max_ratio,
                  "couldn't set SIFT matching ratio threshold")
    cmd = add_opt(cmd, available, ["--TwoViewGeometry.min_num_inliers"], min_num_inliers,
                  "couldn't set two-view-geometry minimum inlier count")
    cmd = add_opt(cmd, available, ["--TwoViewGeometry.min_inlier_ratio"], min_inlier_ratio,
                  "couldn't set two-view-geometry minimum inlier ratio")
    return cmd


def build_mapper_cmd(database_path, images_dir, sparse_dir):
    Path(sparse_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        "colmap", "mapper",
        "--database_path", database_path,
        "--image_path", images_dir,
        "--output_path", sparse_dir,
    ]
    available = supported_options("mapper")
    cmd = add_opt(cmd, available, ["--Mapper.init_min_tri_angle"], 2,
                  "couldn't lower the initial-pair triangulation-angle requirement")
    cmd = add_opt(cmd, available, ["--Mapper.abs_pose_min_num_inliers"], 15,
                  "couldn't lower the absolute-pose registration inlier requirement")
    cmd = add_opt(cmd, available, ["--Mapper.ba_refine_focal_length"], 0,
                  "couldn't lock focal length during bundle adjustment")
    cmd = add_opt(cmd, available, ["--Mapper.ba_refine_principal_point"], 0,
                  "couldn't lock principal point during bundle adjustment")
    cmd = add_opt(cmd, available, ["--Mapper.ba_refine_extra_params"], 0,
                  "couldn't lock distortion params during bundle adjustment")
    cmd = add_opt(cmd, available, ["--Mapper.ba_refine_sensor_from_rig"], 1,
                  "couldn't allow bundle adjustment to refine the rig extrinsic")
    return cmd


def build_dense_cmds(images_dir, sparse_model, dense_dir):
    undistort_cmd = [
        "colmap", "image_undistorter",
        "--image_path", images_dir,
        "--input_path", sparse_model,
        "--output_path", dense_dir,
    ]
    undistort_available = supported_options("image_undistorter")
    undistort_cmd = add_opt(undistort_cmd, undistort_available, ["--output_type"], "COLMAP",
                             "couldn't set undistorter output type")

    pmvs_cmd = ["colmap", "patch_match_stereo", "--workspace_path", dense_dir]
    pmvs_available = supported_options("patch_match_stereo")
    pmvs_cmd = add_opt(pmvs_cmd, pmvs_available, ["--workspace_format"], "COLMAP",
                        "couldn't set PatchMatch workspace format")
    pmvs_cmd = add_opt(pmvs_cmd, pmvs_available, ["--PatchMatchStereo.geom_consistency"], "true",
                        "couldn't enable geometric-consistency filtering")

    fusion_cmd = [
        "colmap", "stereo_fusion",
        "--workspace_path", dense_dir,
        "--input_type", "geometric",
        "--output_path", str(Path(dense_dir) / "fused.ply"),
    ]
    fusion_available = supported_options("stereo_fusion")
    fusion_cmd = add_opt(fusion_cmd, fusion_available, ["--StereoFusion.min_num_pixels"], 2,
                          "couldn't lower fusion's minimum-view-agreement requirement")
    fusion_cmd = add_opt(fusion_cmd, fusion_available, ["--StereoFusion.max_reproj_error"], 4,
                          "couldn't loosen fusion's reprojection-error tolerance")
    fusion_cmd = add_opt(fusion_cmd, fusion_available, ["--StereoFusion.max_depth_error"], 0.1,
                          "couldn't loosen fusion's depth-error tolerance")
    fusion_cmd = add_opt(fusion_cmd, fusion_available, ["--StereoFusion.max_normal_error"], 15,
                          "couldn't loosen fusion's normal-error tolerance")

    return undistort_cmd, pmvs_cmd, fusion_cmd


def verify_metric_scale(sparse_model, tol_frac=0.03):
    try:
        import pycolmap
    except ImportError:
        print("NOTE: pycolmap not installed -- skipping scale verification (pip install pycolmap)")
        return

    R_rgb2_from_rig, t_rgb2_from_rig = compute_rgb2_from_rgb1()
    known_baseline_m = float(np.linalg.norm(t_rgb2_from_rig))

    recon = pycolmap.Reconstruction(sparse_model)
    by_name = {img.name: img for img in recon.images.values() if img.has_pose}
    rgb1_frames = sorted(n[len("rgb1_"):-4] for n in by_name if n.startswith("rgb1_"))
    if not rgb1_frames:
        print("NOTE: no registered rgb1 frames -- skipping scale verification.")
        return

    frame_id = rgb1_frames[0]
    name1, name2 = f"rgb1_{frame_id}.png", f"rgb2_{frame_id}.png"
    if name1 not in by_name or name2 not in by_name:
        print(f"NOTE: frame {frame_id} doesn't have both rgb1 and rgb2 registered -- "
              f"skipping scale verification.")
        return

    C1 = by_name[name1].projection_center()
    C2 = by_name[name2].projection_center()
    colmap_baseline = float(np.linalg.norm(C1 - C2))
    ratio = colmap_baseline / known_baseline_m
    print(f"Known physical rgb1<->rgb2 baseline: {known_baseline_m:.4f} m")
    print(f"COLMAP-reconstructed baseline (frame {frame_id}): {colmap_baseline:.4f} m")
    print(f"Ratio (should be ~1.0 if the rig constraint took effect): {ratio:.4f}")
    if abs(ratio - 1.0) > tol_frac:
        print(f"WARNING: reconstructed baseline is {abs(ratio - 1) * 100:.1f}% off the "
              f"calibrated value -- check the rig configuration took effect.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True,
                         help="Output workspace (images/, frames/, database.db, sparse/, dense/)")
    parser.add_argument("--subsample", type=int, default=COLMAP_SUBSAMPLE,
                         help="Use every Nth synchronized rgb_left/rgb_right pair (default: %(default)s)")
    parser.add_argument("--thermal-tol", type=float, default=THERMAL_MATCH_TOL,
                         help="Max timestamp gap (s) to accept a thermal frame match (default: %(default)s)")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--no-guided-matching", action="store_true")
    parser.add_argument("--max-ratio", type=float, default=0.85)
    parser.add_argument("--min-num-inliers", type=int, default=10)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.15)
    parser.add_argument("--sparse-only", action="store_true")
    parser.add_argument("--keep-database", action="store_true")
    args = parser.parse_args()

    use_gpu = not args.no_gpu
    work_dir = Path(args.work_dir)
    images_dir = str(work_dir / "images")
    frame_dataset_dir = str(work_dir / "frames")
    database_path = str(work_dir / "database.db")
    sparse_dir = str(work_dir / "sparse")
    dense_dir = str(work_dir / "dense")
    rig_config_path = str(work_dir / "rig_config.json")

    check_colmap()
    work_dir.mkdir(parents=True, exist_ok=True)

    if Path(database_path).exists():
        if args.keep_database:
            print(f"--keep-database: reusing existing {database_path} as-is.")
        else:
            Path(database_path).unlink()
            print(f"Removed existing {database_path}.")

    print("=" * 60, "\n1. Ingesting rosbag\n", "=" * 60)
    manifest_path, mocap_raw_path, imu_raw_path = ingest_bag(BAG_DIR, CACHE_DIR)

    print("=" * 60, "\n2. Selecting and timestamp-matching frames\n", "=" * 60)
    frames = select_and_match_frames(manifest_path, args.subsample, args.thermal_tol)

    print("=" * 60, "\n3. Exporting COLMAP images + per-frame dataset folders\n", "=" * 60)
    rgb1_names, rgb2_names = export_dataset(frames, images_dir, frame_dataset_dir)
    rgb1_list = write_image_list(rgb1_names, str(work_dir / "rgb1_list.txt"))
    rgb2_list = write_image_list(rgb2_names, str(work_dir / "rgb2_list.txt"))

    print("=" * 60, "\n4. Feature extraction (rgb1, then rgb2 -- known OPENCV calibration)\n", "=" * 60)
    run(build_feature_extractor_cmd(database_path, images_dir, rgb1_list, RGB1_CAMERA_PARAMS, use_gpu))
    run(build_feature_extractor_cmd(database_path, images_dir, rgb2_list, RGB2_CAMERA_PARAMS, use_gpu))

    print("=" * 60, "\n5. Configuring stereo rig (raw-frame extrinsic)\n", "=" * 60)
    run_rig_config = build_rig_config(rig_config_path)
    run_optional(
        ["colmap", "rig_configurator", "--database_path", database_path, "--rig_config_path", run_rig_config],
        "rig_configurator rejected the rig config -- continuing without rig constraints",
    )

    print("=" * 60, "\n6. Exhaustive matching\n", "=" * 60)
    run(build_exhaustive_matcher_cmd(database_path, use_gpu, not args.no_guided_matching,
                                      args.max_ratio, args.min_num_inliers, args.min_inlier_ratio))

    print("=" * 60, "\n7. Incremental mapping (rig-constrained bundle adjustment)\n", "=" * 60)
    run(build_mapper_cmd(database_path, images_dir, sparse_dir))
    sparse_model = str(Path(sparse_dir) / "0")
    print(f"Sparse model written under {sparse_model}")

    print("=" * 60, "\n8. Exporting TEXT format (cameras.txt/images.txt)\n", "=" * 60)
    run(["colmap", "model_converter", "--input_path", sparse_model,
         "--output_path", sparse_model, "--output_type", "TXT"])

    print("=" * 60, "\n9. Verifying metric scale against the known rig baseline\n", "=" * 60)
    verify_metric_scale(sparse_model)

    if args.sparse_only:
        print("\n--sparse-only: skipping dense MVS. Done.")
        print(f"DATASET_ROOT for thermal-wrap2.py:     {frame_dataset_dir}")
        print(f"SPARSE_MODEL_DIR for thermal-wrap2.py: {sparse_model}")
        return

    print("=" * 60, "\n10. Dense reconstruction (undistort -> PatchMatch stereo -> fusion)\n", "=" * 60)
    undistort_cmd, pmvs_cmd, fusion_cmd = build_dense_cmds(images_dir, sparse_model, dense_dir)
    run(undistort_cmd)
    run(pmvs_cmd)
    run(fusion_cmd)

    out_ply = Path(dense_dir) / "fused.ply"
    print(f"\nDone.")
    print(f"DATASET_ROOT for thermal-wrap2.py:     {frame_dataset_dir}")
    print(f"SPARSE_MODEL_DIR for thermal-wrap2.py: {sparse_model}")
    print(f"FUSED_PLY_PATH for thermal-wrap2.py:   {out_ply}")


if __name__ == "__main__":
    main()
