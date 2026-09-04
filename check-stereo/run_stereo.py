"""
In-process replacement for calling Fast-FoundationStereo's scripts/run_demo.py
per pair. Same computation, ported line-for-line (padding, scale handling,
remove_invisible, depth = K[0,0]*baseline/disp, point-cloud build/denoise/
zfar-crop) -- but:
  - the model loads ONCE and is reused across every pair, instead of once
    per subprocess launch
  - the two blocking/interactive calls run_demo.py has -- `cv2.imshow` +
    `cv2.waitKey(0)` for the disparity preview, and the live
    `o3d.visualization.Visualizer()` point-cloud window -- are removed,
    since either one would hang a batch job forever waiting for a keypress
    on a machine that may not even have a display attached.

MUST be run with the Fast-FoundationStereo repo's own Python environment
(the one with torch / open3d / imageio / omegaconf installed) -- this
imports `Utils` and `core.utils.utils` straight out of that repo, so it's
no longer a subprocess and there's no separate --python flag.

Usage:
    conda activate ffs   # or whatever the FFS env is called
    python run_stereo.py \
        --manifest /path/to/extracted/manifest.json \
        --ffs-repo /path/to/Fast-FoundationStereo \
        --ckpt /path/to/Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth \
        --out /path/to/stereo_out
"""
import argparse
import importlib
import json
import os
import shutil
import sys

import cv2
import imageio
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from rectify_utils import build_rectification, rectify_pair, write_intrinsic_file


def load_model_and_args(ffs_repo, cli_args):
    """One-time setup: mirrors run_demo.py's top section (cfg.yaml merge,
    model load, model.args patch) but only once instead of per invocation."""
    if ffs_repo not in sys.path:
        sys.path.insert(0, ffs_repo)
    from core.utils.utils import InputPadder
    import Utils as ffs_utils

    with open(f'{os.path.dirname(cli_args.ckpt)}/cfg.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    overrides = dict(
        scale=cli_args.scale, hiera=cli_args.hiera, valid_iters=cli_args.valid_iters,
        max_disp=cli_args.max_disp, zfar=cli_args.zfar, remove_invisible=cli_args.remove_invisible,
        get_pc=cli_args.get_pc, denoise_cloud=cli_args.denoise_cloud,
        denoise_nb_points=cli_args.denoise_nb_points, denoise_radius=cli_args.denoise_radius,
    )
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    run_args = OmegaConf.create(cfg)

    ffs_utils.set_logging_format()
    ffs_utils.set_seed(0)
    torch.autograd.set_grad_enabled(False)

    model = torch.load(cli_args.ckpt, map_location='cpu', weights_only=False)
    model.args.valid_iters = run_args.valid_iters
    model.args.max_disp = run_args.max_disp
    model.cuda().eval()

    return model, run_args, InputPadder, ffs_utils


def run_one_pair(model, run_args, InputPadder, ffs_utils, left_file, right_file, out_dir, K, baseline):
    """Ported from run_demo.py's __main__ body, minus the interactive display calls."""
    os.makedirs(out_dir, exist_ok=True)

    img0 = imageio.imread(left_file)
    img1 = imageio.imread(right_file)
    if img0.ndim == 2:
        img0 = np.tile(img0[..., None], (1, 1, 3))
        img1 = np.tile(img1[..., None], (1, 1, 3))
    img0 = img0[..., :3]
    img1 = img1[..., :3]

    scale = run_args.scale
    img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
    img1 = cv2.resize(img1, dsize=(img0.shape[1], img0.shape[0]))
    H, W = img0.shape[:2]
    img0_ori, img1_ori = img0.copy(), img1.copy()
    imageio.imwrite(os.path.join(out_dir, 'left.png'), img0)
    imageio.imwrite(os.path.join(out_dir, 'right.png'), img1)

    t0 = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
    t1 = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(t0.shape, divis_by=32, force_square=False)
    t0, t1 = padder.pad(t0, t1)

    with torch.amp.autocast('cuda', enabled=True, dtype=ffs_utils.AMP_DTYPE):
        if not run_args.hiera:
            disp = model.forward(t0, t1, iters=run_args.valid_iters, test_mode=True,
                                  optimize_build_volume='pytorch1')
        else:
            disp = model.run_hierachical(t0, t1, iters=run_args.valid_iters, test_mode=True, small_ratio=0.5)
    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(H, W).clip(0, None)

    vis = ffs_utils.vis_disparity(disp, min_val=None, max_val=None, cmap=None, color_map=cv2.COLORMAP_TURBO)
    vis = np.concatenate([img0_ori, img1_ori, vis], axis=1)
    disp_vis_path = os.path.join(out_dir, 'disp_vis.png')
    imageio.imwrite(disp_vis_path, vis)
    # (run_demo.py shows this with cv2.imshow + cv2.waitKey(0) here; skipped for batch use --
    #  open disp_vis_path directly to inspect a given pair)

    if run_args.remove_invisible:
        yy, xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
        us_right = xx - disp
        disp[us_right < 0] = np.inf

    result = {"disp_vis_path": disp_vis_path, "depth_path": None,
              "cloud_path": None, "cloud_denoise_path": None}

    if run_args.get_pc:
        K_scaled = K.copy()
        K_scaled[:2] *= scale
        depth = K_scaled[0, 0] * baseline / disp
        depth_path = os.path.join(out_dir, 'depth_meter.npy')
        np.save(depth_path, depth)
        result["depth_path"] = depth_path

        xyz_map = ffs_utils.depth2xyzmap(depth, K_scaled)
        pcd = ffs_utils.toOpen3dCloud(xyz_map.reshape(-1, 3), img0_ori.reshape(-1, 3))
        pts = np.asarray(pcd.points)
        keep_mask = (pts[:, 2] > 0) & (pts[:, 2] <= run_args.zfar)
        pcd = pcd.select_by_index(np.arange(len(pts))[keep_mask])

        cloud_path = os.path.join(out_dir, 'cloud.ply')
        ffs_utils.o3d.io.write_point_cloud(cloud_path, pcd)
        result["cloud_path"] = cloud_path

        if run_args.denoise_cloud:
            pcd = pcd.voxel_down_sample(voxel_size=0.001)
            _, ind = pcd.remove_radius_outlier(
                nb_points=run_args.denoise_nb_points, radius=run_args.denoise_radius
            )
            pcd = pcd.select_by_index(ind)
            cloud_denoise_path = os.path.join(out_dir, 'cloud_denoise.ply')
            ffs_utils.o3d.io.write_point_cloud(cloud_denoise_path, pcd)
            result["cloud_denoise_path"] = cloud_denoise_path
        # (run_demo.py opens a live o3d.visualization.Visualizer() here; skipped for batch use --
        #  see visualize_pointcloud.py for a saved-PNG view of any given pair instead)

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="manifest.json from extract_and_match.py")
    ap.add_argument("--index", type=int, default=None,
                     help="Run just this one pair_index from the manifest instead of the whole "
                          "batch. Still writes to <out>/pair_<index>/, same as a full run would.")
    ap.add_argument("--ffs-repo", required=True, help="Path to the Fast-FoundationStereo clone")
    ap.add_argument("--ckpt", required=True, help="Path to model_best_bp2_serialize.pth")
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib", default="calibration",
                     help="Importable module name exposing CALIB (see calibration.py)")
    ap.add_argument("--scale", type=float, default=1)
    ap.add_argument("--hiera", type=int, default=0, choices=[0, 1])
    ap.add_argument("--valid-iters", type=int, default=8)
    ap.add_argument("--max-disp", type=int, default=192)
    ap.add_argument("--zfar", type=float, default=10.0, help="Max depth (m) kept in point cloud")
    ap.add_argument("--remove-invisible", type=int, default=0, choices=[0, 1])
    ap.add_argument("--get-pc", type=int, default=1, choices=[0, 1])
    ap.add_argument("--denoise-cloud", type=int, default=1, choices=[0, 1])
    ap.add_argument("--denoise-nb-points", type=int, default=30)
    ap.add_argument("--denoise-radius", type=float, default=0.03)
    args = ap.parse_args()

    calib_mod = importlib.import_module(args.calib)
    rect = build_rectification(calib_mod.CALIB)
    print(f"Rectified baseline: {rect['baseline_m']:.5f} m, K_rect=\n{rect['K_rect']}")

    with open(args.manifest) as f:
        manifest = json.load(f)

    if args.index is not None:
        filtered = [e for e in manifest if e["pair_index"] == args.index]
        if not filtered:
            raise SystemExit(f"pair_index {args.index} not in manifest. "
                              f"Available: {[e['pair_index'] for e in manifest]}")
        manifest = filtered

    rect_dir = os.path.join(args.out, "rectified")
    os.makedirs(rect_dir, exist_ok=True)
    write_intrinsic_file(os.path.join(args.out, "intrinsics.txt"), rect["K_rect"], rect["baseline_m"])

    model, run_args, InputPadder, ffs_utils = load_model_and_args(args.ffs_repo, args)

    results = []
    for entry in manifest:
        pi = entry["pair_index"]
        img1 = cv2.imread(entry["rgb1_path"], cv2.IMREAD_UNCHANGED)
        img2 = cv2.imread(entry["rgb2_path"], cv2.IMREAD_UNCHANGED)
        r1, r2 = rectify_pair(img1, img2, rect)
        left_img, right_img = (r1, r2) if rect["left_is"] == "rgb1" else (r2, r1)

        pair_rect_dir = os.path.join(rect_dir, f"pair_{pi:06d}")
        os.makedirs(pair_rect_dir, exist_ok=True)
        left_path = os.path.join(pair_rect_dir, "left.png")
        right_path = os.path.join(pair_rect_dir, "right.png")
        cv2.imwrite(left_path, left_img)
        cv2.imwrite(right_path, right_img)

        pair_out_dir = os.path.join(args.out, f"pair_{pi:06d}")
        shutil.rmtree(pair_out_dir, ignore_errors=True)

        print(f"[pair {pi}] running stereo...")
        result = run_one_pair(
            model, run_args, InputPadder, ffs_utils,
            left_path, right_path, pair_out_dir,
            K=rect["K_rect"], baseline=rect["baseline_m"],
        )
        result.update({
            "pair_index": pi,
            "left_path": left_path, "right_path": right_path, "out_dir": pair_out_dir,
            "ply_path": result.get("cloud_denoise_path") or result.get("cloud_path"),
            "t1": entry["t1"], "t2": entry["t2"], "dt": entry["dt"],
        })
        results.append(result)

    results_path = os.path.join(args.out, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {len(results)} results -> {results_path}")


if __name__ == "__main__":
    main()
