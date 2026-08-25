#!/usr/bin/env python3
"""
view_toggle.py
====================
Load the fused_toggle.npz bundle (from thermal_pointcloud_projection.py) and
produce an interactive HTML page with a button to switch between RGB and
thermal point-cloud coloring -- same points, same positions, colors swap.

Usage
-----
    python view_toggle.py [bundle.npz] [output.html]

Defaults to thermal_wraparound_outputs/fused_toggle.npz and
fused_toggle_viz.html alongside it.
"""

import sys
from pathlib import Path
import numpy as np
import plotly.graph_objects as go

DEFAULT_BUNDLE = "./thermal_wraparound_outputs2/fused_toggle.npz"
MAX_POINTS = 300_000  # same default as visualize_ply.py, for browser performance


def to_hex(colors_float):
    """(N,3) float array in 0-1 -> list of '#rrggbb' strings."""
    rgb_u8 = np.clip(colors_float * 255, 0, 255).astype(np.uint8)
    return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in rgb_u8]


def main():
    bundle_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_BUNDLE)
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else \
        bundle_path.parent / (bundle_path.stem + "_viz.html")

    print(f"Reading: {bundle_path}")
    data = np.load(bundle_path)
    points = data["points"]
    rgb_colors = data["rgb_colors"]
    thermal_colors = data["thermal_colors"]
    covered = data["covered"]

    n_points = points.shape[0]
    print(f"  {n_points:,} points, {covered.sum():,} with thermal coverage "
          f"({100 * covered.sum() / n_points:.1f}%)")

    if n_points > MAX_POINTS:
        idx = np.random.choice(n_points, MAX_POINTS, replace=False)
        points = points[idx]
        rgb_colors = rgb_colors[idx]
        thermal_colors = thermal_colors[idx]
        print(f"  Downsampled to {MAX_POINTS:,} points for display")

    rgb_hex = to_hex(rgb_colors)
    thermal_hex = to_hex(thermal_colors)

    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode="markers",
        marker=dict(size=1.5, opacity=0.9, color=rgb_hex),
        name="Point Cloud",
    )])

    fig.update_layout(
        title=f"{bundle_path.stem}  —  {n_points:,} points  (RGB mode)",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
        updatemenus=[dict(
            type="buttons",
            direction="left",
            x=0.02, y=0.98, xanchor="left", yanchor="top",
            buttons=[
                dict(
                    label="RGB",
                    method="update",
                    args=[
                        {"marker.color": [rgb_hex]},
                        {"title": f"{bundle_path.stem}  —  {n_points:,} points  (RGB mode)"},
                    ],
                ),
                dict(
                    label="Thermal",
                    method="update",
                    args=[
                        {"marker.color": [thermal_hex]},
                        {"title": f"{bundle_path.stem}  —  {n_points:,} points  (THERMAL mode, "
                                  f"gray = no coverage)"},
                    ],
                ),
            ],
        )],
    )

    fig.write_html(str(out_path), include_plotlyjs=True)
    print(f"Saved -> {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    print("Open it in a browser; use the RGB / Thermal buttons top-left to toggle.")


if __name__ == "__main__":
    main()