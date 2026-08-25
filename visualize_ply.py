#!/usr/bin/env python3
"""
visualize_ply.py
================
Load a .ply point-cloud or mesh file and produce an interactive
HTML visualization using Plotly + marching-cubes (if the file is
a voxel dump) or a direct mesh/point-cloud render.

Usage
-----
    python visualize_ply.py input.ply [output.html]

If output.html is omitted, it defaults to <input_stem>_viz.html
in the same directory as the input file.
"""

import sys
import argparse
import numpy as np
from pathlib import Path



def read_ply(path: Path):
    with open(path, "rb") as f:
        raw = f.read()
    header_end = raw.find(b"end_header")
    if header_end == -1:
        raise ValueError("Not a valid PLY file (no end_header token)")
    header_bytes = raw[:header_end]
    header = header_bytes.decode("ascii", errors="replace")
    data_start = header_end + len("end_header") + 1   # skip \n

    lines = [l.strip() for l in header.splitlines()]

    fmt = "ascii"
    for l in lines:
        if l.startswith("format"):
            parts = l.split()
            fmt = parts[1]           
            break

    elements = []        # list of dicts
    cur = None
    for l in lines:
        if l.startswith("element"):
            _, name, count = l.split()
            cur = {"name": name, "count": int(count), "props": []}
            elements.append(cur)
        elif l.startswith("property") and cur is not None:
            parts = l.split()
            if parts[1] == "list":
                # e.g. property list uchar int vertex_indices
                cur["props"].append({"kind": "list",
                                     "count_type": parts[2],
                                     "val_type": parts[3],
                                     "name": parts[4]})
            else:
                cur["props"].append({"kind": "scalar",
                                     "type": parts[1],
                                     "name": parts[2]})

    def np_dtype(type_str):
        mapping = {
            "float": "f4", "float32": "f4",
            "double": "f8", "float64": "f8",
            "int": "i4", "int32": "i4",
            "uint": "u4", "uint32": "u4",
            "short": "i2", "int16": "i2",
            "ushort": "u2", "uint16": "u2",
            "uchar": "u1", "uint8": "u1",
            "char": "i1", "int8": "i1",
        }
        return mapping.get(type_str, "f4")
    
    vertices = None
    colors   = None
    faces    = None

    if fmt == "ascii":
        data_text = raw[data_start:].decode("ascii", errors="replace").splitlines()
        line_iter = iter(data_text)

        for el in elements:
            n = el["count"]
            scalar_props = [p for p in el["props"] if p["kind"] == "scalar"]
            list_props   = [p for p in el["props"] if p["kind"] == "list"]

            rows = []
            for _ in range(n):
                vals = next(line_iter).split()
                rows.append(vals)

            if el["name"] == "vertex":
                arr = np.array(rows, dtype=np.float32)
                pnames = [p["name"] for p in scalar_props]
                xi = pnames.index("x") if "x" in pnames else 0
                yi = pnames.index("y") if "y" in pnames else 1
                zi = pnames.index("z") if "z" in pnames else 2
                vertices = arr[:, [xi, yi, zi]]
                for ch, col in [("red", 0), ("green", 1), ("blue", 2)]:
                    if ch in pnames:
                        if colors is None:
                            colors = np.zeros((n, 3), dtype=np.uint8)
                        colors[:, col] = arr[:, pnames.index(ch)].astype(np.uint8)

            elif el["name"] == "face" and list_props:
                face_list = []
                for row in rows:
                    cnt = int(row[0])
                    face_list.append([int(row[i+1]) for i in range(cnt)])
                # keep only triangles
                tris = [f for f in face_list if len(f) == 3]
                if tris:
                    faces = np.array(tris, dtype=np.int32)

    else:
        endian = "<" if "little" in fmt else ">"
        offset = data_start

        for el in elements:
            n = el["count"]
            scalar_props = [p for p in el["props"] if p["kind"] == "scalar"]
            list_props   = [p for p in el["props"] if p["kind"] == "list"]

            if not list_props:
                # All scalars → read as structured array
                dt = np.dtype([(p["name"], endian + np_dtype(p["type"]))
                               for p in scalar_props])
                arr = np.frombuffer(raw, dtype=dt, count=n, offset=offset)
                offset += n * dt.itemsize

                if el["name"] == "vertex":
                    vertices = np.column_stack([arr["x"].astype(np.float32),
                                                arr["y"].astype(np.float32),
                                                arr["z"].astype(np.float32)])
                    pnames = [p["name"] for p in scalar_props]
                    has_color = all(c in pnames for c in ("red", "green", "blue"))
                    if has_color:
                        colors = np.column_stack([arr["red"],
                                                  arr["green"],
                                                  arr["blue"]]).astype(np.uint8)
            else:
                # Face element with list property
                lp = list_props[0]
                cnt_dt  = np.dtype(endian + np_dtype(lp["count_type"]))
                val_dt  = np.dtype(endian + np_dtype(lp["val_type"]))
                face_list = []
                for _ in range(n):
                    cnt = int(np.frombuffer(raw, dtype=cnt_dt, count=1,
                                            offset=offset)[0])
                    offset += cnt_dt.itemsize
                    idxs = np.frombuffer(raw, dtype=val_dt, count=cnt,
                                         offset=offset).tolist()
                    offset += cnt * val_dt.itemsize
                    face_list.append(idxs)
                tris = [f for f in face_list if len(f) == 3]
                if tris:
                    faces = np.array(tris, dtype=np.int32)

    if vertices is None:
        raise ValueError("No vertex data found in PLY file")

    print(f"  Vertices : {len(vertices):,}")
    if faces is not None:
        print(f"  Faces    : {len(faces):,}")
    if colors is not None:
        print(f"  Colors   : yes")

    return vertices, colors, faces


def build_color_array(vertices, colors):
    """
    Return a list of '#rrggbb' strings (one per vertex) for Plotly,
    or fall back to a height-based Viridis colorscale.
    """
    if colors is not None:
        hex_colors = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in colors]
        return hex_colors, None        # vertex_color, colorscale
    else:
        # Height-based color
        z = vertices[:, 2]
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-10)
        return z_norm.tolist(), "Viridis"


def save_html_mesh(vertices, colors, faces, out_path: Path, title: str):
    """Render a triangle mesh (Mesh3d)."""
    import plotly.graph_objects as go

    vertex_color, colorscale = build_color_array(vertices, colors)

    fig = go.Figure(data=[go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        vertexcolor=vertex_color if colors is not None else None,
        intensity=vertex_color if colors is None else None,
        colorscale=colorscale,
        opacity=1.0,
        name="Mesh",
    )])
    fig.update_layout(
        title=f"{title}  —  {len(faces):,} triangles, {len(vertices):,} vertices",
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(str(out_path), include_plotlyjs=True)
    print(f"Saved mesh HTML → {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


def save_html_points(vertices, colors, out_path: Path, title: str,
                     max_pts: int = 300_000):
    """Render a point cloud (Scatter3d)."""
    import plotly.graph_objects as go

    if len(vertices) > max_pts:
        idx = np.random.choice(len(vertices), max_pts, replace=False)
        vertices = vertices[idx]
        if colors is not None:
            colors = colors[idx]
        print(f"  Downsampled to {max_pts:,} points for display")

    vertex_color, colorscale = build_color_array(vertices, colors)

    marker = dict(size=1.5, opacity=0.85)
    if colors is not None:
        marker["color"] = vertex_color
    else:
        marker["color"] = vertex_color
        marker["colorscale"] = colorscale
        marker["showscale"] = True

    fig = go.Figure(data=[go.Scatter3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        mode="markers",
        marker=marker,
        name="Point Cloud",
    )])
    fig.update_layout(
        title=f"{title}  —  {len(vertices):,} points",
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(str(out_path), include_plotlyjs=True)
    print(f"Saved point-cloud HTML → {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")

def main():
    parser = argparse.ArgumentParser(
        description="Visualize a .ply file as an interactive HTML page.")
    parser.add_argument("input",  help="Input .ply file")
    parser.add_argument("output", nargs="?",
                        help="Output .html file (default: <input_stem>_viz.html)")
    parser.add_argument("--max-points", type=int, default=300_000,
                        help="Max points for point-cloud mode (default: 300 000)")
    args = parser.parse_args()

    in_path  = Path(args.input).resolve()
    out_path = Path(args.output).resolve() if args.output else \
               in_path.parent / (in_path.stem + "_viz.html")

    print(f"\nReading  : {in_path}")
    vertices, colors, faces = read_ply(in_path)

    title = in_path.stem

    if faces is not None and len(faces) > 0:
        print("Mode: mesh (Mesh3d)")
        save_html_mesh(vertices, colors, faces, out_path, title)
    else:
        print("Mode: point cloud (Scatter3d)")
        save_html_points(vertices, colors, out_path, title,
                         max_pts=args.max_points)

    print("Done.")


if __name__ == "__main__":
    main()