"""
inspect_episode.py  –  Quick HDF5 episode inspector (new format).

Usage:
    python inspect_episode.py                        # latest episode in logs/
    python inspect_episode.py logs/003               # episode folder (finds episode.hdf5)
    python inspect_episode.py logs/003/episode.hdf5  # explicit file
    python inspect_episode.py logs/003 --no-plot     # skip matplotlib windows
"""

import sys
import os
import glob
import math
import argparse

import h5py
import numpy as np


# ── Helpers ────────────────────────────────────────────────────────────────────

def find_file(arg=None):
    """Resolve path → HDF5 file.  Accepts a folder or a direct .hdf5 path."""
    if arg is None:
        # Latest numeric episode folder under logs/
        candidates = sorted(glob.glob("logs/[0-9][0-9][0-9]"))
        if not candidates:
            sys.exit("No episode folders found under logs/. Pass a path explicitly.")
        arg = candidates[-1]

    if os.path.isfile(arg):
        return arg

    if os.path.isdir(arg):
        for name in ("episode.hdf5", "images.hdf5"):
            p = os.path.join(arg, name)
            if os.path.exists(p):
                return p
        sys.exit(f"No episode.hdf5 (or images.hdf5) found in {arg}")

    sys.exit(f"Path not found: {arg}")


def _sep(title=""):
    w = 60
    if title:
        pad = (w - len(title) - 2) // 2
        print("─" * pad + f" {title} " + "─" * (w - pad - len(title) - 2))
    else:
        print("─" * w)


def fmt_arr(arr, fmt=".4f"):
    return "[" + "  ".join(f"{v:{fmt}}" for v in arr) + "]"


# ── Structure dump ─────────────────────────────────────────────────────────────

def print_structure(f):
    _sep("Structure")
    def _visit(name, obj):
        depth  = name.count("/")
        indent = "  " * depth
        short  = name.split("/")[-1]
        if isinstance(obj, h5py.Dataset):
            print(f"{indent}{short:35s}  {str(obj.shape):20s}  {obj.dtype}")
        else:
            print(f"{indent}{short}/")
    f.visititems(_visit)
    print()


# ── Attributes ────────────────────────────────────────────────────────────────

def print_attrs(f):
    if not f.attrs:
        return
    _sep("Episode attributes")
    for k in f.attrs:
        try:
            print(f"  {k:<22s} {f.attrs[k]}")
        except Exception as e:
            print(f"  {k:<22s} <unreadable: {e}>")
    print()


# ── Scene ─────────────────────────────────────────────────────────────────────

def print_scene(f):
    if "scene" not in f:
        return
    sg = f["scene"]
    _sep("Scene — spawn config (episode constants)")

    n_objects = sum(1 for i in range(4) if f"obj{i}_pose" in sg)
    print(f"  Active objects : {n_objects}")
    print()

    for i in range(4):
        pose_key = f"obj{i}_pose"
        if pose_key not in sg:
            continue
        pose  = sg[pose_key][0]            # first frame (constant)
        x, y, z = pose[0], pose[1], pose[2]
        yaw_key   = f"obj{i}_spawn_yaw"
        scale_key = f"obj{i}_scale"
        yaw_deg = math.degrees(float(sg[yaw_key][0])) if yaw_key in sg else None
        scale   = float(sg[scale_key][0])  if scale_key in sg else None
        yaw_str   = f"{yaw_deg:6.1f}°" if yaw_deg is not None else "  n/a  "
        scale_str = f"{scale:.3f}"    if scale   is not None else " n/a "
        print(f"  obj{i}  pos=({x:.3f}, {y:.3f}, {z:.3f})  "
              f"spawn_yaw={yaw_str}  scale={scale_str}")

    if "lighting" in sg:
        lg = sg["lighting"]
        print()
        print("  Lighting")
        for name in ("ambient", "diffuse", "specular", "key_dir", "key_diffuse"):
            if name in lg:
                print(f"    {name:<12s} {fmt_arr(lg[name][:])}")
    print()


# ── Telemetry stats ────────────────────────────────────────────────────────────

def print_telemetry(f):
    obs = f.get("observations")
    if obs is None:
        return
    _sep("Telemetry")

    ts = obs["timestamp_ns"][:] if "timestamp_ns" in obs else None
    if ts is not None:
        T = len(ts)
        dur = (ts[-1] - ts[0]) / 1e9
        fps = (T - 1) / dur if dur > 0 else 0
        gaps = np.diff(ts) / 1e6
        print(f"  Grid frames  : {T}")
        print(f"  Duration     : {dur:.2f} s  ({fps:.1f} Hz)")
        print(f"  Frame gap    : {gaps.mean():.1f} ms avg  "
              f"{gaps.min():.1f} min  {gaps.max():.1f} max ms")
        print()

    for arm in ("arm_left", "arm_right"):
        if arm not in obs:
            continue
        ag = obs[arm]
        print(f"  {arm}")
        if "q" in ag:
            q = ag["q"][:]
            print(f"    q range      : [{q.min():.3f}, {q.max():.3f}]")
        if "gripper_width" in ag:
            gw = ag["gripper_width"][:]
            print(f"    gripper      : [{gw.min():.4f}, {gw.max():.4f}] m")
        print()

    if "head" in obs:
        hg = obs["head"]
        if "q" in hg:
            q = hg["q"][:]
            print(f"  head  q range : [{q.min():.3f}, {q.max():.3f}]")
            print()


# ── Image stats + plots ────────────────────────────────────────────────────────

def print_images(f, path, do_plot):
    obs = f.get("observations")
    if obs is None or "images" not in obs:
        return

    ts  = obs["timestamp_ns"][:] if "timestamp_ns" in obs else None
    fid = obs["frame_id"][:]     if "frame_id"     in obs else None

    img_group = obs["images"]
    cameras   = list(img_group.keys())
    if not cameras:
        return

    _sep("Images")

    for camera in cameras:
        imgs = img_group[camera][:]
        T, H, W, C = imgs.shape

        print(f"  Camera: {camera}")
        print(f"    Frames       : {T}  ({W}×{H}  {C}ch  {imgs.dtype})")
        print(f"    RAM          : {imgs.nbytes / 1e6:.1f} MB")

        if ts is not None and T > 1:
            dur = (ts[-1] - ts[0]) / 1e9
            fps = (T - 1) / dur if dur > 0 else 0
            print(f"    Duration     : {dur:.2f} s  ({fps:.1f} fps)")

        if fid is not None:
            dropped = int(fid[-1] - fid[0]) - (T - 1)
            print(f"    Dropped      : {dropped} frames")

        print(f"    Pixel range  : [{imgs.min()}, {imgs.max()}]")
        for ch, name in enumerate(["R", "G", "B"]):
            d = imgs[:, :, :, ch]
            print(f"    {name}            : mean={d.mean():.1f}  std={d.std():.1f}")

        # Save middle frame
        mid      = T // 2
        png_path = os.path.splitext(path)[0] + f"_{camera}_frame{mid}.png"
        try:
            from PIL import Image as PILImage
            PILImage.fromarray(imgs[mid]).save(png_path)
        except ImportError:
            import matplotlib
            matplotlib.image.imsave(png_path, imgs[mid])
        print(f"    Saved frame {mid} → {png_path}")
        print()

        if not do_plot:
            continue

        import matplotlib.pyplot as plt

        n       = min(9, T)
        indices = np.sort(np.random.choice(T, size=n, replace=False))
        cols    = 3
        rows    = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3))
        axes = np.array(axes).reshape(-1)

        for ax_idx, fi in enumerate(indices):
            axes[ax_idx].imshow(imgs[fi])
            label = f"frame {fi}"
            if ts is not None:
                label += f"\nt={(ts[fi] - ts[0]) / 1e9:.2f}s"
            axes[ax_idx].set_title(label, fontsize=8)
            axes[ax_idx].axis("off")
        for ax in axes[n:]:
            ax.set_visible(False)

        ep_id = os.path.basename(os.path.dirname(path))
        fig.suptitle(f"Episode {ep_id}  –  {camera}  ({T} frames)", fontsize=11)
        plt.tight_layout()
        plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default=None,
                    help="episode folder or .hdf5 file (default: latest under logs/)")
    ap.add_argument("--no-plot", action="store_true",
                    help="skip matplotlib image windows (still saves PNG)")
    args = ap.parse_args()

    path = find_file(args.path)
    print(f"File : {path}\n")

    with h5py.File(path, "r") as f:
        print_structure(f)
        print_attrs(f)
        print_scene(f)
        print_telemetry(f)
        print_images(f, path, do_plot=not args.no_plot)


if __name__ == "__main__":
    main()
