"""
inspect_episode.py  –  Quick HDF5 episode inspector.

Usage:
    python inspect_episode.py                        # latest episode in logs/
    python inspect_episode.py logs/003/images.hdf5   # specific file
"""

import sys
import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt


# ── Locate file ───────────────────────────────────────────────────────────────

def find_file(arg=None):
    if arg:
        return arg
    candidates = sorted(glob.glob("logs/*/images.hdf5"))
    if not candidates:
        sys.exit("No images.hdf5 found under logs/. Pass a path explicitly.")
    return candidates[-1]   # most recent episode folder (alphabetical = numeric)


path = find_file(sys.argv[1] if len(sys.argv) > 1 else None)
print(f"File: {path}\n")


# ── Read ──────────────────────────────────────────────────────────────────────

with h5py.File(path, "r") as f:

    # ── Structure ─────────────────────────────────────────────────────────────
    print("─" * 50)
    print("Structure")
    print("─" * 50)
    def _print(name, obj):
        indent = "  " * name.count("/")
        if isinstance(obj, h5py.Dataset):
            print(f"{indent}{name.split('/')[-1]:30s}  {str(obj.shape):20s}  {obj.dtype}")
        else:
            print(f"{indent}{name.split('/')[-1]}/")
    f.visititems(_print)
    print()

    # ── Attributes ────────────────────────────────────────────────────────────
    print("─" * 50)
    print("Attributes")
    print("─" * 50)
    for k in f.attrs:
        try:
            v = f.attrs[k]
            print(f"  {k:<20s} {v}")
        except Exception as e:
            print(f"  {k:<20s} <unreadable: {e}>")
    print()

    # ── Stats ─────────────────────────────────────────────────────────────────
    # Find the first image dataset (supports multiple cameras)
    img_datasets = []
    def _find_images(name, obj):
        if isinstance(obj, h5py.Dataset) and "images" in name:
            img_datasets.append(name)
    f.visititems(_find_images)

    if not img_datasets:
        sys.exit("No image datasets found.")

    for dset_name in img_datasets:
        imgs = f[dset_name][:]
        ts   = f["observations/timestamp_ns"][:] if "observations/timestamp_ns" in f else None
        fids = f["observations/frame_id"][:]     if "observations/frame_id"     in f else None

        T, H, W, C = imgs.shape
        camera = dset_name.split("/")[-1]

        print("─" * 50)
        print(f"Camera: {camera}")
        print("─" * 50)
        print(f"  Frames       : {T}")
        print(f"  Resolution   : {W} × {H}")
        print(f"  Channels     : {C}")
        print(f"  dtype        : {imgs.dtype}")
        print(f"  Size in RAM  : {imgs.nbytes / 1e6:.1f} MB")

        if ts is not None:
            duration_s = (ts[-1] - ts[0]) / 1e9
            fps_actual = (T - 1) / duration_s if duration_s > 0 else 0
            print(f"  Duration     : {duration_s:.2f} s")
            print(f"  Actual FPS   : {fps_actual:.1f}")
            gaps = np.diff(ts) / 1e6  # ms
            print(f"  Frame gap    : {gaps.mean():.1f} ms avg  "
                  f"{gaps.min():.1f} ms min  {gaps.max():.1f} ms max")

        if fids is not None:
            dropped = int(fids[-1] - fids[0] - (T - 1))
            print(f"  Dropped frames: {dropped}")

        print(f"  Pixel range  : [{imgs.min()}, {imgs.max()}]")
        for ch, name in enumerate(["R", "G", "B"]):
            ch_data = imgs[:, :, :, ch]
            print(f"  {name} channel  : mean={ch_data.mean():.1f}  "
                  f"std={ch_data.std():.1f}  "
                  f"non-zero={np.count_nonzero(ch_data) * 100 / ch_data.size:.1f}%")
        print()

        # ── Save middle frame as PNG for raw inspection ───────────────────────
        mid = T // 2
        png_path = os.path.splitext(path)[0] + f"_{camera}_frame{mid}.png"
        try:
            from PIL import Image
            Image.fromarray(imgs[mid]).save(png_path)
            print(f"  Saved frame {mid} → {png_path}")
        except ImportError:
            import matplotlib
            matplotlib.image.imsave(png_path, imgs[mid])
            print(f"  Saved frame {mid} → {png_path}")

        # ── Plot 9 random frames ───────────────────────────────────────────────
        n = min(9, T)
        indices = np.sort(np.random.choice(T, size=n, replace=False))

        cols = 3
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3))
        axes = np.array(axes).reshape(-1)

        for ax_idx, frame_idx in enumerate(indices):
            axes[ax_idx].imshow(imgs[frame_idx])
            label = f"frame {frame_idx}"
            if ts is not None:
                t_rel = (ts[frame_idx] - ts[0]) / 1e9
                label += f"\nt={t_rel:.2f}s"
            axes[ax_idx].set_title(label, fontsize=8)
            axes[ax_idx].axis("off")

        # Hide unused axes
        for ax in axes[n:]:
            ax.set_visible(False)

        episode_idx = os.path.basename(os.path.dirname(path))
        fig.suptitle(f"Episode {episode_idx}  –  {camera}  ({T} frames)", fontsize=11)
        plt.tight_layout()
        plt.show()
