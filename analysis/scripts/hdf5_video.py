"""
hdf5_video.py  --  inspect and export the log-only camera channels

Channels with stream.enabled = false are written by VideoLogger as HDF5 rather
than H.264, because there is no encoder in that path to tee from. The layout
(see include/pipeline/video_logger.hpp) is:

    /observations/images/<camera_name>   (T, H, W, 3)  uint8, deflate-1
    /observations/timestamp_ns           (T,)          uint64
    /observations/frame_id               (T,)          uint64
    attrs: session_id, episode_index, end_reason, frame_count

The frames are LOSSLESS. The .h264 logs from streaming channels are not -- they
are a tee of the encode that was actually transmitted, so they carry the
artefacts the operator saw. Which one is "right" depends on the question:
lossless for policy observations, the H.264 for what the operator experienced.

Usage:
    python hdf5_video.py <file.hdf5>                      # inspect only
    python hdf5_video.py <file.hdf5> --mp4 out.mp4        # export video
    python hdf5_video.py <file.hdf5> --mp4 out.mp4 --fps 20
    python hdf5_video.py <file.hdf5> --frame 120 --png f.png
    python hdf5_video.py <dir>/ --mp4-all                 # every hdf5 in a dir
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


def find_image_dset(f):
    """Returns (name, dataset) for the single camera under /observations/images."""
    grp = f.get("/observations/images")
    if grp is None:
        raise KeyError("no /observations/images group -- not a VideoLogger file")
    names = list(grp.keys())
    if not names:
        raise KeyError("/observations/images is empty")
    if len(names) > 1:
        print(f"  note: {len(names)} cameras in one file, using '{names[0]}'")
    return names[0], grp[names[0]]


def timing(f, n_frames):
    """Effective fps and duration from the stored capture timestamps."""
    ts = f.get("/observations/timestamp_ns")
    if ts is None or len(ts) < 2:
        return None, None, None
    t = np.asarray(ts, dtype=np.uint64).astype(np.float64) / 1e9
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return None, None, None
    return 1.0 / np.median(dt), t[-1] - t[0], dt


def inspect(path):
    with h5py.File(path, "r") as f:
        name, dset = find_image_dset(f)
        T, H, W, C = dset.shape
        fps, dur, dt = timing(f, T)

        raw = T * H * W * C
        on_disk = Path(path).stat().st_size

        print(f"{path}")
        print(f"  camera        {name}")
        print(f"  frames        {T}   {W}x{H}x{C}  {dset.dtype}")
        for k in ("session_id", "episode_index", "end_reason", "frame_count"):
            if k in f.attrs:
                v = f.attrs[k]
                print(f"  {k:<13} {v.decode() if isinstance(v, bytes) else v}")
        if fps:
            print(f"  duration      {dur:.2f} s   median {fps:.2f} fps"
                  f"   (worst gap {dt.max()*1000:.0f} ms)")
        print(f"  raw pixels    {raw/1e6:.1f} MB")
        print(f"  on disk       {on_disk/1e6:.1f} MB   ({raw/max(on_disk,1):.2f}x compression)")

        fid = f.get("/observations/frame_id")
        if fid is not None and len(fid) > 1:
            gaps = np.diff(np.asarray(fid, dtype=np.int64)) - 1
            dropped = int(gaps[gaps > 0].sum())
            if dropped:
                print(f"  DROPPED       {dropped} frames "
                      f"({dropped / (T + dropped) * 100:.1f}%) -- writer queue overran")
            else:
                print(f"  dropped       none")
        return fps


def open_writer(out, fps):
    """imageio-ffmpeg if present (real mp4), else OpenCV."""
    try:
        import imageio.v2 as imageio
        w = imageio.get_writer(out, fps=fps, macro_block_size=1)
        return ("imageio", w)
    except Exception:
        pass
    try:
        import cv2
        return ("cv2", None)
    except ImportError:
        sys.exit("need either imageio + imageio-ffmpeg, or opencv-python, to write video")


def export(path, out, fps_override=None, stride=1):
    with h5py.File(path, "r") as f:
        name, dset = find_image_dset(f)
        T, H, W, _ = dset.shape
        fps, _, _ = timing(f, T)
        fps = fps_override or fps or 20.0
        fps = fps / stride

        kind, writer = open_writer(out, fps)
        if kind == "cv2":
            import cv2
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (W, H))
            if not writer.isOpened():
                sys.exit(f"OpenCV could not open {out} for writing")

        # Read in chunks: the whole episode does not fit in memory at 640x480.
        step = 64
        written = 0
        for i in range(0, T, step):
            block = dset[i:i + step]
            for k in range(0, block.shape[0], stride) if stride > 1 else range(block.shape[0]):
                frame = block[k]
                if kind == "imageio":
                    writer.append_data(frame)
                else:
                    import cv2
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                written += 1

        if kind == "imageio":
            writer.close()
        else:
            writer.release()
        print(f"  -> {out}   {written} frames @ {fps:.2f} fps  "
              f"({Path(out).stat().st_size/1e6:.1f} MB)")


def save_frame(path, index, png):
    import imageio.v2 as imageio
    with h5py.File(path, "r") as f:
        _, dset = find_image_dset(f)
        if index >= dset.shape[0]:
            sys.exit(f"frame {index} out of range (file has {dset.shape[0]})")
        imageio.imwrite(png, dset[index])
        print(f"  -> {png}  (frame {index})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="an .hdf5 file, or a directory when using --mp4-all")
    ap.add_argument("--mp4", default=None, help="write this file")
    ap.add_argument("--mp4-all", action="store_true",
                    help="convert every images_*.hdf5 under PATH, next to each source")
    ap.add_argument("--fps", type=float, default=None,
                    help="override the fps derived from the timestamps")
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth frame")
    ap.add_argument("--frame", type=int, default=None, help="index of a single frame")
    ap.add_argument("--png", default=None, help="where to write --frame")
    args = ap.parse_args()

    p = Path(args.path)

    if args.mp4_all:
        files = sorted(p.rglob("images_*.hdf5"))
        if not files:
            sys.exit(f"no images_*.hdf5 under {p}")
        for h in files:
            inspect(h)
            export(h, h.with_suffix(".mp4"), args.fps, args.stride)
            print()
        return

    if not p.is_file():
        sys.exit(f"{p} is not a file (use --mp4-all for a directory)")

    inspect(p)
    if args.frame is not None:
        save_frame(p, args.frame, args.png or f"frame_{args.frame}.png")
    if args.mp4:
        export(p, args.mp4, args.fps, args.stride)


if __name__ == "__main__":
    main()
