"""
Convert a logged episode's camera video into MP4.
H.264 streams (video_<cam>.h264) are remuxed losslessly unless --crop-markers / --split-stereo;
log-only cameras (images_<cam>.hdf5) are encoded from the raw frames at their logged frame rate.

Usage:
    python scripts/episode_video_to_mp4.py logs/007                       # lists available cameras
    python scripts/episode_video_to_mp4.py logs/007 desk_cam
    python scripts/episode_video_to_mp4.py logs/007 wrist_cam_left --crop-markers
    python scripts/episode_video_to_mp4.py logs/007 head_cam_stereo --split-stereo
    python scripts/episode_video_to_mp4.py logs/007 workspace_cam_1
    python scripts/episode_video_to_mp4.py logs/007 --all
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

MARKER_ROWS = 2


def available_cameras(episode_dir):
    """Map camera name to its source file, H.264 preferred over HDF5."""
    cams = {p.stem.removeprefix("images_"): p for p in episode_dir.glob("images_*.hdf5")}
    cams.update({p.stem.removeprefix("video_"): p for p in episode_dir.glob("video_*.h264")})
    return dict(sorted(cams.items()))


def probe_width(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout.strip()
    return int(out)


def run_ffmpeg(args):
    subprocess.run(["ffmpeg", "-y", "-v", "error"] + args, check=True)


def convert(src, dst, crop_markers):
    if crop_markers:
        run_ffmpeg(["-i", str(src), "-vf", f"crop=iw:ih-{MARKER_ROWS}:0:0",
                    "-c:v", "libx264", "-crf", "18", str(dst)])
    else:
        run_ffmpeg(["-i", str(src), "-c:v", "copy", str(dst)])
    print(f"wrote {dst}")


def hdf5_fps(ts_ns):
    """Mean frame rate from the logged capture timestamps, 30 if there are too few."""
    if len(ts_ns) < 2 or ts_ns[-1] <= ts_ns[0]:
        return 30.0
    return (len(ts_ns) - 1) / ((ts_ns[-1] - ts_ns[0]) * 1e-9)


def convert_hdf5(src, camera, dst, crop_markers):
    """Pipe the raw RGB frames of an images_<cam>.hdf5 log into an H.264 MP4."""
    import h5py
    try:
        f = h5py.File(src, "r")
    except OSError as e:
        print(f"skipped {src.name}: cannot open ({e}); is the episode still recording?")
        return
    with f:
        imgs = f[f"observations/images/{camera}"]
        ts = f["observations/timestamp_ns"][:]
        n, h, w, _ = imgs.shape
        if n == 0:
            print(f"skipped {src.name}: no frames")
            return
        fps = hdf5_fps(ts)
        vf = f"crop=iw:ih-{MARKER_ROWS}:0:0," if crop_markers else ""
        cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{w}x{h}", "-r", f"{fps:.3f}", "-i", "-",
               "-vf", vf + "pad=ceil(iw/2)*2:ceil(ih/2)*2",
               "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(dst)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for i in range(0, n, 64):
            proc.stdin.write(imgs[i:i + 64].tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            sys.exit(f"ffmpeg failed on {src}")
    print(f"wrote {dst}  ({n} frames, {fps:.1f} fps)")


def convert_stereo(src, dst_stem, crop_markers):
    w = probe_width(src)
    half = w // 2
    h_expr = f"ih-{MARKER_ROWS}" if crop_markers else "ih"
    for eye, x0 in (("left", 0), ("right", half)):
        dst = dst_stem.with_name(f"{dst_stem.name}_{eye}.mp4")
        run_ffmpeg(["-i", str(src), "-vf", f"crop={half}:{h_expr}:{x0}:0",
                    "-c:v", "libx264", "-crf", "18", str(dst)])
        print(f"wrote {dst}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode_dir", help="e.g. logs/007")
    ap.add_argument("camera", nargs="?", help="e.g. desk_cam, wrist_cam_left, head_cam_stereo, workspace_cam_1")
    ap.add_argument("--all", action="store_true", help="convert every camera in the episode")
    ap.add_argument("--out", help="output path (default: <episode_dir>/<camera>.mp4)")
    ap.add_argument("--crop-markers", action="store_true", help="strip the 2px wall-clock/frame-id marker rows")
    ap.add_argument("--split-stereo", action="store_true", help="split a side-by-side stereo stream into _left/_right MP4s")
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        sys.exit("ffmpeg/ffprobe not found on PATH")

    episode_dir = Path(args.episode_dir)
    cams = available_cameras(episode_dir)
    if not cams:
        sys.exit(f"no video_*.h264 or images_*.hdf5 files found in {episode_dir}")

    if args.all:
        names = list(cams)
    elif args.camera:
        if args.camera not in cams:
            sys.exit(f"'{args.camera}' not found in {episode_dir} (available: {', '.join(cams)})")
        names = [args.camera]
    else:
        print("available cameras:", ", ".join(f"{c} ({p.suffix[1:]})" for c, p in cams.items()))
        return

    for name in names:
        src = cams[name]
        dst = Path(args.out) if args.out and not args.all else episode_dir / f"{name}.mp4"
        if src.suffix == ".hdf5":
            convert_hdf5(src, name, dst, args.crop_markers)
        elif args.split_stereo:
            convert_stereo(src, dst.with_suffix(""), args.crop_markers)
        else:
            convert(src, dst, args.crop_markers)


if __name__ == "__main__":
    main()
