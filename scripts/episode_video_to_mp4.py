"""
episode_video_to_mp4.py  -  Remux a logged episode's H.264 video into an MP4.

Each episode folder (logs/NNN) holds raw H.264 elementary streams named
video_<camera>.h264 (see camera_channel.cpp / video_streamer.cpp). This just
remuxes one into an MP4 container so it opens in a normal player; no
re-encoding unless --crop-markers or --split-stereo is used, so it's fast
and lossless by default.

Streamed frames carry two extra marker rows at the bottom (wall-clock ns +
frame id — see MARKER_ROWS in episode_to_hdf5.py). Pass --crop-markers to
strip them.

head_cam_stereo is left|right side-by-side; --split-stereo writes
<camera>_left.mp4 and <camera>_right.mp4 instead of a single file.

Usage:
    python scripts/episode_video_to_mp4.py logs/007                       # lists available cameras
    python scripts/episode_video_to_mp4.py logs/007 desk_cam
    python scripts/episode_video_to_mp4.py logs/007 wrist_cam_left --crop-markers
    python scripts/episode_video_to_mp4.py logs/007 head_cam_stereo --split-stereo
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

MARKER_ROWS = 2


def available_cameras(episode_dir):
    return sorted(p.stem.removeprefix("video_") for p in episode_dir.glob("video_*.h264"))


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
    ap.add_argument("camera", nargs="?", help="e.g. desk_cam, wrist_cam_left, head_cam_stereo")
    ap.add_argument("--out", help="output path (default: <episode_dir>/<camera>.mp4)")
    ap.add_argument("--crop-markers", action="store_true", help="strip the 2px wall-clock/frame-id marker rows")
    ap.add_argument("--split-stereo", action="store_true", help="split a side-by-side stereo stream into _left/_right MP4s")
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        sys.exit("ffmpeg/ffprobe not found on PATH")

    episode_dir = Path(args.episode_dir)
    cams = available_cameras(episode_dir)
    if not cams:
        sys.exit(f"no video_*.h264 files found in {episode_dir}")

    if not args.camera:
        print("available cameras:", ", ".join(cams))
        return

    if args.camera not in cams:
        sys.exit(f"'{args.camera}' not found in {episode_dir} (available: {', '.join(cams)})")

    src = episode_dir / f"video_{args.camera}.h264"
    dst = Path(args.out) if args.out else episode_dir / f"{args.camera}.mp4"

    if args.split_stereo:
        convert_stereo(src, dst.with_suffix(""), args.crop_markers)
    else:
        convert(src, dst, args.crop_markers)


if __name__ == "__main__":
    main()
