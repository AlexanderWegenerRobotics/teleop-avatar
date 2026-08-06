"""
analyze_latency.py  -  Summarize capture->encode latency from a logged episode's
*.timestamps.csv sidecar(s) (see VideoStreamer::pushFrame / onNewSample in
video_streamer.cpp). Prints min/median/p95/max and plots capture_to_encode_ns
per frame for each camera found in the episode folder.

Usage:
    python scripts/analyze_latency.py 7                       # logs/007, all cameras
    python scripts/analyze_latency.py logs/007
    python scripts/analyze_latency.py logs/007 --camera head_cam_stereo
    python scripts/analyze_latency.py logs/007 --no-plot
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def resolve_episode_dir(arg):
    if arg.isdigit():
        return Path("logs") / f"{int(arg):03d}"
    return Path(arg)


def load_timestamps_csv(path):
    frame_idx, wall_clock_ns, capture_time_ns, capture_to_encode_ns = [], [], [], []
    skipped = 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                fi  = int(row["frame_idx"])
                wc  = int(row["wall_clock_ns"])
                cap = int(row["capture_time_ns"])
                c2e = int(row["capture_to_encode_ns"])
            except (TypeError, ValueError):
                # Truncated row, e.g. the last line of a file that wasn't closed
                # cleanly (process killed mid-write instead of stopEncodedLog()).
                skipped += 1
                continue
            frame_idx.append(fi)
            wall_clock_ns.append(wc)
            capture_time_ns.append(cap)
            capture_to_encode_ns.append(c2e)
    if skipped:
        print(f"  ({skipped} malformed/truncated row(s) skipped in {path.name})")
    return {
        "frame_idx": np.array(frame_idx),
        "wall_clock_ns": np.array(wall_clock_ns),
        "capture_time_ns": np.array(capture_time_ns),
        "capture_to_encode_ns": np.array(capture_to_encode_ns),
    }


def print_stats(camera, data):
    valid = data["capture_time_ns"] != 0
    delta_ms = data["capture_to_encode_ns"][valid] / 1e6
    dropped = int((~valid).sum())

    print(f"Camera: {camera}")
    print(f"  Frames               : {len(data['frame_idx'])}  ({dropped} without a capture_time_ns)")
    if delta_ms.size == 0:
        print("  No frames with valid capture_time_ns - nothing to summarize.\n")
        return
    print(f"  capture->encode (ms) : min={delta_ms.min():.1f}  "
          f"median={np.median(delta_ms):.1f}  "
          f"p95={np.percentile(delta_ms, 95):.1f}  "
          f"max={delta_ms.max():.1f}")

    if len(data["wall_clock_ns"]) > 1:
        gaps_ms = np.diff(data["wall_clock_ns"]) / 1e6
        fps = 1000.0 / gaps_ms.mean() if gaps_ms.mean() > 0 else 0.0
        print(f"  encode frame gap (ms): mean={gaps_ms.mean():.1f}  "
              f"({fps:.1f} fps)")
    print()


def plot(camera, data, out_path):
    import matplotlib.pyplot as plt

    valid = data["capture_time_ns"] != 0
    idx = data["frame_idx"][valid]
    delta_ms = data["capture_to_encode_ns"][valid] / 1e6

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(idx, delta_ms, marker=".", linewidth=0.8, markersize=3)
    ax.set_xlabel("frame_idx")
    ax.set_ylabel("capture -> encode (ms)")
    ax.set_title(f"{camera}  ({len(idx)} frames)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  Saved plot -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode", help="episode number (7) or folder (logs/007)")
    ap.add_argument("--camera", help="only analyze video_<camera>.timestamps.csv")
    ap.add_argument("--no-plot", action="store_true", help="skip matplotlib plots (stats only)")
    args = ap.parse_args()

    episode_dir = resolve_episode_dir(args.episode)
    if not episode_dir.is_dir():
        sys.exit(f"episode folder not found: {episode_dir}")

    csv_paths = sorted(episode_dir.glob("video_*.timestamps.csv"))
    if args.camera:
        csv_paths = [p for p in csv_paths if p.stem == f"video_{args.camera}.timestamps"]

    if not csv_paths:
        sys.exit(f"no *.timestamps.csv files found in {episode_dir}")

    for path in csv_paths:
        camera = path.stem.removeprefix("video_").removesuffix(".timestamps")
        data = load_timestamps_csv(path)
        print_stats(camera, data)
        if not args.no_plot:
            plot(camera, data, episode_dir / f"{camera}_latency.png")


if __name__ == "__main__":
    main()
