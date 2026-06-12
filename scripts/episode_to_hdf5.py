"""
episode_to_hdf5.py  -  Build a synchronized training HDF5 from a logged episode.

Each episode folder holds video elementary streams (video_<cam>.h264) plus a
per-frame timestamp sidecar (video_<cam>.timestamps.csv) and telemetry CSVs
(arm_left.csv, arm_right.csv, head.csv, scene.csv).  Video and telemetry run at
different, uneven rates, so they are aligned by wall-clock: a fixed-rate master
grid is built over the overlapping window and every stream is sampled
nearest-neighbour onto it.

If a video has no sidecar (older logs), the wall-clock is recovered from the two
marker rows the streamer embeds at the bottom of every frame; failing that, a
uniform rate from the container is assumed.

Usage:
    python episode_to_hdf5.py logs/007                  # one episode -> logs/007/episode.hdf5
    python episode_to_hdf5.py logs --all                # every NNN/ folder under logs/
    python episode_to_hdf5.py logs/007 --rate 30 --scale 0.25
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

SCHEMA_VERSION = 2

try:
    import h5py
except ImportError:
    sys.exit("h5py is required: pip install h5py")

try:
    import cv2
    def _resize(img, w, h):
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
except ImportError:
    from PIL import Image
    def _resize(img, w, h):
        return np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR))

MARKER_ROWS = 2   # streamer appends a timestamp row + a frame-id row below each frame


# ── Video ──────────────────────────────────────────────────────────────────

def probe_dims(path):
    """Returns (width, full_height, fps) of an H.264 elementary stream."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate",
         "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True).stdout.strip()
    w, h, rate = out.split(",")
    num, den = (rate.split("/") + ["1"])[:2]
    fps = float(num) / float(den) if float(den) else 30.0
    return int(w), int(h), fps


def decode_frames(path, full_w, full_h):
    """Yields each decoded RGB frame (full height, including marker rows)."""
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE)
    frame_bytes = full_w * full_h * 3
    while True:
        raw = proc.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        yield np.frombuffer(raw, np.uint8).reshape(full_h, full_w, 3)
    proc.stdout.close()
    proc.wait()


def decode_marker_u64(row):
    """Decodes a 64-bit little-endian value from a marker row (1 px/bit, white=1)."""
    bits = (row[:64, 0].astype(np.uint64) > 128)
    val = np.uint64(0)
    for b in range(64):
        if bits[b]:
            val |= np.uint64(1) << np.uint64(b)
    return int(val)


def load_video(path, scale):
    """Returns (frames[N,h,w,3] uint8, wall_clock_ns[N], frame_ids[N])."""
    full_w, full_h, fps = probe_dims(path)
    img_h = full_h - MARKER_ROWS
    out_w = max(1, int(round(full_w * scale)))
    out_h = max(1, int(round(img_h * scale)))

    sidecar = os.path.splitext(path)[0] + ".timestamps.csv"
    side_ts = None
    if os.path.exists(sidecar):
        arr = np.genfromtxt(sidecar, delimiter=",", skip_header=1)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        side_ts = arr[:, 1].astype(np.int64)

    frames, ts, fids = [], [], []
    for i, frame in enumerate(decode_frames(path, full_w, full_h)):
        if side_ts is not None and i < len(side_ts):
            wall, fid = int(side_ts[i]), i
        else:
            wall = decode_marker_u64(frame[img_h])         # ts row
            fid  = decode_marker_u64(frame[img_h + 1])      # frame-id row
        img = frame[:img_h]
        if scale != 1.0:
            img = _resize(img, out_w, out_h)
        frames.append(img)
        ts.append(wall)
        fids.append(fid)

    if not frames:
        return None
    return np.asarray(frames), np.asarray(ts, np.int64), np.asarray(fids, np.int64)


# ── Telemetry ────────────────────────────────────────────────────────────────

def load_csv(path):
    """Returns (header list, data[N,cols] float, wall_clock_ns[N])."""
    with open(path) as f:
        header = f.readline().strip().split(";")
    data = np.genfromtxt(path, delimiter=";", skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    wall = data[:, header.index("wall_clock_ns")].astype(np.int64)
    return header, data, wall


def load_csv_rows(path):
    """Returns (header list, rows list-of-dicts) — for mixed string/numeric CSVs."""
    with open(path) as f:
        header = f.readline().strip().split(";")
        rows = []
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(dict(zip(header, ln.split(";"))))
    return header, rows


def col_group(header, data, prefix, count):
    """Stacks columns named '<prefix>0..count-1' into [N,count]; None if absent."""
    names = [f"{prefix}{i}" for i in range(count)]
    if not all(n in header for n in names):
        return None
    idx = [header.index(n) for n in names]
    return data[:, idx]


def col_one(header, data, name):
    return data[:, header.index(name)] if name in header else None


# ── Alignment ────────────────────────────────────────────────────────────────

def nearest_idx(stream_ts, grid_ts):
    """For each grid time, index of the nearest stream sample."""
    pos = np.searchsorted(stream_ts, grid_ts)
    pos = np.clip(pos, 1, len(stream_ts) - 1)
    left, right = stream_ts[pos - 1], stream_ts[pos]
    return np.where(np.abs(grid_ts - left) <= np.abs(right - grid_ts), pos - 1, pos)


# ── Episode ────────────────────────────────────────────────────────────────

def read_meta(folder):
    """Pulls seed/mode/color_bin_mapping/success from arm_left_meta.csv if present."""
    out = {}
    mpath = os.path.join(folder, "arm_left_meta.csv")
    if not os.path.exists(mpath):
        return out
    with open(mpath) as f:
        rows = [ln.strip().split(";") for ln in f if ln.strip()]
    if not rows:
        return out
    hdr = rows[0]
    for r in rows[1:]:
        d = dict(zip(hdr, r))
        if d.get("event") == "episode_config":
            out["seed"] = d.get("seed", "")
            out["mode"] = d.get("mode", "")
            out["color_bin_mapping"] = d.get("color_bin_mapping", "")
        if d.get("event") == "episode_end":
            out["success"] = d.get("color_bin_mapping", "")  # reason lands in this col
    return out


def load_intent(folder, grid):
    """
    Loads intention_log.csv and resamples it onto grid (int64 ns timestamps).
    Returns a dict of {col_name: array[T]} using nearest-neighbor alignment on
    timestamp_arrival_ns.  Returns None if the file is absent or unreadable.
    """
    path = os.path.join(folder, "intention_log.csv")
    if not os.path.exists(path):
        return None

    with open(path) as f:
        hdr = f.readline().strip().split(";")
    data = np.genfromtxt(path, delimiter=";", skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[0] == 0 or "timestamp_arrival_ns" not in hdr:
        return None

    ts = data[:, hdr.index("timestamp_arrival_ns")].astype(np.int64)
    sel = nearest_idx(ts, grid)

    out = {}
    for col in hdr:
        if col in ("time",):
            continue
        idx = hdr.index(col)
        out[col] = data[:, idx][sel]
    return out


def load_camera_params(folder, params_path):
    """
    Loads camera intrinsics/extrinsics from a JSON file.
    Searches: explicit path > <folder>/camera_params.json > <folder>/../camera_params.json
    Returns dict {cam_name: {fx,fy,cx,cy,width,height,T_world_cam[[4,4]]}} or {}.
    """
    candidates = []
    if params_path:
        candidates.append(params_path)
    candidates += [
        os.path.join(folder, "camera_params.json"),
        os.path.join(folder, "..", "camera_params.json"),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return {}


def convert(folder, out_path, rate, scale, cameras, camera_params_path=None):
    cams = {}
    cam_native_dims = {}  # name -> (full_w, img_h) before scaling
    for name in cameras:
        vpath = os.path.join(folder, f"video_{name}.h264")
        if os.path.exists(vpath):
            full_w, full_h, _ = probe_dims(vpath)
            cam_native_dims[name] = (full_w, full_h - MARKER_ROWS)
            v = load_video(vpath, scale)
            if v is not None:
                cams[name] = v

    cam_params = load_camera_params(folder, camera_params_path)

    arms, arm_ts = {}, {}
    for arm in ("arm_left", "arm_right"):
        p = os.path.join(folder, f"{arm}.csv")
        if os.path.exists(p):
            arms[arm] = load_csv(p)
            arm_ts[arm] = arms[arm][2]

    head = None
    hp = os.path.join(folder, "head.csv")
    if os.path.exists(hp):
        head = load_csv(hp)

    scene = None
    scene_rows = None
    sp = os.path.join(folder, "scene.csv")
    if os.path.exists(sp):
        scene = load_csv(sp)
        _, scene_rows = load_csv_rows(sp)

    # Master grid over the window all present streams cover.
    starts, ends = [], []
    for _, ts, _ in cams.values():
        starts.append(ts[0]); ends.append(ts[-1])
    for ts in arm_ts.values():
        starts.append(ts[0]); ends.append(ts[-1])
    if head is not None:
        starts.append(head[2][0]); ends.append(head[2][-1])
    if not starts:
        print(f"  [skip] {folder}: no streams found")
        return
    t0, t1 = max(starts), min(ends)
    if t1 <= t0:
        print(f"  [skip] {folder}: streams do not overlap")
        return
    dt = int(1e9 / rate)
    grid = np.arange(t0, t1, dt, dtype=np.int64)
    T = len(grid)

    meta = read_meta(folder)

    with h5py.File(out_path, "w") as f:
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["rate_hz"] = rate
        f.attrs["image_scale"] = scale
        f.attrs["episode_id"] = os.path.basename(folder.rstrip("/\\"))
        for k, v in meta.items():
            f.attrs[k] = v

        obs = f.create_group("observations")
        obs.create_dataset("timestamp_ns", data=grid)

        imgs = obs.create_group("images")
        first_cam = True
        for name, (frames, ts, fids) in cams.items():
            sel = nearest_idx(ts, grid)
            ds = imgs.create_dataset(name, data=frames[sel],
                                     compression="gzip", compression_opts=4,
                                     chunks=(1,) + frames.shape[1:])
            # Reference resolution: pixel coords (gaze_px_*, slot_px_*) are in
            # full-res space; scaled images are stored at image_scale * native.
            if name in cam_native_dims:
                ds.attrs["native_width"]  = cam_native_dims[name][0]
                ds.attrs["native_height"] = cam_native_dims[name][1]
            # Intrinsics + extrinsics from sidecar JSON if available
            if name in cam_params:
                cp = cam_params[name]
                for key in ("fx", "fy", "cx", "cy", "width", "height"):
                    if key in cp:
                        ds.attrs[key] = cp[key]
                if "T_world_cam" in cp:
                    ds.attrs["T_world_cam"] = np.asarray(cp["T_world_cam"], dtype=np.float64)
            if first_cam:
                obs.create_dataset("frame_id", data=fids[sel])
                first_cam = False

        act = f.create_group("actions")
        for arm, (hdr, data, ts) in arms.items():
            sel = nearest_idx(ts, grid)
            g = obs.create_group(arm)
            for field, n in (("q_", 7), ("dq_", 7), ("tau_J_", 7),
                             ("tau_ext_", 7), ("O_T_EE_", 16), ("F_ext_", 6)):
                grp = col_group(hdr, data, field, n)
                if grp is not None:
                    g.create_dataset(field.rstrip("_"), data=grp[sel])
            gw = col_one(hdr, data, "gripper_width")
            if gw is not None:
                g.create_dataset("gripper_width", data=gw[sel])

            ag = act.create_group(arm)
            for field, n in (("q_cmd_", 7), ("O_T_EE_cmd_", 16)):
                grp = col_group(hdr, data, field, n)
                if grp is not None:
                    ag.create_dataset(field.rstrip("_"), data=grp[sel])
            gc = col_one(hdr, data, "gripper_cmd")
            if gc is not None:
                ag.create_dataset("gripper_cmd", data=gc[sel])

        if head is not None:
            hdr, data, ts = head
            sel = nearest_idx(ts, grid)
            hg = obs.create_group("head")
            for field, n in (("q_", 2), ("dq_", 2)):
                grp = col_group(hdr, data, field, n)
                if grp is not None:
                    hg.create_dataset(field.rstrip("_"), data=grp[sel])

        # ── Intent stream ────────────────────────────────────────────────
        intent = load_intent(folder, grid)
        if intent:
            ig = obs.create_group("intent")
            for col, arr in intent.items():
                ig.create_dataset(col, data=arr)

        if scene is not None:
            hdr, data, ts = scene
            sel = nearest_idx(ts, grid)
            sg = f.create_group("scene")

            # Live poses (sampled at grid rate)
            for kind, count in (("obj", 4), ("bin", 4)):
                for i in range(count):
                    pose_cols = [f"{kind}{i}_{c}" for c in ("x", "y", "z", "qw", "qx", "qy", "qz")]
                    if all(c in hdr for c in pose_cols):
                        idx = [hdr.index(c) for c in pose_cols]
                        vals = data[:, idx][sel]
                        if np.any(np.abs(vals) > 1e-9):
                            sg.create_dataset(f"{kind}{i}_pose", data=vals)

            # Per-object spawn config — constant per episode
            for i in range(4):
                for field in ("spawn_yaw", "scale"):
                    col = f"obj{i}_{field}"
                    if col in hdr:
                        sg.create_dataset(f"obj{i}_{field}",
                                          data=data[:, hdr.index(col)][sel])

            # Per-object color — constant string; read from first scene row
            if scene_rows:
                first = scene_rows[0]
                for i in range(4):
                    col = f"obj{i}_color"
                    if col in first and first[col]:
                        sg.create_dataset(f"obj{i}_color",
                                          data=np.bytes_(first[col]))
                    col_name = f"obj{i}_name"
                    if col_name in first and first[col_name]:
                        sg.create_dataset(f"obj{i}_name",
                                          data=np.bytes_(first[col_name]))

            # Lighting — constant per episode
            light_groups = {
                "main_pos":      ["light_main_pos_x",      "light_main_pos_y",      "light_main_pos_z"],
                "main_diffuse":  ["light_main_diffuse_r",  "light_main_diffuse_g",  "light_main_diffuse_b"],
                "main_specular": ["light_main_specular_r", "light_main_specular_g", "light_main_specular_b"],
                "fill_diffuse":  ["light_fill_diffuse_r",  "light_fill_diffuse_g",  "light_fill_diffuse_b"],
            }
            if any(cols[0] in hdr for cols in light_groups.values()):
                lg = sg.create_group("lighting")
                for name, cols in light_groups.items():
                    if all(c in hdr for c in cols):
                        vals = np.array([data[:, hdr.index(c)].mean() for c in cols],
                                        dtype=np.float32)
                        lg.create_dataset(name, data=vals)

    print(f"  [ok] {out_path}  T={T}  cams={list(cams)}  "
          f"dur={(t1 - t0) / 1e9:.2f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="episode folder, or logs root with --all")
    ap.add_argument("--all", action="store_true", help="process every NNN/ folder under path")
    ap.add_argument("--rate", type=float, default=30.0, help="output control rate (Hz)")
    ap.add_argument("--scale", type=float, default=0.25, help="image downscale factor (1.0 = native)")
    ap.add_argument("--cameras", nargs="+",
                    default=["head_cam_stereo", "wrist_cam_left", "wrist_cam_right"])
    ap.add_argument("--out", default="episode.hdf5", help="output filename within each folder")
    ap.add_argument("--camera-params", default=None,
                    help="path to camera_params.json with intrinsics/extrinsics; "
                         "also searched automatically at <episode>/camera_params.json")
    args = ap.parse_args()

    if args.all:
        folders = sorted(d for d in glob.glob(os.path.join(args.path, "[0-9]" * 3))
                         if os.path.isdir(d))
    else:
        folders = [args.path]

    for folder in folders:
        out_path = os.path.join(folder, args.out)
        print(f"Converting {folder} ...")
        try:
            convert(folder, out_path, args.rate, args.scale, args.cameras,
                    camera_params_path=args.camera_params)
        except Exception as e:
            print(f"  [error] {folder}: {e}")


if __name__ == "__main__":
    main()
