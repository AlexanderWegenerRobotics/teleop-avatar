"""
validate_episode.py  -  Pre-collection gate check for a converted episode.

Asserts all P0/P1/P2 acceptance criteria on a freshly converted episode.hdf5.
Exit 0 = all checks passed. Exit 1 = one or more failures.

Usage:
    python validate_episode.py logs/007
    python validate_episode.py logs/007/episode.hdf5
    python validate_episode.py logs --all
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("h5py is required: pip install h5py")


PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
failures = []


def check(condition, label, detail=""):
    if condition:
        print(f"  [{PASS}] {label}")
    else:
        print(f"  [{FAIL}] {label}" + (f": {detail}" if detail else ""))
        failures.append(label)


def validate(episode_hdf5):
    folder = os.path.dirname(episode_hdf5)

    print(f"\n=== {episode_hdf5} ===")

    if not os.path.exists(episode_hdf5):
        print(f"  [{FAIL}] HDF5 file not found")
        failures.append("HDF5 file exists")
        return

    with h5py.File(episode_hdf5, "r") as f:

        # ── Reference length ──────────────────────────────────────────────
        T = None
        if "observations/timestamp_ns" in f:
            T = len(f["observations/timestamp_ns"])
        check(T is not None and T > 0, "observations/timestamp_ns present and non-empty",
              f"T={T}")

        # ── P0: gripper_cmd in actions ────────────────────────────────────
        for arm in ("arm_left", "arm_right"):
            key = f"actions/{arm}/gripper_cmd"
            exists = key in f
            check(exists, f"{key} exists")
            if exists and T:
                arr = f[key][:]
                check(len(arr) == T, f"{key} length == T", f"len={len(arr)} T={T}")
                check(not np.all(arr == arr[0]), f"{key} not constant (varies over episode)",
                      "constant signal — ensure a grasp episode is tested; OK for open-only runs")

        # ── P1: intent group ──────────────────────────────────────────────
        intent_path = "observations/intent"
        check(intent_path in f, f"{intent_path} group exists")
        if intent_path in f:
            ig = f[intent_path]
            check(len(ig) > 0, f"{intent_path} is non-empty")
            if T and len(ig) > 0:
                first_key = next(iter(ig))
                arr_len = len(ig[first_key])
                check(arr_len == T,
                      f"{intent_path}/{first_key} aligned to timestamp_ns",
                      f"len={arr_len} T={T}")
            for col in ("gaze_px_x", "gaze_px_y", "gaze_valid"):
                check(col in ig, f"{intent_path}/{col} present")

        # ── P1: scene metadata ────────────────────────────────────────────
        check("scene" in f, "scene group exists")
        if "scene" in f:
            sg = f["scene"]
            for i in range(4):
                check(f"obj{i}_spawn_yaw" in sg, f"scene/obj{i}_spawn_yaw present")
                check(f"obj{i}_scale"     in sg, f"scene/obj{i}_scale present")
                check(f"obj{i}_color"     in sg, f"scene/obj{i}_color present")
                check(f"obj{i}_name"      in sg, f"scene/obj{i}_name present")
            check("lighting" in sg, "scene/lighting group present")

        check("color_bin_mapping" in f.attrs, "attrs/color_bin_mapping present",
              str(dict(f.attrs)))
        cbm = f.attrs.get("color_bin_mapping", "")
        check(cbm not in ("", "n/a", None), "color_bin_mapping non-empty",
              f"value={cbm!r}")

        check("seed" in f.attrs, "attrs/seed present")
        check("mode" in f.attrs, "attrs/mode present")

    # ── P2: intention_log_meta.csv populated ──────────────────────────────
    imeta = os.path.join(folder, "intention_log_meta.csv")
    check(os.path.exists(imeta), "intention_log_meta.csv exists")
    if os.path.exists(imeta):
        with open(imeta) as fh:
            rows = list(csv.DictReader(fh, delimiter=";"))
        check(len(rows) > 0, "intention_log_meta.csv non-empty")
        has_end = any(r.get("event") == "episode_end" for r in rows)
        check(has_end, "intention_log_meta.csv has episode_end row")
        if has_end:
            end_row = next(r for r in rows if r.get("event") == "episode_end")
            label = end_row.get("color_bin_mapping", "")
            check(label not in ("", None), "episode_end has non-empty success label",
                  f"value={label!r}")

    # ── P2: scene_meta.csv annotation alignment ───────────────────────────
    smeta = os.path.join(folder, "scene_meta.csv")
    check(os.path.exists(smeta), "scene_meta.csv exists")
    if os.path.exists(smeta):
        with open(smeta) as fh:
            rows = list(csv.DictReader(fh, delimiter=";"))
        ann_rows = [r for r in rows if r.get("event") == "annotation"]
        if ann_rows:
            for r in ann_rows:
                seed_val = r.get("seed", "")
                check(seed_val in ("", None) or seed_val.lstrip("-").isdigit() or seed_val == "",
                      "annotation seed column is empty (not the label)",
                      f"seed={seed_val!r}")
                ann_label = r.get("ann_label", "")
                check(ann_label not in ("", None),
                      "annotation ann_label column populated",
                      f"value={ann_label!r}")
        else:
            print(f"  [skip] no annotation rows in scene_meta.csv")

    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="episode folder, episode.hdf5 path, or logs root with --all")
    ap.add_argument("--all", action="store_true",
                    help="validate every NNN/episode.hdf5 under path")
    ap.add_argument("--hdf5", default="episode.hdf5",
                    help="HDF5 filename inside each episode folder")
    args = ap.parse_args()

    if args.path.endswith(".hdf5") and os.path.isfile(args.path):
        targets = [args.path]
    elif args.all:
        targets = sorted(
            os.path.join(d, args.hdf5)
            for d in glob.glob(os.path.join(args.path, "[0-9]" * 3))
            if os.path.isdir(d)
        )
    else:
        targets = [os.path.join(args.path, args.hdf5)]

    if not targets:
        sys.exit(f"No episode HDF5 files found under {args.path}")

    for t in targets:
        validate(t)

    if failures:
        print(f"{len(failures)} check(s) FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print(f"All checks passed ({len(targets)} episode(s)).")
        sys.exit(0)


if __name__ == "__main__":
    main()
