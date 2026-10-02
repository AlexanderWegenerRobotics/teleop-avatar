#!/usr/bin/env python3
"""
Plot the intention log of one episode (default: latest), saves intention_plot.png.

Usage: python plot_intention.py [episode_folder]
"""

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# must match SlotType in intention_sample.hpp
SLOT_TYPE_NAMES = {0: "EE-Left", 1: "EE-Right", 2: "Pick", 3: "Place"}
SLOT_COLORS     = {0: "#4C72B0", 1: "#DD8452", 2: "#55A868", 3: "#C44E52"}
NONE_COLOR      = "#AAAAAA"


def find_latest_episode() -> str:
    search_roots = [os.getcwd()]
    for _ in range(3):
        search_roots.append(os.path.dirname(search_roots[-1]))

    candidates = []
    for root in search_roots:
        candidates += glob.glob(os.path.join(root, "build", "logs", "[0-9][0-9][0-9]"))
        candidates += glob.glob(os.path.join(root, "logs", "[0-9][0-9][0-9]"))

    if not candidates:
        sys.exit("No episode folders found. Pass the folder path explicitly.")

    return sorted(candidates)[-1]


def load_csv(path: str, required: bool = True) -> pd.DataFrame:
    if not os.path.exists(path):
        if required:
            sys.exit(f"Required file not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, sep=";")


def load_episode(folder: str):
    df_int   = load_csv(os.path.join(folder, "intention_log.csv"))
    df_meta  = load_csv(os.path.join(folder, "intention_log_meta.csv"), required=False)
    df_scene = load_csv(os.path.join(folder, "scene.csv"), required=False)
    return df_int, df_meta, df_scene


def infer_schema(df: pd.DataFrame):
    """Return (n_slots, slot_types, slot_labels, noneef_slot_indices)."""
    n_slots = int(df["n_slots"].mode().iloc[0])
    slot_types  = [int(df[f"slot_type_{i}"].mode().iloc[0]) for i in range(n_slots)]
    slot_labels = [SLOT_TYPE_NAMES.get(t, f"slot_{i}") for i, t in enumerate(slot_types)]
    # distances are only logged for non-EEF slots
    noneef = [(i, slot_labels[i]) for i, t in enumerate(slot_types) if t not in (0, 1)]
    return n_slots, slot_types, slot_labels, noneef


def extract_annotations(df_meta: pd.DataFrame):
    """Return list of (time_s, label) for voice annotation events."""
    if df_meta.empty or "event" not in df_meta.columns:
        return []
    ann = df_meta[df_meta["event"] == "annotation"]
    return list(zip(ann["time_s"].astype(float), ann["reason"].astype(str)))


def extract_episode_bounds(df_meta: pd.DataFrame):
    if df_meta.empty or "event" not in df_meta.columns:
        return [], []
    starts = df_meta[df_meta["event"] == "episode_start"]["time_s"].astype(float).tolist()
    ends   = df_meta[df_meta["event"] == "episode_end"]["time_s"].astype(float).tolist()
    return starts, ends


def plot_episode(folder: str):
    df_int, df_meta, df_scene = load_episode(folder)

    n_slots, slot_types, slot_labels, noneef = infer_schema(df_int)
    annotations = extract_annotations(df_meta)
    ep_starts, ep_ends = extract_episode_bounds(df_meta)

    gv = df_int[df_int["gaze_valid"] == 1].copy()
    t_all = df_int["time"].values
    t_gv  = gv["time"].values

    belief_cols = [f"slot_belief_{i}" for i in range(n_slots + 1)]  # last one is "none"
    colors_all  = [SLOT_COLORS.get(slot_types[i], f"C{i}") for i in range(n_slots)] + [NONE_COLOR]
    labels_all  = slot_labels + ["None"]

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    fig.suptitle(f"Intention  —  {os.path.basename(folder)}", fontsize=13, fontweight="bold")
    fig.subplots_adjust(hspace=0.35)

    ax = axes[0]
    if len(gv):
        belief_vals = [gv[c].values for c in belief_cols if c in gv.columns]
        ax.stackplot(t_gv, belief_vals, labels=labels_all, colors=colors_all, alpha=0.85)
    ax.set_ylim(0, 1.01)
    ax.set_ylabel("Belief")
    ax.set_title("Gaze Belief  (gaze-valid frames)")
    ax.legend(loc="upper right", fontsize=8, ncol=len(labels_all))

    ax = axes[1]
    if len(gv):
        belief_arr = gv[[c for c in belief_cols if c in gv.columns]].values
        dominant   = np.argmax(belief_arr, axis=1)
        dom_colors = [colors_all[d] for d in dominant]
        ax.scatter(t_gv, dominant, c=dom_colors, s=5, zorder=3)
    ax.set_yticks(range(len(labels_all)))
    ax.set_yticklabels(labels_all, fontsize=8)
    ax.set_ylabel("Target")
    ax.set_title("Dominant Gaze Target")

    ax = axes[2]
    for k, (slot_idx, slot_name) in enumerate(noneef):
        col_l = f"slot_dist_{k * 2}"
        col_r = f"slot_dist_{k * 2 + 1}"
        color = SLOT_COLORS.get(slot_types[slot_idx], f"C{k}")
        if col_l in df_int.columns:
            ax.plot(t_all, df_int[col_l], color=color, linewidth=0.9,
                    label=f"L → {slot_name}")
        if col_r in df_int.columns:
            ax.plot(t_all, df_int[col_r], color=color, linewidth=0.9,
                    linestyle="--", alpha=0.7, label=f"R → {slot_name}")
    ax.set_ylabel("Distance (m)")
    ax.set_title("EEF Distance to Pick / Place")
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    ax  = axes[3]
    ax2 = ax.twinx()

    if not df_scene.empty:
        ax.plot(df_scene["time"], df_scene["object_z"], color="steelblue",
                linewidth=1.0, label="Object Z")
        row = df_scene.iloc[0]
        ax.axhline(row["pick_z"],  color="steelblue", linewidth=0.6,
                   linestyle=":", alpha=0.6, label="Pick Z target")
        ax.axhline(row["place_z"], color="mediumseagreen", linewidth=0.6,
                   linestyle=":", alpha=0.6, label="Place Z target")

    ax2.plot(t_all, df_int["gripper_left"],  color="darkorange", linewidth=0.8,
             alpha=0.8, label="Gripper L")
    ax2.plot(t_all, df_int["gripper_right"], color="green", linewidth=0.8,
             alpha=0.8, linestyle="--", label="Gripper R")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("Gripper (0=open, 1=closed)", fontsize=8)

    ax.set_ylabel("Z (m)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Object Trajectory + Gripper State")

    lines,  lbls  = ax.get_legend_handles_labels()
    lines2, lbls2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, lbls + lbls2, loc="upper right", fontsize=8, ncol=2)

    for axt in axes:
        for ts in ep_starts:
            axt.axvline(ts, color="lime", linewidth=1.0, linestyle="-", alpha=0.5, zorder=2)
        for ts in ep_ends:
            axt.axvline(ts, color="tomato", linewidth=1.0, linestyle="-", alpha=0.5, zorder=2)
        for ts, label in annotations:
            axt.axvline(ts, color="purple", linewidth=0.8, linestyle=":", alpha=0.7, zorder=2)

    if annotations:
        y_top = axes[0].get_ylim()[1]
        for ts, label in annotations:
            axes[0].text(ts, y_top * 0.95, label, fontsize=6, rotation=90,
                         va="top", ha="right", color="purple", alpha=0.85)

    from matplotlib.lines import Line2D
    marker_legend = []
    if ep_starts:
        marker_legend.append(Line2D([0], [0], color="lime",   linewidth=1.2, label="Episode start"))
    if ep_ends:
        marker_legend.append(Line2D([0], [0], color="tomato", linewidth=1.2, label="Episode end"))
    if annotations:
        marker_legend.append(Line2D([0], [0], color="purple", linewidth=0.8,
                                    linestyle=":", label="Voice annotation"))
    if marker_legend:
        axes[0].legend(
            handles=list(axes[0].get_legend().legend_handles) + marker_legend,
            loc="upper right", fontsize=8, ncol=len(labels_all) + len(marker_legend)
        )

    out_path = os.path.join(folder, "intention_plot.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot intention log for one teleoperation episode")
    parser.add_argument("folder", nargs="?", default=None,
                        help="Episode folder (e.g. build/logs/000). "
                             "Omit to auto-select the latest.")
    args = parser.parse_args()

    folder = os.path.abspath(args.folder) if args.folder else find_latest_episode()
    print(f"Plotting: {folder}")
    plot_episode(folder)
