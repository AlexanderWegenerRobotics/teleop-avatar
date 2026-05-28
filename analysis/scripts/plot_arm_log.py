"""
plot_arm_log.py  —  plot Franka arm logs from the avatar logger

O_T_EE and F_ext are stored in the robot base frame (Franka convention).
Pass --config <robot_config_local.yaml> to rotate everything into the shared
world frame using each arm's base_pose (T_world_from_base).

Usage:
    python plot_arm_log.py <episode_folder>   [--config <yaml>]
    python plot_arm_log.py <logs_root> <episode_index>  [--config <yaml>]

Examples:
    python plot_arm_log.py build/logs/000
    python plot_arm_log.py build/logs 0        # same thing
    python plot_arm_log.py build/logs          # uses the latest episode folder
    python plot_arm_log.py build/logs --config config/robot_config_local.yaml
"""

import sys
import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    import yaml
    YAML_OK = True
except ImportError:
    YAML_OK = False


# ── Config helpers ──────────────────────────────────────────────────────────────

def load_base_poses(config_path):
    """
    Parse robot_config_local.yaml and return a dict:
        arm_name -> {"R": (3,3) ndarray, "t": (3,) ndarray}
    where R and t are the world-from-base rotation and translation.

    YAML orientation is [w, x, y, z] (Eigen Quaterniond constructor order).
    """
    if not YAML_OK:
        print("[WARN] PyYAML not installed — skipping world-frame transform.")
        return {}

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    poses = {}
    for dev in cfg.get("devices", []):
        if dev.get("type") != "arm":
            continue
        name = dev["name"]
        bp   = dev.get("base_pose", {})
        pos  = bp.get("position", [0, 0, 0])
        ori  = bp.get("orientation", [1, 0, 0, 0])   # [w, x, y, z]
        w, x, y, z = ori
        # Quaternion → rotation matrix (row-major)
        R = np.array([
            [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)     ],
            [2*(x*y + w*z),       1 - 2*(x*x + z*z),  2*(y*z - w*x)     ],
            [2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x*x + y*y)],
        ])
        poses[name] = {"R": R, "t": np.array(pos)}
    return poses


# ── CSV / geometry helpers ──────────────────────────────────────────────────────

def find_episode_folder(root, index=None):
    """Return path to episode folder. If index is None, uses the latest."""
    folders = sorted(glob.glob(os.path.join(root, "[0-9][0-9][0-9]")))
    if not folders:
        return root  # assume root itself is the episode folder
    if index is not None:
        target = os.path.join(root, f"{int(index):03d}")
        if not os.path.isdir(target):
            raise FileNotFoundError(f"Episode folder not found: {target}")
        return target
    return folders[-1]  # latest


def load_arm_csv(path):
    return pd.read_csv(path, sep=";")


def rot_from_flat(df, prefix):
    """Extract rotation matrix columns from column-major flat 4×4.

    Franka column-major layout:
        col0 = indices [0,1,2,3]   → first column of the 4×4
        col1 = indices [4,5,6,7]
        col2 = indices [8,9,10,11]
    Rotation rows: R[row, col] = flat[col*4 + row]
        R[0,0]=flat[0], R[0,1]=flat[4], R[0,2]=flat[8]   → indices 0,4,8
        R[1,0]=flat[1], R[1,1]=flat[5], R[1,2]=flat[9]   → indices 1,5,9
        R[2,0]=flat[2], R[2,1]=flat[6], R[2,2]=flat[10]  → indices 2,6,10
    """
    cols = df[[f"{prefix}_{i}" for i in range(16)]].values
    R = np.stack([
        cols[:, [0, 4,  8]],   # row 0
        cols[:, [1, 5,  9]],   # row 1
        cols[:, [2, 6, 10]],   # row 2
    ], axis=1)  # (N, 3, 3)
    return R


def rot_to_quat(R_batch):
    """Batch rotation matrix → quaternion [w, x, y, z], sign-consistent."""
    n = R_batch.shape[0]
    quats = np.zeros((n, 4))
    for i, R in enumerate(R_batch):
        tr = np.trace(R)
        w = np.sqrt(max(0.0, 1.0 + tr)) / 2.0
        x = np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0
        y = np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0
        z = np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0
        x = np.copysign(x, R[2, 1] - R[1, 2])
        y = np.copysign(y, R[0, 2] - R[2, 0])
        z = np.copysign(z, R[1, 0] - R[0, 1])
        quats[i] = [w, x, y, z]
    # consistent sign across time
    for i in range(1, n):
        if np.dot(quats[i], quats[i - 1]) < 0.0:
            quats[i] *= -1.0
    return quats


def apply_base_transform(pos_base, R_ee_base, f_base, m_base, base):
    """
    Rotate position, orientation, force and torque from robot base frame
    to world frame using the arm's base_pose.

        p_world = R_base @ p_base + t_base
        R_ee_world = R_base @ R_ee_base
        f_world = R_base @ f_base          (pure rotation, no offset)
        m_world = R_base @ m_base
    """
    R = base["R"]   # (3,3)
    t = base["t"]   # (3,)

    pos_world = (R @ pos_base.T).T + t              # (N,3)
    R_ee_world = R[None, :, :] @ R_ee_base          # (N,3,3)
    f_world    = (R @ f_base.T).T                   # (N,3)
    m_world    = (R @ m_base.T).T                   # (N,3)

    return pos_world, R_ee_world, f_world, m_world


# ── Per-arm plot ────────────────────────────────────────────────────────────────

def plot_arm(name, df, out_dir, base=None):
    t = df["time"].values - df["time"].values[0]

    # EE position — column-major 4×4: translation in indices [12, 13, 14]
    pos_base     = df[["O_T_EE_12",     "O_T_EE_13",     "O_T_EE_14"    ]].values
    pos_cmd_base = df[["O_T_EE_cmd_12", "O_T_EE_cmd_13", "O_T_EE_cmd_14"]].values

    # EE rotation matrices (in robot base frame)
    R_ee_base  = rot_from_flat(df, "O_T_EE")
    R_cmd_base = rot_from_flat(df, "O_T_EE_cmd")

    # External wrench in robot base frame (Franka O_F_ext_hat_K)
    f_base = df[["F_ext_0", "F_ext_1", "F_ext_2"]].values
    m_base = df[["F_ext_3", "F_ext_4", "F_ext_5"]].values

    frame_label = "base frame"

    if base is not None:
        # ── transform everything into world frame ──────────────────────────
        pos, R_ee, f_ext, m_ext = apply_base_transform(
            pos_base, R_ee_base, f_base, m_base, base)
        pos_cmd, R_cmd, _, _ = apply_base_transform(
            pos_cmd_base, R_cmd_base, f_base, m_base, base)
        frame_label = "world frame"
    else:
        pos, pos_cmd = pos_base, pos_cmd_base
        R_ee, R_cmd  = R_ee_base, R_cmd_base
        f_ext, m_ext = f_base, m_base

    q_ee  = rot_to_quat(R_ee)
    q_cmd = rot_to_quat(R_cmd)

    state = df["state"].values

    fig = plt.figure(figsize=(14, 16))
    fig.suptitle(f"{name}  ({frame_label})", fontsize=13)
    gs = gridspec.GridSpec(5, 1, figure=fig, hspace=0.4)

    labels_xyz = ["x", "y", "z"]
    colors_xyz = ["#2166ac", "#4dac26", "#d01c8b"]
    colors_q   = ["#333333", "#2166ac", "#4dac26", "#d01c8b"]
    labels_q   = ["w", "x", "y", "z"]

    # ── Row 0: EE position ─────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    for i, (lbl, col) in enumerate(zip(labels_xyz, colors_xyz)):
        ax0.plot(t, pos[:, i],     color=col, lw=1.5, label=f"{lbl} actual")
        ax0.plot(t, pos_cmd[:, i], color=col, lw=1.0, ls="--", alpha=0.6,
                 label=f"{lbl} cmd")
    ax0.set_ylabel(f"EE position\n{frame_label} [m]")
    ax0.legend(ncol=3, fontsize=7)
    ax0.grid(True, alpha=0.3)
    plt.setp(ax0.get_xticklabels(), visible=False)

    # ── Row 1: EE orientation ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    for i, (lbl, col) in enumerate(zip(labels_q, colors_q)):
        ax1.plot(t, q_ee[:, i],  color=col, lw=1.5, label=f"q_{lbl} actual")
        ax1.plot(t, q_cmd[:, i], color=col, lw=1.0, ls="--", alpha=0.6,
                 label=f"q_{lbl} cmd")
    ax1.set_ylabel("EE orientation\n(quaternion)")
    ax1.legend(ncol=4, fontsize=7)
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # ── Row 2: External forces ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    for i, (lbl, col) in enumerate(zip(labels_xyz, colors_xyz)):
        ax2.plot(t, f_ext[:, i], color=col, lw=1.5, label=f"F_{lbl}")
    ax2.set_ylabel(f"ext. force\n{frame_label} [N]")
    ax2.legend(ncol=3, fontsize=8)
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.get_xticklabels(), visible=False)

    # ── Row 3: External torques ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[3], sharex=ax0)
    for i, (lbl, col) in enumerate(zip(labels_xyz, colors_xyz)):
        ax3.plot(t, m_ext[:, i], color=col, lw=1.5, label=f"M_{lbl}")
    ax3.set_ylabel(f"ext. torque\n{frame_label} [Nm]")
    ax3.legend(ncol=3, fontsize=8)
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.get_xticklabels(), visible=False)

    # ── Row 4: State ──────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[4], sharex=ax0)
    ax4.step(t, state, color="#555555", lw=1.2, where="post")
    state_labels = {
        1: "IDLE", 2: "HOMING", 3: "AWAITING", 4: "ENGAGED",
        5: "PAUSED", 6: "RECOVERING", 7: "STOP", 8: "FAULT", 9: "OFFLINE",
    }
    yticks = sorted(state_labels.keys())
    ax4.set_yticks(yticks)
    ax4.set_yticklabels([state_labels.get(v, str(v)) for v in yticks],
                        fontsize=7)
    ax4.set_ylabel("state")
    ax4.set_xlabel("time [s]")
    ax4.grid(True, alpha=0.3)

    out_path = os.path.join(out_dir, f"{name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot Franka arm logs from the avatar logger.")
    parser.add_argument("path",  help="Episode folder or logs root directory")
    parser.add_argument("index", nargs="?", type=int, default=None,
                        help="Episode index (optional; default = latest)")
    parser.add_argument("--config", default=None, metavar="YAML",
                        help="Path to robot_config_local.yaml — enables "
                             "world-frame transform via base_pose")
    args = parser.parse_args()

    episode_dir = find_episode_folder(args.path, args.index)
    print(f"Episode folder: {episode_dir}")

    arm_files = sorted(
        f for f in glob.glob(os.path.join(episode_dir, "arm_*.csv"))
        if not os.path.basename(f).endswith("_meta.csv")
    )
    if not arm_files:
        print(f"No arm_*.csv files found in {episode_dir}")
        sys.exit(1)

    # Optional world-frame transforms keyed by arm name
    base_poses = {}
    if args.config:
        base_poses = load_base_poses(args.config)
        if base_poses:
            print(f"Loaded base poses for: {list(base_poses.keys())}")
        else:
            print("[WARN] No arm base poses found in config.")

    out_dir = episode_dir
    os.makedirs(out_dir, exist_ok=True)

    for f in arm_files:
        name = os.path.basename(f).replace(".csv", "")
        base = base_poses.get(name)   # None if config not provided / not found
        if args.config and base is None:
            print(f"[WARN] No base_pose found for '{name}' — plotting in base frame.")
        print(f"Plotting {name} ...")
        df = load_arm_csv(f)
        plot_arm(name, df, out_dir, base=base)


if __name__ == "__main__":
    main()
