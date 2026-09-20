"""
compare_gmo_truth.py  --  momentum observer vs MuJoCo ground truth

Joins <device>_log.csv (the observer's tau_ext / F_ext) against
<device>_wrench_truth.csv (MuJoCo's exact contact forces) on the sim_time
column and plots them together.

Two things this script does that a naive overlay does not:

1. It filters the truth through the SAME first-order lag as the observer.
   The GMO is r += K*(p - p_prev - (tau - tau_model + r)*dt) with K = 50 rad/s,
   i.e. a ~20 ms lag and about 8 Hz of bandwidth. The truth is instantaneous.
   Comparing them raw shows the observer's designed bandwidth as if it were
   error. The dashed trace is the fair comparison.

2. It reconciles the decomposition. tau_contact + tau_friction + tau_limit
   must equal tau_constraint; a non-zero residual means a sign convention or a
   constraint type is being mishandled, and nothing else in the plot is
   trustworthy until it is flat.

Joint space is the primary comparison -- it is what the observer estimates, and
it carries no frame. F_ext is the observer's tau_ext pushed through a DAMPED
least-squares inverse of J^T, so disagreement there can be projection error
rather than observer error.

Usage:
    python compare_gmo_truth.py <log_dir> [--device arm_left] [--k-gmo 50]
    python compare_gmo_truth.py build/log --device arm_right --save out.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load(path):
    df = pd.read_csv(path, sep=";", engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def cols(df, base, n):
    names = [f"{base}_{i}" for i in range(n)]
    missing = [c for c in names if c not in df.columns]
    if missing:
        raise KeyError(f"{base}: missing columns {missing}")
    return df[names].to_numpy(dtype=float)


def first_order(x, t, k):
    """Same lag the observer has: dy/dt = k*(x - y), integrated on t."""
    y = np.zeros_like(x)
    if len(t) == 0:
        return y
    y[0] = x[0]
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        if not np.isfinite(dt) or dt <= 0:
            y[i] = y[i - 1]
            continue
        a = 1.0 - np.exp(-k * dt)      # exact for a step, stable at any dt
        y[i] = y[i - 1] + a * (x[i] - y[i - 1])
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir")
    ap.add_argument("--device", default="arm_left")
    ap.add_argument("--k-gmo", type=float, default=50.0,
                    help="observer gain, must match K_GMO in robot.hpp")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    d = Path(args.log_dir)
    arm_path = d / f"{args.device}_log.csv"
    truth_path = d / f"{args.device}_wrench_truth.csv"
    for p in (arm_path, truth_path):
        if not p.exists():
            sys.exit(f"missing {p}")

    arm = load(arm_path)
    tru = load(truth_path)

    if "sim_time" not in arm.columns:
        sys.exit("arm log has no sim_time column -- nothing to join on")

    t_a = arm["sim_time"].to_numpy(dtype=float)
    t_t = tru["sim_time"].to_numpy(dtype=float)
    ok = np.isfinite(t_a) & (t_a > 0)
    arm, t_a = arm[ok], t_a[ok]

    tau_obs = cols(arm, "tau_ext", 7)
    tau_con = cols(tru, "tau_contact", 7)
    tau_fri = cols(tru, "tau_friction", 7)
    tau_lim = cols(tru, "tau_limit", 7)
    tau_all = cols(tru, "tau_constraint", 7)

    resid = tau_all - (tau_con + tau_fri + tau_lim)
    print(f"decomposition residual: max |tau_constraint - (contact+friction+limit)| = "
          f"{np.nanmax(np.abs(resid)):.3e} Nm")
    if np.nanmax(np.abs(resid)) > 1e-6:
        print("  ^ NOT flat. A constraint sign or type is mishandled; fix that first.")

    # truth onto the observer's timebase, then through the observer's own lag
    tau_t = np.column_stack([np.interp(t_a, t_t, tau_con[:, j]) for j in range(7)])
    tau_f = np.column_stack([first_order(tau_t[:, j], t_a, args.k_gmo) for j in range(7)])

    print(f"\n{args.device}: {len(t_a)} observer rows, {len(t_t)} truth rows, "
          f"{t_a[-1] - t_a[0]:.1f} s of sim time")
    print(f"{'joint':>6} {'rms err':>9} {'rms err (lag-matched)':>23} {'truth rms':>10}")
    for j in range(7):
        e_raw = np.sqrt(np.nanmean((tau_obs[:, j] - tau_t[:, j]) ** 2))
        e_lag = np.sqrt(np.nanmean((tau_obs[:, j] - tau_f[:, j]) ** 2))
        print(f"{j:>6} {e_raw:>9.4f} {e_lag:>23.4f} "
              f"{np.sqrt(np.nanmean(tau_t[:, j] ** 2)):>10.4f}")

    fig, axes = plt.subplots(4, 2, figsize=(14, 11), sharex=True)
    for j in range(7):
        ax = axes[j // 2, j % 2]
        ax.plot(t_a, tau_t[:, j], lw=0.8, color="0.7", label="MuJoCo contact")
        ax.plot(t_a, tau_f[:, j], lw=1.0, ls="--", color="tab:orange",
                label=f"MuJoCo, {args.k_gmo:.0f} rad/s lag")
        ax.plot(t_a, tau_obs[:, j], lw=1.0, color="tab:blue", label="observer tau_ext")
        ax.set_ylabel(f"j{j}  [Nm]")
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=8, loc="upper right")

    ax = axes[3, 1]
    if "F_base_fx" in tru.columns and "F_ext_0" in arm.columns:
        F_obs = cols(arm, "F_ext", 6)
        F_tru = tru[[f"F_base_{a}" for a in ("fx", "fy", "fz")]].to_numpy(dtype=float)
        for i, lbl in enumerate(("x", "y", "z")):
            f_i = np.interp(t_a, t_t, F_tru[:, i])
            ax.plot(t_a, first_order(f_i, t_a, args.k_gmo), lw=0.9, ls="--",
                    color=f"C{i}", alpha=0.7)
            ax.plot(t_a, F_obs[:, i], lw=0.9, color=f"C{i}", label=f"F{lbl}")
        ax.set_ylabel("force [N]  base frame")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_title("Cartesian (secondary: damped J^T inverse on the observer side)",
                     fontsize=8)
    else:
        ax.text(0.5, 0.5, "no Cartesian columns\n(ee_body not configured)",
                ha="center", va="center", transform=ax.transAxes)
    ax.grid(alpha=0.3)

    for ax in axes[3, :]:
        ax.set_xlabel("sim_time [s]")
    fig.suptitle(f"{args.device}: momentum observer vs MuJoCo contact forces")
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"\nwrote {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
