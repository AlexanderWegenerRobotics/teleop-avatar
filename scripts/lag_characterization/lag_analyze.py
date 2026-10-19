#!/usr/bin/env python3
"""Evaluate a lag_test.py run against the avatar arm log: step timing and ramp lag per axis."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AXES = "xyz"


def find_arm_logs(paths, device, t_start_ns):
    """Arm log CSVs of the device under the given files/dirs, written during or after the run."""
    files = []
    for p in paths:
        p = Path(p)
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            for f in p.rglob(f"{device}*.csv"):
                if f.name in (f"{device}.csv", f"{device}_log.csv"):
                    files.append(f)
    return sorted({f for f in files if f.stat().st_mtime * 1e9 >= t_start_ns - 60e9})


def load_arm_log(files, t0_ns, t1_ns):
    """Measured EE, interpolated reference and applied sequence from the arm log, within the run."""
    meas = [f"O_T_EE_world_{i}" for i in (12, 13, 14)]
    ref = [f"O_T_EE_cmd_world_{i}" for i in (12, 13, 14)]
    cols = ["wall_clock_ns", "sim_time", "applied_cmd_sequence", "state"] + meas + ref
    parts = []
    for f in files:
        header = f.open().readline().strip().split(";")
        if not set(cols) <= set(header):
            continue
        df = pd.read_csv(f, sep=";", usecols=cols, dtype={"wall_clock_ns": np.int64})
        df = df[(df.wall_clock_ns >= t0_ns - int(1e9)) & (df.wall_clock_ns <= t1_ns + int(1e9))]
        if len(df):
            parts.append(df)
    if not parts:
        return None
    df = pd.concat(parts).sort_values("wall_clock_ns").drop_duplicates("wall_clock_ns")
    return dict(t_ns=df.wall_clock_ns.to_numpy(np.int64), sim=df.sim_time.to_numpy(float),
                applied=df.applied_cmd_sequence.to_numpy(np.int64), p=df[meas].to_numpy(float),
                r=df[ref].to_numpy(float), source="arm log")


def load_state_stream(run_dir):
    """Fallback: measured EE and applied sequence from the 200 Hz state stream the script recorded."""
    df = pd.read_csv(run_dir / "states.csv").sort_values("sample_ns").drop_duplicates("sample_ns")
    df = df[df.sample_ns > 0]
    return dict(t_ns=df.sample_ns.to_numpy(np.int64), sim=None, applied=df.applied_seq.to_numpy(np.int64),
                p=df[["x", "y", "z"]].to_numpy(float), r=None, source="state stream")


def window_mean(t, y, a, b):
    """Mean of y over a <= t <= b, NaN when empty."""
    m = (t >= a) & (t <= b)
    return float(np.mean(y[m])) if m.any() else np.nan


def first_cross(t, s, level, t_from):
    """First time >= t_from where s reaches level, NaN if never."""
    m = (t >= t_from) & (s >= level)
    return float(t[m][0]) if m.any() else np.nan


def apply_delay(t, applied, seq, t_event):
    """Time from sending a command until the avatar reports it applied (s)."""
    m = applied >= seq
    return float(t[m][0] - t_event) if m.any() else np.nan


def step_metrics(t, y, t0, hold_s, a_cmd, target):
    """Dead time, t50, t90, settle time, overshoot and static error of one step on one axis."""
    p0 = window_mean(t, y, t0 - 0.1, t0)
    pf = window_mean(t, y, t0 + hold_s - 0.25, t0 + hold_s)
    out = dict(steady_err_mm=(pf - target) * 1000.0, reached=bool(abs(pf - p0) >= 0.3 * abs(a_cmd)))
    if not np.isfinite(p0) or not np.isfinite(pf) or abs(pf - p0) < 1e-6:
        return out
    m = (t >= t0) & (t <= t0 + hold_s)
    tt, s = t[m], (y[m] - p0) / (pf - p0)
    for name, level in (("dead_ms", 0.05), ("t50_ms", 0.5), ("t90_ms", 0.9)):
        out[name] = (first_cross(tt, s, level, t0) - t0) * 1000.0
    out["overshoot_pct"] = max(0.0, (float(np.max(s)) - 1.0) * 100.0)
    tol = max(0.0005, 0.02 * abs(a_cmd))
    bad = np.nonzero(np.abs(y[m] - pf) > tol)[0]
    if len(bad) == 0:
        out["settle_ms"] = 0.0
    elif bad[-1] + 1 < len(tt):
        out["settle_ms"] = (tt[bad[-1] + 1] - t0) * 1000.0
    return out


def ramp_metrics(t, y, ts, ramp_s, hold_s, x0, v):
    """Steady lag, tracking error, end overshoot and stop settle time of one ramp on one axis."""
    x1 = x0 + v * ramp_s
    b0 = window_mean(t, y - x0, ts - 0.1, ts)
    b1 = window_mean(t, y - x1, ts + ramp_s + hold_s - 0.25, ts + ramp_s + hold_s)
    out = dict(start_offset_mm=b0 * 1000.0, end_offset_mm=b1 * 1000.0)
    if not (np.isfinite(b0) and np.isfinite(b1)):
        return out
    m = (t >= ts + 0.4 * ramp_s) & (t <= ts + ramp_s)
    if m.sum() < 5:
        return out
    tt, yy = t[m], y[m]
    frac = np.clip((yy - b0 - x0) / (x1 - x0), 0.0, 1.0)
    yc = yy - (b0 + (b1 - b0) * frac)
    lag = (tt - ts) - (yc - x0) / v
    out["lag_ms"] = float(np.median(lag)) * 1000.0
    out["track_err_mm"] = float(np.median(np.abs(x0 + v * (tt - ts) - yc))) * 1000.0
    after = (t >= ts + ramp_s) & (t <= ts + ramp_s + hold_s)
    if after.any():
        ya = (y[after] - b1 - x1) * np.sign(v)
        out["end_overshoot_mm"] = max(0.0, float(np.max(ya)) * 1000.0)
        tol = 0.0005
        bad = np.nonzero(np.abs(ya) > tol)[0]
        ta = t[after]
        if len(bad) == 0:
            out["stop_settle_ms"] = 0.0
        elif bad[-1] + 1 < len(ta):
            out["stop_settle_ms"] = (ta[bad[-1] + 1] - (ts + ramp_s)) * 1000.0
    return out


def analyze(run_dir, data, trials, meta, cmds):
    """Per-trial metrics table."""
    t_ref = int(meta["t_start_ns"])
    t = (data["t_ns"] - t_ref) / 1e9
    origin = np.array(meta["origin_world"], float)
    rows = []
    for tr in trials:
        if "t_event_ns" not in tr:
            continue
        i = AXES.index(tr["axis"])
        t0 = (tr["t_event_ns"] - t_ref) / 1e9
        start = origin + np.array(tr["start"])
        end = origin + np.array(tr["end"])
        row = dict(trial=tr["id"], kind=tr["kind"], axis=tr["axis"], value=tr["value"], sign=tr["sign"],
                   point=tr["point"], rep=tr["rep"])
        row["apply_ms"] = apply_delay(t, data["applied"], tr["seq_event"], t0) * 1000.0
        if tr["kind"] == "step":
            a_cmd = end[i] - start[i]
            row.update(step_metrics(t, data["p"][:, i], t0, tr["hold_s"], a_cmd, end[i]))
            if data["r"] is not None:
                ref = step_metrics(t, data["r"][:, i], t0, tr["hold_s"], a_cmd, end[i])
                row["ref_t90_ms"] = ref.get("t90_ms", np.nan)
                row["ref_settle_ms"] = ref.get("settle_ms", np.nan)
        else:
            v = tr["sign"] * tr["value"] / 1000.0
            row.update(ramp_metrics(t, data["p"][:, i], t0, tr["ramp_s"], tr["hold_s"], start[i], v))
            if data["r"] is not None:
                ref = ramp_metrics(t, data["r"][:, i], t0, tr["ramp_s"], tr["hold_s"], start[i], v)
                row["ref_lag_ms"] = ref.get("lag_ms", np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df, cols, kind):
    """Median and interquartile range of the metrics, grouped by axis and amplitude/speed."""
    d = df[df.kind == kind]
    if d.empty:
        return pd.DataFrame()
    g = d.groupby(["axis", "value"])
    parts = {"n": g.size()}
    for c in cols:
        if c in d:
            parts[c] = g[c].median()
            parts[c + "_q25"] = g[c].quantile(0.25)
            parts[c + "_q75"] = g[c].quantile(0.75)
    return pd.DataFrame(parts).reset_index()


def health(data, cmds, meta):
    """Real-time factor of the sim and command timing over the run."""
    lines = []
    t_ns = data["t_ns"]
    if data["sim"] is not None and np.nanmax(data["sim"]) > 0:
        m = (t_ns >= meta["t_start_ns"]) & (t_ns <= meta["t_end_ns"])
        if m.sum() > 2:
            sim = data["sim"][m]
            wall = (t_ns[m][-1] - t_ns[m][0]) / 1e9
            rtf = (sim[-1] - sim[0]) / wall if wall > 0 else np.nan
            lines.append(f"sim real-time factor {rtf:.3f}" + ("  <-- below 0.98, times are inflated" if rtf < 0.98 else ""))
    dt = np.diff(cmds.t_send_ns.to_numpy(np.int64)) / 1e6
    if len(dt):
        lines.append(f"command interval median {np.median(dt):.2f} ms, p99 {np.percentile(dt, 99):.2f}, max {dt.max():.2f}")
    lines.append(f"measured from: {data['source']}")
    return lines


def plots(out, df, steps, ramps):
    """Step timing and ramp lag figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    colors = {"x": "#4285F4", "y": "#34A853", "z": "#EA4335"}
    if not steps.empty:
        fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
        for ax, metric, title in zip(axs, ("dead_ms", "t90_ms", "settle_ms"), ("dead time", "t90", "settle (±0.5 mm / 2 %)")):
            for a in AXES:
                s = steps[steps.axis == a]
                if s.empty or metric not in s:
                    continue
                ax.errorbar(s.value, s[metric], yerr=[s[metric] - s[metric + "_q25"], s[metric + "_q75"] - s[metric]],
                            marker="o", capsize=3, color=colors[a], label=a)
            ax.set_xscale("log")
            ax.set_xticks(sorted(steps.value.unique()))
            ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_xlabel("step size (mm)")
            ax.set_title(title)
            ax.grid(alpha=0.3)
        axs[0].set_ylabel("ms")
        axs[0].legend()
        fig.tight_layout()
        fig.savefig(out / "step_timing.png", dpi=150)
        plt.close(fig)
    if not ramps.empty and "lag_ms" in ramps:
        fig, ax = plt.subplots(figsize=(6, 3.8))
        for a in AXES:
            s = ramps[ramps.axis == a]
            if s.empty:
                continue
            ax.errorbar(s.value, s.lag_ms, yerr=[s.lag_ms - s.lag_ms_q25, s.lag_ms_q75 - s.lag_ms],
                        marker="o", capsize=3, color=colors[a], label=f"{a} measured")
            if "ref_lag_ms" in s:
                ax.plot(s.value, s.ref_lag_ms, "--", color=colors[a], alpha=0.6, label=f"{a} reference")
        ax.set_xscale("log")
        ax.set_xticks(sorted(ramps.value.unique()))
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_ylim(bottom=0)
        ax.set_xlabel("commanded speed (mm/s)")
        ax.set_ylabel("lag behind command (ms)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "ramp_lag.png", dpi=150)
        plt.close(fig)


def fmt(v, digits=0):
    """Number or dash."""
    return "-" if v is None or not np.isfinite(v) else f"{v:.{digits}f}"


def report(steps, ramps, df, owd_ms, video_ms, health_lines):
    """Plain-text summary."""
    out = ["# Lag characterization", ""] + health_lines + [""]
    if not steps.empty:
        out.append("## Steps (median over reps, ms)")
        out.append("axis  step_mm   n  apply  dead  t50   t90  settle  ref_t90  overshoot_%  static_err_mm")
        for _, r in steps.iterrows():
            out.append(f"{r.axis:>4}  {r.value:>7g}  {r.n:>2}  {fmt(r.get('apply_ms'), 1):>5}  {fmt(r.get('dead_ms')):>4}"
                       f"  {fmt(r.get('t50_ms')):>4}  {fmt(r.get('t90_ms')):>4}  {fmt(r.get('settle_ms')):>6}"
                       f"  {fmt(r.get('ref_t90_ms')):>7}  {fmt(r.get('overshoot_pct'), 1):>11}  {fmt(r.get('steady_err_mm'), 2):>13}")
        out.append("")
    if not ramps.empty:
        out.append("## Ramps (median over reps)")
        out.append("axis  v_mm_s   n  lag_ms  ref_lag_ms  track_err_mm  end_overshoot_mm  stop_settle_ms")
        for _, r in ramps.iterrows():
            out.append(f"{r.axis:>4}  {r.value:>6g}  {r.n:>2}  {fmt(r.get('lag_ms')):>6}  {fmt(r.get('ref_lag_ms')):>10}"
                       f"  {fmt(r.get('track_err_mm'), 2):>12}  {fmt(r.get('end_overshoot_mm'), 2):>16}"
                       f"  {fmt(r.get('stop_settle_ms')):>14}")
        out.append("")
        apply_ms = float(np.nanmedian(df.apply_ms)) if df.apply_ms.notna().any() else 0.0
        out.append(f"## Movement to eye during continuous motion (OWD {owd_ms:g} ms each way, video {video_ms:g} ms, apply {apply_ms:.1f} ms)")
        out.append("axis  v_mm_s  total_ms")
        for _, r in ramps.iterrows():
            lag = r.get("lag_ms")
            if lag is not None and np.isfinite(lag):
                out.append(f"{r.axis:>4}  {r.value:>6g}  {owd_ms + apply_ms + lag + video_ms + owd_ms:>8.0f}")
    return "\n".join(out) + "\n"


def main():
    """Load a run, compute metrics, write tables, plots and a summary."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--arm-log", nargs="*", default=None,
                    help="arm log files or directories (default: <repo>/logs and <repo>/log)")
    ap.add_argument("--state-stream", action="store_true", help="use the recorded 200 Hz state stream instead of the arm log")
    ap.add_argument("--owd-ms", type=float, default=98.0, help="one-way network delay added for the end-to-end table")
    ap.add_argument("--video-ms", type=float, default=63.0, help="local video glass-to-glass added for the end-to-end table")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    meta = json.loads((run_dir / "meta.json").read_text())
    trials = json.loads((run_dir / "trials.json").read_text())
    cmds = pd.read_csv(run_dir / "commands.csv")
    if "origin_world" not in meta:
        sys.exit("run has no origin (it stopped before engaging)")

    data = None
    if not args.state_stream:
        repo = Path(__file__).resolve().parents[2]
        paths = args.arm_log or [repo / "logs", repo / "log"]
        files = find_arm_logs(paths, meta["device"], meta["t_start_ns"])
        data = load_arm_log(files, meta["t_start_ns"], meta["t_end_ns"])
        if data is None:
            print("no arm log rows for this run, falling back to the state stream")
    if data is None:
        data = load_state_stream(run_dir)

    df = analyze(run_dir, data, trials, meta, cmds)
    df.to_csv(run_dir / "trial_metrics.csv", index=False)
    steps = summarize(df, ["apply_ms", "dead_ms", "t50_ms", "t90_ms", "settle_ms", "ref_t90_ms",
                           "overshoot_pct", "steady_err_mm"], "step")
    ramps = summarize(df, ["apply_ms", "lag_ms", "ref_lag_ms", "track_err_mm", "end_overshoot_mm",
                           "stop_settle_ms"], "ramp")
    steps.to_csv(run_dir / "summary_steps.csv", index=False)
    ramps.to_csv(run_dir / "summary_ramps.csv", index=False)
    plots(run_dir, df, steps, ramps)
    text = report(steps, ramps, df, args.owd_ms, args.video_ms, health(data, cmds, meta))
    (run_dir / "summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
