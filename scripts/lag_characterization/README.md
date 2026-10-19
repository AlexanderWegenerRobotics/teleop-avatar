# Lag characterization

Measures how long the arm takes to follow a command, as a function of step size and speed, on one clock.
`lag_test.py` sends delta-from-origin pose commands to the avatar exactly like the VR interface (same UDP
stream, 120 Hz), and `lag_analyze.py` joins them with the avatar's own arm log via `applied_cmd_sequence`
and `wall_clock_ns`. Nothing in the avatar changes.

## Run

1. Start the avatar (sim: `launch.bat`) with the robot config named in `lag_test.yaml`. Close the VR interface,
   it uses the same ports.
2. Check the plan length: `python lag_test.py --dry-run` (default plan: 390 trials, ~16 min).
3. Run it:

```
python lag_test.py --label "sim desk 800/40"
```

The script switches the arm IDLE → HOMING → ENGAGED, checks that the whole plan stays inside the robot
config's workspace and above the table, plays the trials and requests IDLE at the end (also on Ctrl+C or
when the arm leaves ENGAGED). Logs go to `runs/<date>_<label>/`.

4. Evaluate:

```
python lag_analyze.py runs/<date>_<label>
```

It looks for the arm log under `<repo>/logs` and `<repo>/log` (pass `--arm-log <file or dir>` otherwise). Without an
arm log it falls back to the 200 Hz state stream the script recorded (`--state-stream`), which has no reference
signal. `--owd-ms` and `--video-ms` set the numbers used for the end-to-end table (default 98 and 63).

Needs `numpy pandas matplotlib msgpack pyyaml`.

## What is measured

| Trial | Metric | Meaning |
|---|---|---|
| step | `apply_ms` | command sent → avatar applied it (UDP + receive polling) |
| step | `dead_ms`, `t50_ms`, `t90_ms` | send → 5 / 50 / 90 % of the measured motion |
| step | `settle_ms` | send → stays within ±0.5 mm or 2 % of the step (for 1–2 mm steps the 0.5 mm band dominates) |
| step | `ref_t90_ms` | same for the interpolated reference, i.e. the shaping before the controller |
| step | `steady_err_mm` | static offset at the end of the hold (posture leakage, gravity) |
| ramp | `lag_ms` | steady time lag behind a constant-velocity command, static offset removed |
| ramp | `ref_lag_ms` | lag of the interpolated reference alone |
| ramp | `track_err_mm`, `end_overshoot_mm`, `stop_settle_ms` | position error while moving, overshoot and settle after the ramp stops |

Steps above `max_command_velocity × command interval` (0.7 m/s × 8.3 ms ≈ 6 mm) are rate limited by the avatar
and arrive as a fast ramp; that is the real system behaviour and part of the result.

## Outputs per run

`commands.csv`, `states.csv`, `trials.json`, `meta.json` from the test; `trial_metrics.csv`, `summary_steps.csv`,
`summary_ramps.csv`, `summary.txt`, `step_timing.png`, `ramp_lag.png` from the analysis.

## Notes

- The sim has to run in real time; the summary prints the real-time factor from `sim_time`. Below 0.98 every time is inflated.
- The sim is a lower bound for the real arm: no stiction, torque filter or actuation delay. Small, slow motions are where the difference is largest.
- On the real arm run the script on the NUC itself (avatar on localhost), so it stays one clock.
- Network delay is added afterwards (`--owd-ms`); the command path has no feedback loop, so it adds directly.
