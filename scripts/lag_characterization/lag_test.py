#!/usr/bin/env python3
"""Drive the avatar arm with step and ramp commands and log them for the lag characterization."""

import argparse
import csv
import json
import math
import random
import select
import socket
import struct
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import msgpack
import yaml

CMD_FMT = "<IQQBBB3f4ffB"
STATE_FMT = "<IQQBBB3f4f7f7fBfBIII6f"
CMD_SIZE = struct.calcsize(CMD_FMT)
STATE_SIZE = struct.calcsize(STATE_FMT)
assert CMD_SIZE == 56 and STATE_SIZE == 149

STATE = {"OFFLINE": 0, "IDLE": 1, "HOMING": 2, "AWAITING": 3, "ENGAGED": 4,
         "PAUSED": 5, "FAULT": 6, "STOP": 7, "RECOVERING": 8}
STATE_NAME = {v: k for k, v in STATE.items()}
DEVICE_ID = {"arm_left": 1, "arm_right": 2}
AXIS = {"x": 0, "y": 1, "z": 2}

if sys.platform == "win32":
    import ctypes

    ctypes.windll.winmm.timeBeginPeriod(1)

    def now_ns():
        """Wall clock in ns from GetSystemTimePreciseAsFileTime, the source of std::chrono::system_clock."""
        ft = ctypes.c_ulonglong()
        ctypes.windll.kernel32.GetSystemTimePreciseAsFileTime(ctypes.byref(ft))
        return (ft.value - 116444736000000000) * 100
else:
    def now_ns():
        """Wall clock in ns."""
        return time.time_ns()


class Segment:
    """Piece of the command profile: linear move from a to b over a duration, optionally a measured trial."""

    def __init__(self, a, b, duration_s, trial=None):
        self.a = a
        self.b = b
        self.duration_s = max(duration_s, 0.0)
        self.trial = trial

    def at(self, t):
        """Commanded delta (m) at time t into the segment."""
        if self.duration_s <= 0.0:
            return self.b
        s = min(max(t / self.duration_s, 0.0), 1.0)
        return tuple(self.a[i] + s * (self.b[i] - self.a[i]) for i in range(3))


def hold(p, duration_s):
    """Segment that holds p."""
    return Segment(p, p, duration_s)


def move(a, b, speed_m_s, trial=None):
    """Segment that moves from a to b at constant speed."""
    dist = math.dist(a, b)
    return Segment(a, b, dist / speed_m_s if speed_m_s > 0 else 0.0, trial)


def offset(p, axis, d):
    """p shifted by d along a world axis."""
    q = list(p)
    q[AXIS[axis]] += d
    return tuple(q)


def ramp_distance(cfg, v_mm_s):
    """Ramp distance in mm for a speed, from the configured duration and limits."""
    r = cfg["ramps"]
    return min(max(v_mm_s * r["duration_s"], r["min_distance_mm"]), r["max_distance_mm"])


def build_plan(cfg):
    """Full command profile as a list of segments, plus the trial list."""
    rng = random.Random(cfg["seed"])
    transit = cfg["transit_speed_mm_s"] / 1000.0
    segments = []
    trials = []
    here = (0.0, 0.0, 0.0)
    for pi, point_mm in enumerate(cfg["points_mm"]):
        base = tuple(v / 1000.0 for v in point_mm)
        segments.append(move(here, base, transit))
        segments.append(hold(base, cfg["settle_s"]))
        here = base
        for rep in range(cfg["reps"]):
            blocks = []
            for axis in cfg["axes"]:
                for a_mm in cfg["steps"]["amplitudes_mm"]:
                    blocks.append(("step", axis, a_mm))
                for v_mm_s in cfg["ramps"]["speeds_mm_s"]:
                    blocks.append(("ramp", axis, v_mm_s))
            rng.shuffle(blocks)
            for kind, axis, value in blocks:
                sign = rng.choice((1.0, -1.0))
                if kind == "step":
                    far = offset(base, axis, sign * value / 1000.0)
                    for direction, (a, b) in ((1, (base, far)), (-1, (far, base))):
                        trial = dict(id=len(trials), kind="step", axis=axis, value=value, sign=sign * direction,
                                     point=pi, rep=rep, start=a, end=b, hold_s=cfg["steps"]["hold_s"])
                        trials.append(trial)
                        segments.append(Segment(a, b, 0.0, trial))
                        segments.append(hold(b, cfg["steps"]["hold_s"]))
                else:
                    dist = ramp_distance(cfg, value) / 1000.0
                    far = offset(base, axis, sign * dist)
                    for direction, (a, b) in ((1, (base, far)), (-1, (far, base))):
                        trial = dict(id=len(trials), kind="ramp", axis=axis, value=value, sign=sign * direction,
                                     point=pi, rep=rep, start=a, end=b, hold_s=cfg["ramps"]["hold_s"],
                                     ramp_s=dist / (value / 1000.0))
                        trials.append(trial)
                        segments.append(move(a, b, value / 1000.0, trial))
                        segments.append(hold(b, cfg["ramps"]["hold_s"]))
    segments.append(move(here, (0.0, 0.0, 0.0), transit))
    segments.append(hold((0.0, 0.0, 0.0), 1.0))
    return segments, trials


def plan_extent(cfg):
    """Largest excursion (m) of the plan per axis, both directions, relative to the engage pose."""
    lo = [0.0, 0.0, 0.0]
    hi = [0.0, 0.0, 0.0]
    reach = max(max(cfg["steps"]["amplitudes_mm"]),
                max(ramp_distance(cfg, v) for v in cfg["ramps"]["speeds_mm_s"])) / 1000.0
    for point_mm in cfg["points_mm"]:
        for i in range(3):
            c = point_mm[i] / 1000.0
            extra = reach if "xyz"[i] in cfg["axes"] else 0.0
            lo[i] = min(lo[i], c - extra)
            hi[i] = max(hi[i], c + extra)
    return lo, hi


class Avatar:
    """UDP link to the avatar: reliable command channel plus the arm command/state stream."""

    def __init__(self, ip, avatar_ports, arm_ports, device_id):
        self.ip = ip
        self.avatar_send = avatar_ports["receive_port"]
        self.arm_send = arm_ports["receive_port"]
        self.device_id = device_id
        self.rel = self._bind(avatar_ports["send_port"])
        self.arm = self._bind(arm_ports["send_port"])
        self.seq = self._seq_base()
        self.rel_seq = 0
        self.state = None
        self.avatar_state = None
        self.state_pos = None
        self.applied_seq = 0
        self.state_rows = []
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._recv_loop, daemon=True)
        self.thread.start()

    @staticmethod
    def _bind(port):
        """UDP socket bound to a local port, with a readable error when the port is taken."""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.bind(("0.0.0.0", port))
        except OSError as e:
            sys.exit(f"cannot bind UDP port {port} ({e}). Close the VR interface / arm_tester first.")
        s.setblocking(False)
        return s

    @staticmethod
    def _seq_base():
        """Command sequence start that grows with wall time, so a rerun is never rejected as old."""
        return (int(time.time()) - 1_790_000_000) * 200

    def _recv_loop(self):
        """Read arm state packets and drain the reliable channel."""
        while self.running:
            r, _, _ = select.select([self.arm, self.rel], [], [], 0.05)
            for s in r:
                while True:
                    try:
                        data, _ = s.recvfrom(4096)
                    except OSError:
                        break
                    if s is self.arm and len(data) == STATE_SIZE:
                        self._on_state(data)
                    elif s is self.rel:
                        self._on_envelope(data)

    def _on_state(self, data):
        """Store one ArmStateMsg."""
        f = struct.unpack(STATE_FMT, data)
        t_recv = now_ns()
        with self.lock:
            self.state = f[3]
            self.state_pos = f[6:9]
            self.applied_seq = f[30]
            self.state_rows.append((t_recv, f[1], f[2], f[3], f[6], f[7], f[8], f[30]))

    def _on_envelope(self, data):
        """Track the avatar system state from its heartbeats and acks."""
        try:
            env = msgpack.unpackb(data, raw=False, strict_map_key=False)
        except Exception:
            return
        if isinstance(env, dict) and "state" in env:
            with self.lock:
                self.avatar_state = env["state"]

    def snapshot(self):
        """Latest arm state and world position."""
        with self.lock:
            return self.state, self.state_pos

    def send_reliable(self, msg_type, payload, state):
        """Send one msgpack envelope on the avatar command channel."""
        self.rel_seq += 1
        env = {"sequence": self.rel_seq, "timestamp_ns": now_ns(), "state": state, "fault_code": 0,
               "msg_type": msg_type, "ack_requested": False, "payload": payload}
        self.rel.sendto(msgpack.packb(env, use_bin_type=True), (self.ip, self.avatar_send))

    def request_state(self, name, current):
        """Ask the avatar for a system state."""
        self.send_reliable("state_change", {"requested_state": STATE[name]}, current)

    def heartbeat(self, current, uptime_ms):
        """Keep the avatar's operator link alive."""
        self.send_reliable("heartbeat", {"uptime_ms": uptime_ms}, current)

    def send_cmd(self, delta, state):
        """Send one delta-from-origin pose command, returns (sequence, send time)."""
        self.seq += 1
        t = now_ns()
        msg = struct.pack(CMD_FMT, self.seq, t, t, state, 0, self.device_id,
                          delta[0], delta[1], delta[2], 1.0, 0.0, 0.0, 0.0, 0.0, 0)
        self.arm.sendto(msg, (self.ip, self.arm_send))
        return self.seq, t

    def close(self):
        """Stop the receiver."""
        self.running = False
        self.thread.join(timeout=1.0)


class Runner:
    """Fixed-rate command loop with heartbeats and logging."""

    def __init__(self, avatar, rate_hz, writer):
        self.av = avatar
        self.period_ns = int(1e9 / rate_hz)
        self.rate_hz = rate_hz
        self.writer = writer
        self.next_tick = time.perf_counter_ns()
        self.last_hb = 0
        self.t0 = time.monotonic()
        self.sys_state = STATE["IDLE"]
        self.segment_id = -1

    def _wait_tick(self):
        """Sleep until the next tick, spin the last millisecond."""
        while True:
            dt = self.next_tick - time.perf_counter_ns()
            if dt <= 0:
                break
            if dt > 3_000_000:
                time.sleep((dt - 2_000_000) / 1e9)
        self.next_tick += self.period_ns
        if time.perf_counter_ns() - self.next_tick > 50 * self.period_ns:
            self.next_tick = time.perf_counter_ns()

    def tick(self, delta, trial_id=-1):
        """Send one command and heartbeat if due."""
        self._wait_tick()
        seq, t = self.av.send_cmd(delta, self.sys_state)
        self.writer.writerow((seq, t, self.segment_id, trial_id,
                              f"{delta[0]:.6f}", f"{delta[1]:.6f}", f"{delta[2]:.6f}"))
        if time.monotonic() - self.last_hb >= 0.5:
            self.last_hb = time.monotonic()
            self.av.heartbeat(self.sys_state, int((self.last_hb - self.t0) * 1000))
        return seq, t

    def wait_for(self, predicate, timeout_s, every=None, delta=(0.0, 0.0, 0.0)):
        """Keep ticking until predicate() is true, calling every() once a second; False on timeout."""
        end = time.monotonic() + timeout_s
        last = 0.0
        while time.monotonic() < end:
            if every is not None and time.monotonic() - last >= 1.0:
                last = time.monotonic()
                every()
            self.tick(delta)
            if predicate():
                return True
        return False

    def run_segment(self, seg):
        """Play one segment; returns the first tick's (sequence, send time)."""
        self.segment_id += 1
        n = max(1, int(math.ceil(seg.duration_s * self.rate_hz)))
        tid = seg.trial["id"] if seg.trial else -1
        first = None
        for k in range(n + (1 if seg.duration_s > 0 else 0)):
            delta = seg.at(k / self.rate_hz) if seg.duration_s > 0 else seg.b
            st = self.tick(delta, tid)
            if first is None:
                first = st
            if seg.duration_s <= 0:
                break
        return first


def load_ports(robot_cfg, device):
    """Avatar and arm UDP ports, workspace and table limits from the robot config."""
    avatar = robot_cfg["avatar"]["transmission"]
    dev = next((d for d in robot_cfg["devices"] if d["name"] == device), None)
    if dev is None:
        sys.exit(f"device {device} not in robot config")
    s = dev["safety"]
    z_min = s["table_height_world"] + s["table_safety_margin"] + s["ee_fingertip_length"]
    return avatar, dev["transmission"], s["workspace_min"], s["workspace_max"], z_min


def check_workspace(origin, lo, hi, ws_min, ws_max, z_min):
    """List of problems if the plan leaves the workspace or gets too close to the table."""
    problems = []
    for i, ax in enumerate("xyz"):
        if origin[i] + lo[i] < ws_min[i] or origin[i] + hi[i] > ws_max[i]:
            problems.append(f"{ax}: {origin[i] + lo[i]:.3f}..{origin[i] + hi[i]:.3f} outside "
                            f"{ws_min[i]:.3f}..{ws_max[i]:.3f}")
    if origin[2] + lo[2] < z_min:
        problems.append(f"z: lowest point {origin[2] + lo[2]:.3f} below table limit {z_min:.3f}")
    return problems


def main():
    """Engage the arm, play the plan, write the logs."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(Path(__file__).with_name("lag_test.yaml")))
    ap.add_argument("--label", default="sim", help="free text stored with the run, e.g. 'sim desk 800/40'")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text())
    segments, trials = build_plan(cfg)
    total_s = sum(s.duration_s for s in segments) + len(segments) / cfg["rate_hz"]
    print(f"plan: {len(trials)} trials, {len(segments)} segments, ~{total_s / 60:.1f} min")
    if args.dry_run:
        return

    robot_path = (cfg_path.parent / cfg["robot_config"]).resolve()
    robot_cfg = yaml.safe_load(robot_path.read_text())
    avatar_ports, arm_ports, ws_min, ws_max, z_min = load_ports(robot_cfg, cfg["device"])
    z_min += cfg["table_clearance_mm"] / 1000.0

    run_dir = (cfg_path.parent / cfg["output_dir"] /
               (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + "".join(c if c.isalnum() or c in "-_." else "_" for c in args.label))).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    av = Avatar(cfg["avatar_ip"], avatar_ports, arm_ports, DEVICE_ID[cfg["device"]])
    cmd_file = open(run_dir / "commands.csv", "w", newline="")
    writer = csv.writer(cmd_file)
    writer.writerow(("seq", "t_send_ns", "segment", "trial", "dx", "dy", "dz"))
    runner = Runner(av, cfg["rate_hz"], writer)
    meta = dict(label=args.label, config=cfg, robot_config=str(robot_path), device=cfg["device"],
                rate_hz=cfg["rate_hz"], t_start_ns=now_ns(), python=sys.version, platform=sys.platform)
    status = "aborted"
    timeout = cfg["state_timeout_s"]

    def arm_in(name):
        st = av.snapshot()[0]
        if name == "IDLE":
            return st in (STATE["OFFLINE"], STATE["IDLE"])
        return st == STATE[name]

    try:
        if not runner.wait_for(lambda: av.snapshot()[0] is not None, 5.0):
            raise RuntimeError("no arm state from the avatar, is it running with this robot config?")
        runner.wait_for(lambda: av.avatar_state is not None, 2.0)
        print(f"arm state: {STATE_NAME.get(av.snapshot()[0])} (OFFLINE = idle since start), "
              f"avatar state: {STATE_NAME.get(av.avatar_state, 'no heartbeat yet')}")
        if not arm_in("IDLE"):
            if not runner.wait_for(lambda: arm_in("IDLE"), timeout,
                                   every=lambda: av.request_state("IDLE", runner.sys_state)):
                raise RuntimeError("arm did not go to IDLE")
        runner.sys_state = STATE["HOMING"]
        print("homing ...")
        if not runner.wait_for(lambda: arm_in("AWAITING"), timeout,
                               every=lambda: av.request_state("HOMING", runner.sys_state)):
            raise RuntimeError(f"arm did not reach AWAITING after homing (arm {STATE_NAME.get(av.snapshot()[0])}, "
                               f"avatar {STATE_NAME.get(av.avatar_state)})")
        runner.sys_state = STATE["AWAITING"]
        runner.wait_for(lambda: False, 1.0)
        if not runner.wait_for(lambda: arm_in("ENGAGED"), 10.0,
                               every=lambda: av.request_state("ENGAGED", runner.sys_state)):
            raise RuntimeError("arm did not engage")
        runner.sys_state = STATE["ENGAGED"]
        runner.wait_for(lambda: False, 0.5)
        origin = av.snapshot()[1]
        meta["origin_world"] = list(origin)
        print(f"engaged, origin {origin[0]:.3f} {origin[1]:.3f} {origin[2]:.3f}")

        lo, hi = plan_extent(cfg)
        problems = check_workspace(origin, lo, hi, ws_min, ws_max, z_min)
        if problems:
            raise RuntimeError("plan leaves the safe workspace: " + "; ".join(problems))

        t_begin = time.monotonic()
        done = 0
        for seg in segments:
            seq, t = runner.run_segment(seg)
            if seg.trial is not None:
                seg.trial["seq_event"] = seq
                seg.trial["t_event_ns"] = t
                done += 1
                if done % 20 == 0:
                    el = time.monotonic() - t_begin
                    print(f"  {done}/{len(trials)} trials, {el / 60:.1f} min")
            if not arm_in("ENGAGED"):
                raise RuntimeError(f"arm left ENGAGED ({STATE_NAME.get(av.snapshot()[0])})")
        status = "complete"
    except KeyboardInterrupt:
        print("interrupted")
    except RuntimeError as e:
        print(f"stopped: {e}")
    finally:
        end = cfg["end_state"].upper()
        av.request_state(end, runner.sys_state)
        runner.sys_state = STATE[end]
        runner.wait_for(lambda: False, 0.5)
        meta["t_end_ns"] = now_ns()
        meta["status"] = status
        cmd_file.close()
        av.close()
        with open(run_dir / "states.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(("t_recv_ns", "t_send_ns", "sample_ns", "state", "x", "y", "z", "applied_seq"))
            w.writerows(av.state_rows)
        (run_dir / "trials.json").write_text(json.dumps(trials, indent=1))
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=1))
        print(f"{status}, logs in {run_dir}")


if __name__ == "__main__":
    main()
