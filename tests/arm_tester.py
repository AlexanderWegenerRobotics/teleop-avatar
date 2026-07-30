#!/usr/bin/env python3
"""
arm_tester.py — interactive pose command tool for avatar system

Avatar cmd channel : UdpReliable (msgpack envelopes, heartbeat)
Arm channels       : UdpStream   (raw packed structs)

State machine keys : h=home  g=engage  p=pause  b=idle  x=stop
Arm select         : 1=left  2=right  3=both
Pose nudge         : w/s=+/-X  a/d=+/-Y  q/e=+/-Z
Orientation nudge  : I/K=+/-Rx  J/L=+/-Ry  U/O=+/-Rz  (shift)
Step size          : +/-  to increase/decrease nudge step
Gripper            : f=open  v=close
Reset arm pose     : r
Reset left arm     : z
Reset right arm    : X  (shift+x)
Reset both arms    : c
Resume left arm    : n  (only when awaiting resume)
Resume right arm   : m  (only when awaiting resume)
Quit               : ESC or ctrl-c
"""

import socket
import struct
import time
import math
import threading
import curses
import msgpack
import sys

# ── Config ────────────────────────────────────────────────────────────────────

AVATAR_IP = "10.112.24.173"
AVATAR_IP = "127.0.0.1"

AVATAR_SEND_PORT    = 7000
AVATAR_RECEIVE_PORT = 8000

ARM_LEFT_SEND_PORT    = 7001
ARM_LEFT_RECEIVE_PORT = 8001

ARM_RIGHT_SEND_PORT    = 7002
ARM_RIGHT_RECEIVE_PORT = 8002

HEARTBEAT_INTERVAL_S = 0.5
ARM_CMD_RATE_HZ      = 50

# ── SysState ──────────────────────────────────────────────────────────────────

class SysState:
    OFFLINE  = 0
    IDLE     = 1
    HOMING   = 2
    AWAITING = 3
    ENGAGED  = 4
    PAUSED   = 5
    FAULT    = 6
    STOP     = 7

STATE_NAMES = {v: k for k, v in vars(SysState).items() if not k.startswith('_')}

class ResetState:
    IDLE           = 0
    RECOVERING     = 1
    AWAITING_RESUME = 2

RESET_STATE_NAMES = {
    ResetState.IDLE:            "idle",
    ResetState.RECOVERING:      "resetting...",
    ResetState.AWAITING_RESUME: "RESUME READY",
}

# ── Message formats ───────────────────────────────────────────────────────────

# MsgHeader: uint32 seq, uint64 ts, uint8 state, uint8 fault, uint8 device_id = 15 bytes
# ArmCommandMsg: header + 3f pos + 4f quat + 1f gripper = 47 bytes
# ArmStateMsg:   header + 3f pos + 4f quat + 7f joints + 7f tau_ext + 1B recovering
#                + 1f gripper_width + 1B grasp_confirmed = 105 bytes
ARM_CMD_FMT  = '<IQ3B3f4ff'
ARM_CMD_SIZE = struct.calcsize(ARM_CMD_FMT)

ARM_STATE_FMT  = '<IQ3B3f4f7f7fBfB'
ARM_STATE_SIZE = struct.calcsize(ARM_STATE_FMT)

# ── Math helpers ──────────────────────────────────────────────────────────────

def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    )

def quat_from_axis_angle(axis, angle_rad):
    s = math.sin(angle_rad / 2)
    c = math.cos(angle_rad / 2)
    ax, ay, az = axis
    n = math.sqrt(ax*ax + ay*ay + az*az)
    if n < 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    return (c, ax/n*s, ay/n*s, az/n*s)

def quat_normalize(q):
    w, x, y, z = q
    n = math.sqrt(w*w + x*x + y*y + z*z)
    if n < 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    return (w/n, x/n, y/n, z/n)

# ── UdpReliable envelope (avatar cmd channel) ─────────────────────────────────

_send_seq = 0
_send_seq_lock = threading.Lock()

def _next_seq():
    global _send_seq
    with _send_seq_lock:
        _send_seq += 1
        return _send_seq

def _now_ns():
    return int(time.time() * 1e9)

def pack_reliable_envelope(msg_type, payload_dict, state, ack_requested=False):
    seq = _next_seq()
    envelope = {
        "sequence":      seq,
        "timestamp_ns":  _now_ns(),
        "state":         state,
        "fault_code":    0,
        "msg_type":      msg_type,
        "ack_requested": ack_requested,
        "payload":       payload_dict,
    }
    return msgpack.packb(envelope, use_bin_type=True), seq

def send_state_change(sock, requested_state, current_state):
    payload = {"requested_state": requested_state}
    data, _ = pack_reliable_envelope("state_change", payload, current_state, ack_requested=False)
    sock.sendto(data, (AVATAR_IP, AVATAR_SEND_PORT))

def send_heartbeat(sock, current_state, uptime_ms):
    payload = {"uptime_ms": uptime_ms}
    data, _ = pack_reliable_envelope("heartbeat", payload, current_state, ack_requested=False)
    sock.sendto(data, (AVATAR_IP, AVATAR_SEND_PORT))

def send_arm_reset(sock, device_name, current_state):
    payload = {"device": device_name}
    data, _ = pack_reliable_envelope("arm_reset", payload, current_state, ack_requested=True)
    sock.sendto(data, (AVATAR_IP, AVATAR_SEND_PORT))

def send_arm_resume(sock, device_name, current_state):
    payload = {"device": device_name}
    data, _ = pack_reliable_envelope("arm_resume", payload, current_state, ack_requested=True)
    sock.sendto(data, (AVATAR_IP, AVATAR_SEND_PORT))

# ── Arm struct helpers ────────────────────────────────────────────────────────

def send_arm_cmd(sock, send_port, device_id, state, position, quaternion, gripper):
    w, x, y, z = quaternion
    px, py, pz = position
    msg = struct.pack(ARM_CMD_FMT,
        0, _now_ns(), state, 0, device_id,
        px, py, pz,
        w, x, y, z,
        gripper)
    sock.sendto(msg, (AVATAR_IP, send_port))

def parse_arm_state(data):
    if len(data) != ARM_STATE_SIZE:
        return None
    fields = struct.unpack(ARM_STATE_FMT, data)
    # seq, ts, state, fault, device_id, px, py, pz, qw, qx, qy, qz, j0-6, tau0-6,
    # recovering, gripper_width, grasp_confirmed
    return {
        "device_id": fields[4],
        "state":     fields[2],
        "pos":       fields[5:8],
        "quat":      fields[8:12],
        "joints":    fields[12:19],
        "tau_ext":   fields[19:26],
        "recovering": fields[26],
        "gripper_width": fields[27],
        "grasp_confirmed": fields[28],
    }

# ── Shared state ──────────────────────────────────────────────────────────────

class AppState:
    def __init__(self):
        self.lock = threading.Lock()

        self.avatar_state    = SysState.OFFLINE
        self.arm_left_state  = None
        self.arm_right_state = None

        self.target_arm = 3

        self.pos   = [0.0, 0.0, 0.0]
        self.quat  = (1.0, 0.0, 0.0, 0.0)

        self.gripper      = 0.0
        self.step_gripper = 0.1

        self.step_pos = 0.01
        self.step_ang = math.radians(2.0)

        self.requested_state = SysState.IDLE
        self.running = True

        self.left_reset_state  = ResetState.IDLE
        self.right_reset_state = ResetState.IDLE

        self.log_lines = []
        self.start_time = time.time()

    def log(self, msg):
        with self.lock:
            self.log_lines.append(msg)
            if len(self.log_lines) > 20:
                self.log_lines.pop(0)

    def uptime_ms(self):
        return int((time.time() - self.start_time) * 1000)

# ── Background threads ────────────────────────────────────────────────────────

def avatar_recv_thread(sock, app):
    while app.running:
        try:
            data, _ = sock.recvfrom(65536)
            envelope = msgpack.unpackb(data, raw=False)
            state_byte = envelope.get("state", SysState.OFFLINE)
            msg_type   = envelope.get("msg_type", "")
            with app.lock:
                app.avatar_state = state_byte

            if msg_type == "device_event":
                payload = envelope.get("payload", {})
                device = payload.get("device", "")
                event  = payload.get("event", "")
                if event == "reset_complete":
                    with app.lock:
                        if device == "arm_left":
                            app.left_reset_state = ResetState.AWAITING_RESUME
                        elif device == "arm_right":
                            app.right_reset_state = ResetState.AWAITING_RESUME
                    app.log(f"[{device}] reset done — press resume key")

        except Exception:
            pass

def arm_recv_thread(sock, key, app):
    while app.running:
        try:
            data, _ = sock.recvfrom(ARM_STATE_SIZE + 64)
            parsed = parse_arm_state(data)
            if parsed:
                with app.lock:
                    setattr(app, key, parsed)
        except Exception:
            pass

def heartbeat_thread(avatar_sock, app):
    while app.running:
        with app.lock:
            cs = app.avatar_state
            up = app.uptime_ms()
        send_heartbeat(avatar_sock, cs, up)
        time.sleep(HEARTBEAT_INTERVAL_S)

def arm_cmd_thread(arm_left_sock, arm_right_sock, app):
    period = 1.0 / ARM_CMD_RATE_HZ
    while app.running:
        t0 = time.time()
        with app.lock:
            av_state = app.avatar_state
            pos      = list(app.pos)
            quat     = app.quat
            target   = app.target_arm
            gripper  = app.gripper

        if av_state == SysState.ENGAGED:
            if target in (1, 3):
                send_arm_cmd(arm_left_sock,  ARM_LEFT_SEND_PORT,  0, SysState.ENGAGED, pos, quat, gripper)
            if target in (2, 3):
                send_arm_cmd(arm_right_sock, ARM_RIGHT_SEND_PORT, 1, SysState.ENGAGED, pos, quat, gripper)

        elapsed = time.time() - t0
        remaining = period - elapsed
        if remaining > 0:
            time.sleep(remaining)

# ── TUI ───────────────────────────────────────────────────────────────────────

HELP = [
    "STATE: h=home  g=engage  p=pause  b=idle  x=stop",
    "ARM:   1=left  2=right   3=both",
    "MOVE:  w/s=+/-X  a/d=+/-Y  q/e=+/-Z",
    "ROT:   I/K=+/-Rx  J/L=+/-Ry  U/O=+/-Rz  (shift)",
    "GRIP:  f=open  v=close",
    "       +/-=step   r=reset pose   ESC=quit",
    "RESET: z=reset L  X=reset R  c=reset both",
    "       n=resume L  m=resume R  (when ready)",
]

def draw(stdscr, app):
    stdscr.erase()
    h, w = stdscr.getmaxyx()
    row = 0

    def put(text, attr=0):
        nonlocal row
        if row < h:
            try:
                stdscr.addstr(row, 0, text[:w], attr)
            except curses.error:
                pass
            row += 1

    with app.lock:
        av              = app.avatar_state
        al              = app.arm_left_state
        ar              = app.arm_right_state
        pos             = list(app.pos)
        quat            = app.quat
        step            = app.step_pos
        sanga           = math.degrees(app.step_ang)
        tgt             = app.target_arm
        l_reset         = app.left_reset_state
        r_reset         = app.right_reset_state

    put("── Avatar Arm Tester ──────────────────────────", curses.A_BOLD)
    put(f"  Avatar state : {STATE_NAMES.get(av, '?')}")

    al_str   = STATE_NAMES.get(al['state'], '?') if al else '—'
    ar_str   = STATE_NAMES.get(ar['state'], '?') if ar else '—'
    al_pos   = f"({al['pos'][0]:.3f}, {al['pos'][1]:.3f}, {al['pos'][2]:.3f})" if al else '—'
    ar_pos   = f"({ar['pos'][0]:.3f}, {ar['pos'][1]:.3f}, {ar['pos'][2]:.3f})" if ar else '—'
    al_rec   = " [RECOVERING]" if (al and al.get('recovering')) else ""
    ar_rec   = " [RECOVERING]" if (ar and ar.get('recovering')) else ""

    l_rst_str = RESET_STATE_NAMES[l_reset]
    r_rst_str = RESET_STATE_NAMES[r_reset]

    put(f"  Left  arm    : {al_str:10}  EE={al_pos}{al_rec}")
    put(f"    reset      : {l_rst_str}")
    put(f"  Right arm    : {ar_str:10}  EE={ar_pos}{ar_rec}")
    put(f"    reset      : {r_rst_str}")
    put(f"  Gripper      : {app.gripper:.2f}  (f=open, v=close)")
    put("")
    put(f"  Target arm   : {'LEFT' if tgt==1 else 'RIGHT' if tgt==2 else 'BOTH'}")
    put(f"  Cmd pos      : ({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f})")
    put(f"  Cmd quat     : w={quat[0]:+.4f} x={quat[1]:+.4f} y={quat[2]:+.4f} z={quat[3]:+.4f}")
    put(f"  Step pos     : {step:.4f} m    step ang: {sanga:.1f} deg")
    put("")
    for line in HELP:
        put("  " + line)
    put("")
    put("── Log ────────────────────────────────────────", curses.A_BOLD)
    with app.lock:
        logs = list(app.log_lines)
    for line in logs[-8:]:
        put("  " + line)

    stdscr.refresh()

def run_tui(stdscr, app, avatar_sock):
    curses.cbreak()
    curses.noecho()
    stdscr.keypad(True)
    stdscr.nodelay(True)

    while app.running:
        draw(stdscr, app)
        time.sleep(0.05)

        try:
            key = stdscr.getch()
        except Exception:
            key = -1

        if key == -1:
            continue

        ch = chr(key) if 32 <= key <= 126 else None

        with app.lock:
            av           = app.avatar_state
            step         = app.step_pos
            sang         = app.step_ang
            l_reset      = app.left_reset_state
            r_reset      = app.right_reset_state

        def nudge_pos(axis, sign):
            with app.lock:
                app.pos[axis] += sign * app.step_pos

        def nudge_rot(axis_vec, sign):
            delta_q = quat_from_axis_angle(axis_vec, sign * sang)
            with app.lock:
                app.quat = quat_normalize(quat_multiply(app.quat, delta_q))

        if key == 27:
            app.running = False

        elif ch == 'h':
            send_state_change(avatar_sock, SysState.HOMING, av)
            app.log("→ HOMING requested")

        elif ch == 'g':
            send_state_change(avatar_sock, SysState.ENGAGED, av)
            app.log("→ ENGAGED requested")

        elif ch == 'p':
            send_state_change(avatar_sock, SysState.PAUSED, av)
            app.log("→ PAUSED requested")

        elif ch == 'b':
            send_state_change(avatar_sock, SysState.IDLE, av)
            app.log("→ IDLE requested")

        elif ch == 'x':
            send_state_change(avatar_sock, SysState.STOP, av)
            app.log("→ STOP requested")

        elif ch == 'f':
            with app.lock:
                app.gripper = max(0.0, app.gripper - app.step_gripper)
            app.log(f"Gripper → {app.gripper:.2f}")

        elif ch == 'v':
            with app.lock:
                app.gripper = min(1.0, app.gripper + app.step_gripper)
            app.log(f"Gripper → {app.gripper:.2f}")

        elif ch == '1':
            with app.lock:
                app.target_arm = 1
            app.log("Target: LEFT arm")

        elif ch == '2':
            with app.lock:
                app.target_arm = 2
            app.log("Target: RIGHT arm")

        elif ch == '3':
            with app.lock:
                app.target_arm = 3
            app.log("Target: BOTH arms")

        elif ch == 'w':
            nudge_pos(0, +1)
        elif ch == 's':
            nudge_pos(0, -1)
        elif ch == 'a':
            nudge_pos(1, +1)
        elif ch == 'd':
            nudge_pos(1, -1)
        elif ch == 'q':
            nudge_pos(2, +1)
        elif ch == 'e':
            nudge_pos(2, -1)

        elif ch == 'I':
            nudge_rot((1, 0, 0), +1)
        elif ch == 'K':
            nudge_rot((1, 0, 0), -1)
        elif ch == 'J':
            nudge_rot((0, 1, 0), +1)
        elif ch == 'L':
            nudge_rot((0, 1, 0), -1)
        elif ch == 'U':
            nudge_rot((0, 0, 1), +1)
        elif ch == 'O':
            nudge_rot((0, 0, 1), -1)

        elif ch == '+' or ch == '=':
            with app.lock:
                app.step_pos = min(app.step_pos * 1.5, 0.5)
                app.step_ang = min(app.step_ang * 1.5, math.radians(45))

        elif ch == '-':
            with app.lock:
                app.step_pos = max(app.step_pos / 1.5, 0.0001)
                app.step_ang = max(app.step_ang / 1.5, math.radians(0.1))

        elif ch == 'r':
            with app.lock:
                app.pos  = [0.0, 0.0, 0.0]
                app.quat = (1.0, 0.0, 0.0, 0.0)
            app.log("Pose reset to origin")

        # z — reset left arm
        elif ch == 'z':
            if l_reset == ResetState.IDLE:
                send_arm_reset(avatar_sock, "arm_left", av)
                with app.lock:
                    app.left_reset_state = ResetState.RECOVERING
                app.log("→ arm_left reset sent")

        # X — reset right arm
        elif ch == 'X':
            if r_reset == ResetState.IDLE:
                send_arm_reset(avatar_sock, "arm_right", av)
                with app.lock:
                    app.right_reset_state = ResetState.RECOVERING
                app.log("→ arm_right reset sent")

        # c — reset both arms
        elif ch == 'c':
            if l_reset == ResetState.IDLE:
                send_arm_reset(avatar_sock, "arm_left", av)
                with app.lock:
                    app.left_reset_state = ResetState.RECOVERING
            if r_reset == ResetState.IDLE:
                send_arm_reset(avatar_sock, "arm_right", av)
                with app.lock:
                    app.right_reset_state = ResetState.RECOVERING
            app.log("→ both arms reset sent")

        # n — resume left arm
        elif ch == 'n':
            if l_reset == ResetState.AWAITING_RESUME:
                send_arm_resume(avatar_sock, "arm_left", av)
                with app.lock:
                    app.left_reset_state = ResetState.IDLE
                    app.pos  = [0.0, 0.0, 0.0]
                    app.quat = (1.0, 0.0, 0.0, 0.0)
                app.log("→ arm_left resumed, pose origin reset")

        # m — resume right arm
        elif ch == 'm':
            if r_reset == ResetState.AWAITING_RESUME:
                send_arm_resume(avatar_sock, "arm_right", av)
                with app.lock:
                    app.right_reset_state = ResetState.IDLE
                    app.pos  = [0.0, 0.0, 0.0]
                    app.quat = (1.0, 0.0, 0.0, 0.0)
                app.log("→ arm_right resumed, pose origin reset")

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    app = AppState()

    avatar_sock    = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    arm_left_sock  = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    arm_right_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    avatar_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    avatar_sock.bind(("0.0.0.0", AVATAR_RECEIVE_PORT))
    avatar_sock.settimeout(0.05)

    arm_left_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    arm_left_sock.bind(("0.0.0.0", ARM_LEFT_RECEIVE_PORT))
    arm_left_sock.settimeout(0.01)

    arm_right_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    arm_right_sock.bind(("0.0.0.0", ARM_RIGHT_RECEIVE_PORT))
    arm_right_sock.settimeout(0.01)

    threads = [
        threading.Thread(target=avatar_recv_thread,  args=(avatar_sock,    app),                    daemon=True),
        threading.Thread(target=arm_recv_thread,     args=(arm_left_sock,  "arm_left_state",  app), daemon=True),
        threading.Thread(target=arm_recv_thread,     args=(arm_right_sock, "arm_right_state", app), daemon=True),
        threading.Thread(target=heartbeat_thread,    args=(avatar_sock,    app),                    daemon=True),
        threading.Thread(target=arm_cmd_thread,      args=(arm_left_sock,  arm_right_sock,    app), daemon=True),
    ]
    for t in threads:
        t.start()

    try:
        curses.wrapper(run_tui, app, avatar_sock)
    finally:
        app.running = False
        send_state_change(avatar_sock, SysState.IDLE, app.avatar_state)
        time.sleep(0.2)
        avatar_sock.close()
        arm_left_sock.close()
        arm_right_sock.close()

if __name__ == "__main__":
    main()