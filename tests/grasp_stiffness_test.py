#!/usr/bin/env python3
"""Grasp stiffness regression test for models/mujoco/robots/franka_fr3/hand.xml.

Loads the real hand model (meshes included), fixes it in an otherwise empty
world with the same <option> the SceneBuilder generates, closes it on a
37 x 37 x 200 mm bar (FMB board-1 bar: 0.2738 kg, PLA friction 0.3), then
loads the bar and reports how far it moves in the fingers. Run it after any
change to the hand model, the SceneBuilder <option>, or the timestep.

    python tests/grasp_stiffness_test.py            # defaults: dt 1 ms
    python tests/grasp_stiffness_test.py --dt 0.0005

Expected with the 2026-09 hand model (kp 30 kN/m, 70 N cap, stiff pads,
impratio 10, dt 1 ms): every "moved" number below 0.3 mm, twist under 1 deg,
10 s hold drift under 0.2 mm, close 80 -> 37 mm in roughly 0.4 s.
The pre-2026-09 model (kp 1 kN/m, soft pads) dropped the bar at 20 N.
"""
import argparse
import os
import sys

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
HAND_XML = os.path.join(HERE, "..", "models", "mujoco", "robots", "franka_fr3", "hand.xml")

# Same as Simulation::setGripper: width -> ctrl in (0, 255) on the half-width.
def ctrl_for_width(w):
    half = np.clip(w, 0.006, 0.08) / 2.0
    return half / 0.04 * 255.0


def build(hand_xml, dt, impratio, bar_w=0.037, bar_mass=0.2738):
    world = mujoco.MjSpec.from_string(f"""
<mujoco model="grasp_test">
  <compiler angle="radian" autolimits="true"/>
  <option gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic"
          timestep="{dt}" impratio="{impratio}"/>
  <worldbody>
    <body name="mount" pos="0 0 0.5"/>
    <body name="bar" pos="0 0 0.6029">
      <freejoint name="bar_free"/>
      <geom name="bar" type="box" size="0.1 {bar_w/2} {bar_w/2}" mass="{bar_mass}"
            friction="0.3 0.02 0.001" condim="6" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>""")
    hand = mujoco.MjSpec.from_file(hand_xml)
    mount = world.body("mount")
    frame = mount.add_frame()
    frame.attach_body(hand.body("hand"), "h_", "")
    return world.compile()


def finger_joints(m):
    ids = [j for j in range(m.njnt) if "finger_joint" in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
    assert len(ids) == 2, ids
    return [m.jnt_qposadr[j] for j in ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", default=HAND_XML)
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--impratio", type=float, default=10.0)
    args = ap.parse_args()

    m = build(args.hand, args.dt, args.impratio)
    d = mujoco.MjData(m)
    dt = m.opt.timestep
    bar = m.body("bar").id
    qa = finger_joints(m)
    width = lambda: d.qpos[qa[0]] + d.qpos[qa[1]]
    step = lambda seconds: [mujoco.mj_step(m, d) for _ in range(int(round(seconds / dt)))]

    # open, no gravity while the bar waits between the fingers
    d.ctrl[0] = ctrl_for_width(0.08)
    d.qpos[qa[0]] = d.qpos[qa[1]] = 0.04
    m.opt.gravity[:] = 0
    step(0.05)

    # close exactly as ArmControl::applyGripper does in sim: setWidth(0)
    d.ctrl[0] = ctrl_for_width(0.0)
    t_close = None
    for i in range(int(1.5 / dt)):
        mujoco.mj_step(m, d)
        if t_close is None and abs(width() - 0.037) < 0.0005:
            t_close = (i + 1) * dt
    w_closed = width()
    m.opt.gravity[:] = [0, 0, -9.81]
    step(0.5)
    p0 = d.xpos[bar].copy()
    q0 = d.xquat[bar].copy()
    f_tendon = d.actuator_force[0]

    def rel_angle_deg():
        qrel, qinv = np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(qinv, q0)
        mujoco.mju_mulQuat(qrel, d.xquat[bar], qinv)
        return 2 * np.degrees(np.arccos(np.clip(abs(qrel[0]), -1, 1)))

    print(f"hand: {os.path.relpath(args.hand)}  dt {dt*1e3:.2f} ms  impratio {m.opt.impratio:g}")
    print(f"close 80 -> 37 mm: {t_close if t_close is None else f'{t_close:.2f} s'}, "
          f"settled width {w_closed*1e3:.2f} mm, tendon force {f_tendon:.1f} N ({0.5*f_tendon:.1f} N per finger)")

    worst_move = worst_rot = 0.0
    for force, torque, name in [
        ((0, 20, 0), (0, 0, 0), "20 N lateral (finger-closing axis)"),
        ((20, 0, 0), (0, 0, 0), "20 N along the bar"),
        ((0, 0, -20), (0, 0, 0), "20 N pull-out"),
        ((0, 0, 0), (0, 1.0, 0), "1 Nm tilt in the fingers"),
        ((0, 0, 0), (0, 0, 1.0), "1 Nm twist about the finger axis"),
    ]:
        d.xfrc_applied[bar, :3] = force
        d.xfrc_applied[bar, 3:] = torque
        pa = d.xpos[bar].copy()
        step(1.0)
        moved = np.linalg.norm(d.xpos[bar] - pa) * 1e3
        rot = rel_angle_deg()
        d.xfrc_applied[bar, :] = 0
        step(0.3)
        worst_move = max(worst_move, moved)
        worst_rot = max(worst_rot, rot)
        print(f"  {name:36s} moved {moved:6.2f} mm  rotated {rot:5.2f} deg")

    step(10.0)
    drift = np.linalg.norm(d.xpos[bar] - p0) * 1e3
    print(f"  10 s hold afterwards: total drift {drift:.2f} mm, width {width()*1e3:.2f} mm, "
          f"finite={bool(np.all(np.isfinite(d.qpos)))}")

    d.ctrl[0] = ctrl_for_width(0.08)
    t_open = None
    for i in range(int(3.0 / dt)):
        mujoco.mj_step(m, d)
        if t_open is None and width() > 0.079:
            t_open = (i + 1) * dt
    print(f"  open 37 -> 79 mm: {t_open if t_open is None else f'{t_open:.2f} s'}")

    ok = worst_move < 0.5 and worst_rot < 1.0 and drift < 1.0 and np.all(np.isfinite(d.qpos))
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
