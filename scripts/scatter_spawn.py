"""
Collision-free scatter of rectilinear parts (FMB pegs) on the table.

Footprints come from the MuJoCo model files (box geoms), so dimensions are never
scaled: every part is placed at scale 1.0, upright in its task-config orientation,
with a random yaw. Placement is rejection sampling of oriented rectangles (SAT test)
against each other and against the static obstacles (board, fixture, bins), with
the whole rectangle inside the reachable x/y range.
"""

import math
import xml.etree.ElementTree as ET
from pathlib import Path


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw]


def quat_rot(q, v):
    w, x, y, z = q
    p = quat_mul(quat_mul(q, [0.0, *v]), [w, -x, -y, -z])
    return p[1:]


def yaw_quat(yaw):
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _euler_quat(e, degrees):
    if degrees:
        e = [math.radians(v) for v in e]
    q = [1.0, 0.0, 0.0, 0.0]
    for angle, axis in zip(e, ([1, 0, 0], [0, 1, 0], [0, 0, 1])):
        s = math.sin(angle / 2.0)
        q = quat_mul(q, [math.cos(angle / 2.0), axis[0] * s, axis[1] * s, axis[2] * s])
    return q


def _floats(s, default):
    return [float(v) for v in s.split()] if s else list(default)


def _box_geoms(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    comp = root.find("compiler")
    degrees = comp is None or comp.get("angle", "degree") != "radian"
    wb = root.find("worldbody")
    if wb is None:
        return []
    boxes = []
    for g in wb.iter("geom"):
        if g.get("type") != "box" or g.get("class") == "vis" or g.get("contype") == "0":
            continue
        size = _floats(g.get("size"), [0, 0, 0])
        pos = _floats(g.get("pos"), [0, 0, 0])
        if g.get("quat"):
            q = _floats(g.get("quat"), [1, 0, 0, 0])
        elif g.get("euler"):
            q = _euler_quat(_floats(g.get("euler"), [0, 0, 0]), degrees)
        else:
            q = [1.0, 0.0, 0.0, 0.0]
        n = math.sqrt(sum(c * c for c in q)) or 1.0
        boxes.append((pos, size, [c / n for c in q]))
    return boxes


def footprint(xml_path: Path, q_default):
    """Axis-aligned xy footprint of the part in its default orientation, relative
    to the body origin: (cx, cy, hx, hy, zmin). None if the model has no boxes."""
    boxes = _box_geoms(xml_path)
    if not boxes:
        return None
    xs, ys, zs = [], [], []
    for pos, size, q in boxes:
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    c = quat_rot(q, [sx * size[0], sy * size[1], sz * size[2]])
                    c = [c[i] + pos[i] for i in range(3)]
                    c = quat_rot(q_default, c)
                    xs.append(c[0]); ys.append(c[1]); zs.append(c[2])
    return ((max(xs) + min(xs)) / 2.0, (max(ys) + min(ys)) / 2.0,
            (max(xs) - min(xs)) / 2.0, (max(ys) - min(ys)) / 2.0, min(zs))


class Rect:
    __slots__ = ("x", "y", "hx", "hy", "th")

    def __init__(self, x, y, hx, hy, th):
        self.x, self.y, self.hx, self.hy, self.th = x, y, hx, hy, th

    @classmethod
    def from_pose(cls, fp, x, y, yaw):
        cx, cy, hx, hy, _ = fp
        c, s = math.cos(yaw), math.sin(yaw)
        return cls(x + c * cx - s * cy, y + s * cx + c * cy, hx, hy, yaw)

    def axes(self):
        c, s = math.cos(self.th), math.sin(self.th)
        return (c, s), (-s, c)

    def corners(self):
        (ux, uy), (vx, vy) = self.axes()
        return [(self.x + a * self.hx * ux + b * self.hy * vx,
                 self.y + a * self.hx * uy + b * self.hy * vy)
                for a in (-1, 1) for b in (-1, 1)]

    def overlaps(self, other, gap):
        dx, dy = other.x - self.x, other.y - self.y
        a_u, a_v = self.axes()
        b_u, b_v = other.axes()
        for ax, ay in (a_u, a_v, b_u, b_v):
            ra = self.hx * abs(a_u[0] * ax + a_u[1] * ay) + self.hy * abs(a_v[0] * ax + a_v[1] * ay)
            rb = other.hx * abs(b_u[0] * ax + b_u[1] * ay) + other.hy * abs(b_v[0] * ax + b_v[1] * ay)
            if abs(dx * ax + dy * ay) >= ra + rb + gap:
                return False
        return True

    def inside(self, x_range, y_range):
        return all(x_range[0] <= cx <= x_range[1] and y_range[0] <= cy <= y_range[1]
                   for cx, cy in self.corners())


def obstacle_rects(obstacles):
    """obstacles: list of (footprint, pos, quat) for the static task parts."""
    rects = []
    for fp, pos, quat in obstacles:
        if fp is None or pos is None:
            continue
        rects.append(Rect.from_pose(fp, pos[0], pos[1], yaw_of(quat)))
    return rects


def scatter(parts, obstacles, spawn, rng, tries_per_part=400, restarts=60):
    """parts: list of dicts with 'name' and 'fp'. Returns {name: (x, y, yaw)} or None.
    Larger parts are placed first; a failed part restarts the whole layout."""
    x_range = spawn["x_range"]
    y_range = spawn["y_range"]
    yaw_range = spawn.get("yaw_range", [0.0, 2.0 * math.pi])
    gap_obj = float(spawn.get("object_gap", 0.04))
    gap_obs = float(spawn.get("obstacle_gap", 0.03))
    order = sorted(parts, key=lambda p: -(p["fp"][2] * p["fp"][3]))

    for _ in range(restarts):
        placed, out = [], {}
        for part in order:
            for _ in range(tries_per_part):
                x = rng.uniform(*x_range)
                y = rng.uniform(*y_range)
                yaw = rng.uniform(*yaw_range)
                r = Rect.from_pose(part["fp"], x, y, yaw)
                if not r.inside(x_range, y_range):
                    continue
                if any(r.overlaps(o, gap_obs) for o in obstacles):
                    continue
                if any(r.overlaps(p, gap_obj) for p in placed):
                    continue
                placed.append(r)
                out[part["name"]] = (x, y, yaw)
                break
            else:
                break
        if len(out) == len(order):
            return out
    return None
