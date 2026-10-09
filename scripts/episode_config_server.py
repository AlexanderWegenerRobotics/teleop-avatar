"""
UDP server that randomizes (or replays) object spawns per episode for the Avatar.
msgpack protocol: {"type": "request_episode_config"} -> {seed, mode (0 uni, 1 bi), color_bin_mapping, objects, lighting}
spawn modes: random (sorting) | fixed (task-config poses, scale 1.0, sends quat)
           | scatter (FMB: collision-free random poses at scale 1.0, plus 0-N spare parts)
spares: role=object entries with `spare: true` in the task config. Their task-config pose is
        the parking pose (under the table); unused spares are sent in "parked".
"""

import argparse
import csv
import json
import logging
import math
import random
import socket
import sys
from pathlib import Path

import msgpack
import yaml

import scatter_spawn

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("episode_config_server")

LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = 9100

DEFAULT_SPAWN = {
    "x_range":         [0.45, 0.78],
    "y_range":         [-0.30, 0.30],
    "z":               0.725,
    "min_object_dist": 0.12,
    "min_bin_dist":    0.20,
    "yaw_range":       [0.0, 2 * math.pi],
    "scale_range":     [0.90, 1.10],
    "mode":            "random",   # "random" | "fixed"
}

SPAWN_MODES = ("random", "fixed", "scatter")

MODE_WEIGHTS = {0: 0.5, 1: 0.5}

LIGHT_TARGET         = (0.8, 0.0, 0.72)
LIGHT_MAIN_DIST      = (1.2, 1.5)
LIGHT_MAIN_ELEV_DEG  = (45.0, 65.0)
LIGHT_MAIN_AZIM_DEG  = (-135.0, 135.0)
LIGHT_MAIN_INTENSITY = (0.25, 0.45)
LIGHT_MAIN_WARMTH    = (-1.0, 1.0)
LIGHT_FILL_INTENSITY = (0.15, 0.3)
LIGHT_HEAD_DIFFUSE   = (0.1,  0.2)
LIGHT_HEAD_AMBIENT   = (0.5,  0.6)
LIGHT_FILL_POS       = (1.2,  0.5, 1.0)
LIGHT_UP_BUDGET      = 1.4
LIGHT_HEAD_UP_COS    = 0.7


def load_sim_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_merged_config(sim_config_path: str) -> tuple[dict, dict]:
    """Return (merged_cfg, spawn_params), same merge as SceneBuilder::loadMergedSimConfig."""
    sim_cfg = load_sim_config(sim_config_path)
    spawn_params = dict(DEFAULT_SPAWN)

    task_path_rel = (sim_cfg.get("simulation") or {}).get("task_config")
    if task_path_rel:
        task_path = (Path(sim_config_path).parent / task_path_rel).resolve()
        log.info("Merging task config: %s", task_path)
        with open(task_path) as f:
            task_cfg = yaml.safe_load(f)

        sim_cfg.setdefault("objects", [])
        sim_cfg["objects"].extend(task_cfg.get("objects", []))

        if "spawn" in task_cfg:
            for k, v in task_cfg["spawn"].items():
                spawn_params[k] = v

    return sim_cfg, spawn_params


def _default_pose(obj: dict):
    """(position, quat_wxyz) from the object's pose: block, or (None, None)."""
    pose = obj.get("pose") or {}
    pos  = pose.get("position")
    quat = pose.get("orientation") or [1.0, 0.0, 0.0, 0.0]
    if pos is None:
        return None, None
    n = math.sqrt(sum(float(c) ** 2 for c in quat)) or 1.0
    return [float(c) for c in pos], [float(c) / n for c in quat]


def _yaw_from_quat(q) -> float:
    """Z-yaw of a wxyz quaternion (what the scene log records as spawn_yaw)."""
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _footprint(obj: dict, quat, base_dir: Path):
    mp = obj.get("model_path")
    if not mp or base_dir is None:
        return None
    path = (base_dir / mp).resolve()
    if not path.exists():
        log.warning("model not found for footprint: %s", path)
        return None
    return scatter_spawn.footprint(path, quat)


def build_object_defs(sim_cfg: dict, base_dir: Path = None) -> list:
    objects = []
    for obj in sim_cfg.get("objects", []):
        if obj.get("role") == "object":
            pos, quat = _default_pose(obj)
            objects.append({
                "name":       obj["name"],
                "color":      obj.get("color", "unknown"),
                "model_path": obj.get("model_path", ""),
                "default_pos":  pos,
                "default_quat": quat,
                "spare":      bool(obj.get("spare", False)),
                "fp":         _footprint(obj, quat or [1, 0, 0, 0], base_dir),
            })
    if not objects:
        log.warning("No objects with role=object found — check your task config.")
    return objects


def build_obstacles(sim_cfg: dict, base_dir: Path) -> list:
    obstacles = []
    for obj in sim_cfg.get("objects", []):
        if obj.get("role") in ("board", "fixture", "bin"):
            pos, quat = _default_pose(obj)
            if pos is None:
                continue
            obstacles.append((_footprint(obj, [1, 0, 0, 0], base_dir), pos, quat))
    return scatter_spawn.obstacle_rects(obstacles)


def build_bin_mapping(sim_cfg: dict) -> dict:
    mapping = {}
    for obj in sim_cfg.get("objects", []):
        if obj.get("role") == "bin":
            color = obj.get("color", "unknown")
            mapping[color] = obj["name"]
    return mapping


def build_bin_positions(sim_cfg: dict) -> list:
    positions = []
    for obj in sim_cfg.get("objects", []):
        if obj.get("role") == "bin":
            pos = (obj.get("pose") or {}).get("position")
            if pos:
                positions.append((pos[0], pos[1]))
    return positions


def _fill_up_cos():
    d = [t - p for t, p in zip(LIGHT_TARGET, LIGHT_FILL_POS)]
    n = math.sqrt(sum(c * c for c in d))
    return max(0.0, -d[2] / n)


def sample_lighting(rng):
    """Key light aimed at LIGHT_TARGET from 45-65 deg elevation; total light on an
    upward face is capped at LIGHT_UP_BUDGET so top faces never clip to white."""
    dist  = rng.uniform(*LIGHT_MAIN_DIST)
    elev  = math.radians(rng.uniform(*LIGHT_MAIN_ELEV_DEG))
    azim  = math.radians(rng.uniform(*LIGHT_MAIN_AZIM_DEG))
    tx, ty, tz = LIGHT_TARGET
    main_pos = [tx + dist * math.cos(elev) * math.cos(azim),
                ty + dist * math.cos(elev) * math.sin(azim),
                tz + dist * math.sin(elev)]

    intensity = rng.uniform(*LIGHT_MAIN_INTENSITY)
    warmth    = rng.uniform(*LIGHT_MAIN_WARMTH)
    fill_i    = rng.uniform(*LIGHT_FILL_INTENSITY)
    head_d    = rng.uniform(*LIGHT_HEAD_DIFFUSE)
    head_a    = rng.uniform(*LIGHT_HEAD_AMBIENT)

    up = (intensity * math.sin(elev) + fill_i * _fill_up_cos()
          + head_d * LIGHT_HEAD_UP_COS + head_a)
    k = min(1.0, LIGHT_UP_BUDGET / up)
    intensity, fill_i, head_d, head_a = (v * k for v in (intensity, fill_i, head_d, head_a))

    def clamp(v): return max(0.0, min(1.0, v))

    main_diffuse  = [clamp(intensity * (1.0 + 0.15 * warmth)),
                     clamp(intensity * (1.0 + 0.05 * warmth)),
                     clamp(intensity * (1.0 - 0.20 * warmth))]
    main_specular = [clamp(v * 0.25) for v in main_diffuse]

    return {
        "main_pos":          main_pos,
        "main_target":       list(LIGHT_TARGET),
        "main_diffuse":      main_diffuse,
        "main_specular":     main_specular,
        "fill_diffuse":      [fill_i] * 3,
        "headlight_diffuse": [head_d] * 3,
        "headlight_ambient": [head_a] * 3,
    }


def sample_positions(n, bin_positions, spawn, rng):
    x_range = spawn["x_range"]
    y_range = spawn["y_range"]
    z       = spawn["z"]
    min_obj = spawn["min_object_dist"]
    min_bin = spawn["min_bin_dist"]

    positions = []
    for _ in range(n):
        for _ in range(200):
            x = rng.uniform(*x_range)
            y = rng.uniform(*y_range)
            if all(math.hypot(x - px, y - py) >= min_obj for px, py, _ in positions) and \
               all(math.hypot(x - bx, y - by) >= min_bin for bx, by in bin_positions):
                positions.append((x, y, z))
                break
        else:
            x = rng.uniform(*x_range)
            y = rng.uniform(*y_range)
            positions.append((x, y, z))
            log.warning("Could not place object %d with min-distance constraint; placed anyway.", len(positions))
    return positions


def _sample_mode(rng):
    modes   = list(MODE_WEIGHTS.keys())
    weights = list(MODE_WEIGHTS.values())
    return rng.choices(modes, weights=weights, k=1)[0]


def _at_default(obj):
    pos, quat = obj["default_pos"], obj["default_quat"]
    return {
        "name":       obj["name"],
        "color":      obj["color"],
        "model_path": obj["model_path"],
        "x": pos[0], "y": pos[1], "z": pos[2],
        "yaw":   _yaw_from_quat(quat),
        "scale": 1.0,
        "quat":  quat,
    }


def _parked(spares):
    return [_at_default(o) for o in spares if o["default_pos"] is not None]


def fixed_episode(all_objects, bin_mapping, n_objects, rng, spares=()):
    """Every role=object body back at its task-config pose, scale 1.0 (FMB clearance is 1-2 mm)."""
    return {
        "mode":              _sample_mode(rng),
        "color_bin_mapping": json.dumps(bin_mapping),
        "objects":           [_at_default(o) for o in all_objects[:n_objects]],
        "parked":            _parked(spares),
        "lighting":          sample_lighting(rng),
    }


def scatter_episode(required, spares, obstacles, bin_mapping, spawn, rng):
    """All required parts plus a random subset of the spare pool, collision-free, scale 1.0.
    If a layout with k spares cannot be found, k is reduced; required parts are never dropped."""
    lo, hi = spawn.get("spares", [0, 0])
    hi = min(int(hi), len(spares))
    lo = min(int(lo), hi)
    n_spare = rng.randint(lo, hi) if hi > 0 else 0
    chosen = rng.sample(spares, n_spare)

    layout = None
    while True:
        layout = scatter_spawn.scatter(required + chosen, obstacles, spawn, rng)
        if layout is not None or not chosen:
            break
        log.warning("scatter: no layout with %d spares, retrying with %d", len(chosen), len(chosen) - 1)
        chosen = chosen[:-1]

    if layout is None:
        log.error("scatter: no collision-free layout for the required parts; using task-config poses. "
                  "Widen spawn x_range/y_range or lower object_gap/obstacle_gap.")
        ep = fixed_episode(required, bin_mapping, len(required), rng, spares)
        return ep, []

    table_z = {o["model_path"]: o["default_pos"][2] for o in required}
    active = required + chosen
    rng.shuffle(active)
    spawned = []
    for obj in active:
        x, y, yaw = layout[obj["name"]]
        quat = scatter_spawn.quat_mul(scatter_spawn.yaw_quat(yaw), obj["default_quat"])
        z = obj["default_pos"][2] if not obj["spare"] else table_z.get(obj["model_path"], spawn["z"])
        spawned.append({
            "name":       obj["name"],
            "color":      obj["color"],
            "model_path": obj["model_path"],
            "x": x, "y": y, "z": z,
            "yaw":   scatter_spawn.yaw_of(quat),
            "scale": 1.0,
            "quat":  quat,
        })
    chosen_names = {o["name"] for o in chosen}
    return {
        "mode":              _sample_mode(rng),
        "color_bin_mapping": json.dumps(bin_mapping),
        "objects":           spawned,
        "parked":            _parked([o for o in spares if o["name"] not in chosen_names]),
        "lighting":          sample_lighting(rng),
    }, sorted(chosen_names)


def sample_episode(all_objects, bin_mapping, bin_positions, n_objects, spawn, rng):
    active    = all_objects[:n_objects]
    positions = sample_positions(len(active), bin_positions, spawn, rng)

    yaw_range   = spawn["yaw_range"]
    scale_range = spawn["scale_range"]

    spawned = []
    for obj, (x, y, z) in zip(active, positions):
        spawned.append({
            "name":       obj["name"],
            "color":      obj["color"],
            "model_path": obj["model_path"],
            "x": x, "y": y, "z": z,
            "yaw":   rng.uniform(*yaw_range),
            "scale": rng.uniform(*scale_range),
        })

    return {
        "mode":              _sample_mode(rng),
        "color_bin_mapping": json.dumps(bin_mapping),
        "objects":           spawned,
        "lighting":          sample_lighting(rng),
    }


# replay: serve a recorded episode's scene instead of a random one
# scene.csv is per session, so rows are filtered by the episode's seed

def _read_episode_meta(folder: Path) -> dict:
    """seed / mode / color_bin_mapping from the episode_config event of any meta file."""
    for name in ("arm_left_meta.csv", "arm_right_meta.csv", "scene_meta.csv"):
        path = folder / name
        if not path.exists():
            continue
        with open(path) as f:
            for row in csv.DictReader(f, delimiter=";"):
                if row.get("event") == "episode_config" and row.get("seed"):
                    return {
                        "seed": int(row["seed"]),
                        "mode": int(row["mode"]) if row.get("mode") else 0,
                        "color_bin_mapping": row.get("color_bin_mapping") or "{}",
                    }
    raise ValueError(f"{folder}: no episode_config event found in any *_meta.csv")


def _spawn_row(folder: Path, seed: int) -> dict:
    """First scene.csv row with this episode's seed."""
    path = folder / "scene.csv"
    if not path.exists():
        raise ValueError(f"{folder}: no scene.csv")
    with open(path) as f:
        for row in csv.DictReader(f, delimiter=";"):
            if row.get("seed") and int(row["seed"]) == seed:
                return row
    raise ValueError(f"{folder}: no scene.csv row with seed {seed}")


def _lighting_from_row(row: dict) -> dict:
    """Lighting block from a scene row, only the triples that are present."""
    def triple(prefix, keys):
        vals = [row.get(f"{prefix}_{k}") for k in keys]
        if any(v is None or v == "" for v in vals):
            return None
        return [float(v) for v in vals]

    out = {}
    for key, prefix, comps in (
        ("main_pos", "light_main_pos", ("x", "y", "z")),
        ("main_diffuse", "light_main_diffuse", ("r", "g", "b")),
        ("main_specular", "light_main_specular", ("r", "g", "b")),
        ("fill_diffuse", "light_fill_diffuse", ("r", "g", "b")),
        ("headlight_diffuse", "light_headlight_diffuse", ("r", "g", "b")),
        ("headlight_ambient", "light_headlight_ambient", ("r", "g", "b")),
    ):
        v = triple(prefix, comps)
        if v is not None:
            out[key] = v
    return out


def load_recorded_episode(folder: Path, model_paths: dict, spare_names=frozenset()) -> dict:
    """Rebuild the recorded episode config; model_path comes from the sim config."""
    meta = _read_episode_meta(folder)
    row = _spawn_row(folder, meta["seed"])

    n_objects = int(float(row["n_objects"]))
    objects, parked = [], []
    for i in range(n_objects):
        name = row.get(f"obj{i}_name") or ""
        if not name:
            continue
        dst = parked if (name in spare_names and not row.get(f"obj{i}_color")) else objects
        dst.append({
            "name": name,
            "color": row.get(f"obj{i}_color", "unknown"),
            "model_path": model_paths.get(name, ""),
            "x": float(row[f"obj{i}_x"]),
            "y": float(row[f"obj{i}_y"]),
            "z": float(row[f"obj{i}_z"]),
            # spawn yaw, not the live pose
            "yaw": float(row.get(f"obj{i}_spawn_yaw") or 0.0),
            "scale": float(row.get(f"obj{i}_scale") or 1.0),
        })

    episode = {
        "seed": meta["seed"],
        "mode": meta["mode"],
        "color_bin_mapping": meta["color_bin_mapping"],
        "objects": objects,
        "parked": parked,
        "lighting": _lighting_from_row(row),
    }
    episode["lighting"].setdefault("main_target", list(LIGHT_TARGET))
    return episode


def build_model_paths(sim_cfg: dict) -> dict:
    return {o["name"]: o.get("model_path", "")
            for o in sim_cfg.get("objects", []) if o.get("role") == "object"}


def run(sim_config_path, n_objects_override, replay_folders=None, replay_loop=True,
        spawn_mode_override=None):
    sim_cfg, spawn = resolve_merged_config(sim_config_path)
    base_dir       = Path(sim_config_path).parent
    defs           = build_object_defs(sim_cfg, base_dir)
    all_objects    = [o for o in defs if not o["spare"]]
    spares         = [o for o in defs if o["spare"]]
    bin_mapping    = build_bin_mapping(sim_cfg)
    bin_positions  = build_bin_positions(sim_cfg)
    obstacles      = build_obstacles(sim_cfg, base_dir)

    spawn_mode = spawn_mode_override or spawn.get("mode", "random")
    if spawn_mode not in SPAWN_MODES:
        log.error("Unknown spawn mode %r (expected one of %s)", spawn_mode, SPAWN_MODES)
        sys.exit(1)
    spawn["mode"] = spawn_mode
    if spawn_mode in ("fixed", "scatter"):
        missing = [o["name"] for o in defs if o["default_pos"] is None]
        if missing:
            log.error("spawn mode '%s' needs pose.position for every role=object; missing: %s",
                      spawn_mode, missing)
            sys.exit(1)
    if spawn_mode == "scatter":
        no_fp = [o["name"] for o in defs if o["fp"] is None]
        if no_fp:
            log.error("spawn mode 'scatter' needs box geoms in the model of every part; none for: %s", no_fp)
            sys.exit(1)
        if len(obstacles) == 0:
            log.warning("scatter: no board/fixture/bin obstacles found")

    # replay mode: one recorded scene per request
    replay_queue = []
    if replay_folders:
        model_paths = build_model_paths(sim_cfg)
        spare_names = frozenset(o["name"] for o in spares)
        for folder in replay_folders:
            try:
                ep = load_recorded_episode(Path(folder), model_paths, spare_names)
            except Exception as e:
                log.error("Cannot replay %s: %s", folder, e)
                sys.exit(1)
            replay_queue.append((Path(folder).name, ep))
            log.info("Loaded replay episode %s | seed=%d mode=%s %d objects",
                     Path(folder).name, ep["seed"], ep["mode"], len(ep["objects"]))
        log.info("REPLAY MODE: %d episode(s), %s. Randomization disabled.",
                 len(replay_queue), "looping" if replay_loop else "then exit")

    if not all_objects and not replay_queue:
        log.error("No pickable objects found — exiting.")
        sys.exit(1)

    n_objects = n_objects_override if n_objects_override is not None else len(all_objects)
    n_objects = min(n_objects, len(all_objects))

    log.info("Loaded %d pickable objects, will spawn %d per episode.", len(all_objects), n_objects)
    if spares:
        log.info("Spare pool: %s, %s per episode", [o["name"] for o in spares], spawn.get("spares", [0, 0]))
    log.info("Spawn mode: %s%s", spawn_mode,
             "  (objects reset to task-config poses, scale 1.0)" if spawn_mode == "fixed" else "")
    log.info("Bin mapping: %s", bin_mapping)
    log.info("Bin positions (exclusion zone): %s", bin_positions)
    log.info("Spawn params: %s", spawn)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((LISTEN_HOST, LISTEN_PORT))
    sock.settimeout(1.0)
    log.info("Listening on %s:%d", LISTEN_HOST, LISTEN_PORT)
    served = 0

    try:
        while True:
            try:
                raw, addr = sock.recvfrom(4096)
            except socket.timeout:
                continue

            try:
                msg = msgpack.unpackb(raw, raw=False)

                if msg.get("type") != "request_episode_config":
                    log.warning("Unknown message type: %s", msg.get("type"))
                    continue

                if replay_queue:
                    if served >= len(replay_queue) and not replay_loop:
                        log.info("All %d replay episodes served; exiting.", len(replay_queue))
                        return
                    label, episode = replay_queue[served % len(replay_queue)]
                    served += 1
                    log.info("REPLAY %d/%d -> episode %s",
                             ((served - 1) % len(replay_queue)) + 1, len(replay_queue), label)
                else:
                    seed    = random.randint(0, 2**31 - 1)
                    rng     = random.Random(seed)
                    if spawn_mode == "fixed":
                        episode = fixed_episode(all_objects, bin_mapping, n_objects, rng, spares)
                    elif spawn_mode == "scatter":
                        episode, used_spares = scatter_episode(
                            all_objects[:n_objects], spares, obstacles, bin_mapping, spawn, rng)
                        log.info("  spares this episode: %s", used_spares or "none")
                    else:
                        episode = sample_episode(all_objects, bin_mapping, bin_positions,
                                                 n_objects, spawn, rng)
                    episode["seed"] = seed
                    episode.setdefault("parked", _parked(spares))

                sock.sendto(msgpack.packb(episode), addr)

                names = [o["name"] for o in episode["objects"]]
                log.info(
                    "Episode sent | seed=%d mode=%s objects=%s",
                    episode["seed"],
                    "bimanual" if episode["mode"] else "unimanual",
                    names,
                )
                for o in episode["objects"]:
                    log.info("  %s (%s) -> (%.3f, %.3f, %.3f) yaw=%.1f° scale=%.2f",
                             o["name"], o["color"], o["x"], o["y"], o["z"],
                             math.degrees(o["yaw"]), o["scale"])
                lt = episode["lighting"]
                mp = lt["main_pos"]
                log.info(
                    "  lighting | main_pos=(%.2f,%.2f,%.2f) diffuse=(%.2f,%.2f,%.2f) fill=%.2f "
                    "head=%.2f amb=%.2f",
                    mp[0], mp[1], mp[2],
                    lt["main_diffuse"][0], lt["main_diffuse"][1], lt["main_diffuse"][2],
                    sum(lt["fill_diffuse"]) / 3,
                    sum(lt.get("headlight_diffuse", [0, 0, 0])) / 3,
                    sum(lt.get("headlight_ambient", [0, 0, 0])) / 3,
                )

            except Exception as e:
                log.error("Error: %s", e, exc_info=True)

    except KeyboardInterrupt:
        log.info("Shutting down.")
    finally:
        sock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Episode config server")
    parser.add_argument(
        "--sim-config",
        default=str(Path(__file__).parent.parent / "config" / "sim_config_franka.yaml"),
        help="Path to sim_config_franka.yaml (task_config is followed automatically)",
    )
    parser.add_argument(
        "--n-objects", type=int, default=None,
        help="Override number of objects to spawn per episode. "
             "Defaults to all objects defined in the task config.",
    )
    parser.add_argument(
        "--spawn-mode", choices=SPAWN_MODES, default=None,
        help="Override the task config's spawn.mode. 'random' randomizes "
             "role=object poses (parcels); 'fixed' puts them back at their "
             "task-config poses every episode; 'scatter' places them "
             "collision-free at random poses, scale 1.0, plus spare parts (FMB).",
    )
    parser.add_argument(
        "--replay", nargs="+", metavar="EPISODE_DIR", default=None,
        help="Replay recorded episodes instead of randomizing: serve each "
             "folder's exact recorded scene (positions, yaw, scale, lighting, "
             "seed, mode), one per request, in order. Point at the episode "
             "folders under the data store, e.g. --replay .../avatar/014 "
             ".../avatar/016. This is what makes a policy rollout directly "
             "comparable to the demonstration it is being scored against.",
    )
    parser.add_argument(
        "--replay-root", default=None,
        help="Data store root; combine with --episodes instead of listing full paths.",
    )
    parser.add_argument(
        "--episodes", nargs="+", default=None,
        help="Episode ids to replay from --replay-root, e.g. --episodes 14 16 22",
    )
    parser.add_argument(
        "--no-loop", action="store_true",
        help="Exit after serving each replay episode once (default: loop forever).",
    )
    args = parser.parse_args()

    replay = args.replay
    if args.replay_root and args.episodes:
        replay = [str(Path(args.replay_root) / str(e).zfill(3)) for e in args.episodes]
    run(args.sim_config, args.n_objects, replay_folders=replay, replay_loop=not args.no_loop,
        spawn_mode_override=args.spawn_mode)
