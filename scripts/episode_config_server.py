"""
episode_config_server.py
------------------------
UDP server that randomizes object positions each episode and returns
full per-object spawn configs to the Avatar process.

Protocol (msgpack over UDP):
  Request  <- Avatar: {"type": "request_episode_config"}
  Response -> Avatar: {
      "seed":              int,
      "mode":              int,      # 0=unimanual, 1=bimanual
      "color_bin_mapping": str,      # JSON, e.g. '{"red":"bin_1","blue":"bin_2"}'
      "objects": [
          {"name": str, "color": str, "model_path": str, "x": float, "y": float, "z": float,
           "yaw": float, "scale": float},
          ...
      ],
      "lighting": {...}
  }

Objects with role=bin are NOT included in the response; their positions are fixed
and the Avatar reads them directly from the (merged) sim_config at startup.

Spawn parameters are read from the task config's "spawn:" block (if present),
falling back to the defaults below.  This means each task can define its own
randomization ranges without touching this script.
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

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("episode_config_server")

LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = 9100

# Fallback spawn defaults (overridden by task config spawn: block)
DEFAULT_SPAWN = {
    "x_range":         [0.45, 0.78],
    "y_range":         [-0.30, 0.30],
    "z":               0.725,
    "min_object_dist": 0.12,
    "min_bin_dist":    0.20,
    "yaw_range":       [0.0, 2 * math.pi],
    "scale_range":     [0.90, 1.10],
}

MODE_WEIGHTS = {0: 0.5, 1: 0.5}

LIGHT_MAIN_X         = (0.2,  0.8)
LIGHT_MAIN_Y         = (-0.4, 0.4)
LIGHT_MAIN_Z         = (1.4,  2.0)
LIGHT_MAIN_INTENSITY = (0.6,  1.0)
LIGHT_MAIN_WARMTH    = (-1.0, 1.0)
LIGHT_FILL_INTENSITY = (0.1,  0.35)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_sim_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_merged_config(sim_config_path: str) -> tuple[dict, dict]:
    """Return (merged_cfg, spawn_params).

    Follows simulation.task_config if present and merges its objects into
    sim_config, exactly mirroring what SceneBuilder::loadMergedSimConfig does
    on the C++ side.
    """
    sim_cfg = load_sim_config(sim_config_path)
    spawn_params = dict(DEFAULT_SPAWN)

    task_path_rel = (sim_cfg.get("simulation") or {}).get("task_config")
    if task_path_rel:
        task_path = (Path(sim_config_path).parent / task_path_rel).resolve()
        log.info("Merging task config: %s", task_path)
        with open(task_path) as f:
            task_cfg = yaml.safe_load(f)

        # Merge objects
        sim_cfg.setdefault("objects", [])
        sim_cfg["objects"].extend(task_cfg.get("objects", []))

        # Override spawn params from task config
        if "spawn" in task_cfg:
            for k, v in task_cfg["spawn"].items():
                spawn_params[k] = v

    return sim_cfg, spawn_params


def build_object_defs(sim_cfg: dict) -> list:
    objects = []
    for obj in sim_cfg.get("objects", []):
        if obj.get("role") == "object":
            objects.append({
                "name":       obj["name"],
                "color":      obj.get("color", "unknown"),
                "model_path": obj.get("model_path", ""),
            })
    if not objects:
        log.warning("No objects with role=object found — check your task config.")
    return objects


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


# ---------------------------------------------------------------------------
# Randomization
# ---------------------------------------------------------------------------

def sample_lighting(rng):
    intensity = rng.uniform(*LIGHT_MAIN_INTENSITY)
    warmth    = rng.uniform(*LIGHT_MAIN_WARMTH)

    def clamp(v): return max(0.0, min(1.0, v))

    main_diffuse  = [clamp(intensity * (1.0 + 0.15 * warmth)),
                     clamp(intensity * (1.0 + 0.05 * warmth)),
                     clamp(intensity * (1.0 - 0.20 * warmth))]
    main_specular = [clamp(v * 0.25) for v in main_diffuse]
    fill_i        = rng.uniform(*LIGHT_FILL_INTENSITY)

    return {
        "main_pos":      [rng.uniform(*LIGHT_MAIN_X),
                          rng.uniform(*LIGHT_MAIN_Y),
                          rng.uniform(*LIGHT_MAIN_Z)],
        "main_diffuse":  main_diffuse,
        "main_specular": main_specular,
        "fill_diffuse":  [fill_i, fill_i, fill_i],
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

    modes   = list(MODE_WEIGHTS.keys())
    weights = list(MODE_WEIGHTS.values())
    mode    = rng.choices(modes, weights=weights, k=1)[0]

    return {
        "mode":              mode,
        "color_bin_mapping": json.dumps(bin_mapping),
        "objects":           spawned,
        "lighting":          sample_lighting(rng),
    }


# ---------------------------------------------------------------------------
# Replay: serve a previously RECORDED episode's scene instead of a random one
# ---------------------------------------------------------------------------
#
# Why: evaluating a trained policy against its own training data needs the two
# runs to face the same scene. Everything the Avatar needs to rebuild a scene
# exactly is already in the recording -- scene.csv carries per-object position,
# spawn yaw and scale plus the full lighting block, and arm_*_meta.csv carries
# the episode_config event with seed/mode/color_bin_mapping -- so replay is just
# reading those back instead of sampling.
#
# Caveat that bit during implementation: scene.csv is written per SESSION, not
# per episode, so the tail of the file can contain rows belonging to the next
# episode (observed: 7008 rows for this episode's seed, then 1 row of the next).
# Rows are therefore filtered to the seed named in the meta file's
# episode_config event before the spawn state is read off the first one.


def _read_episode_meta(folder: Path) -> dict:
    """seed / mode / color_bin_mapping from the episode_config event.

    Tries each arm's meta file: the event is written per-device, and an
    interrupted recording can leave one side without it.
    """
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
    """First scene.csv row belonging to this episode (see the seed caveat above)."""
    path = folder / "scene.csv"
    if not path.exists():
        raise ValueError(f"{folder}: no scene.csv")
    with open(path) as f:
        for row in csv.DictReader(f, delimiter=";"):
            if row.get("seed") and int(row["seed"]) == seed:
                return row
    raise ValueError(f"{folder}: no scene.csv row with seed {seed}")


def _lighting_from_row(row: dict) -> dict:
    """Reconstructs the lighting block. Avatar seeds lighting from cfg.seed and
    then overrides with whatever keys are present, so partial blocks are safe --
    only emit the triples the recording actually has."""
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


def load_recorded_episode(folder: Path, model_paths: dict) -> dict:
    """Rebuilds the exact episode config the Avatar was served when recording.

    model_path isn't in scene.csv (it never changes per object), so it's looked
    up from the merged sim config by object name; unknown names get "" and the
    Avatar falls back to its own default for that body.
    """
    meta = _read_episode_meta(folder)
    row = _spawn_row(folder, meta["seed"])

    n_objects = int(float(row["n_objects"]))
    objects = []
    for i in range(n_objects):
        name = row.get(f"obj{i}_name") or ""
        if not name:
            continue
        objects.append({
            "name": name,
            "color": row.get(f"obj{i}_color", "unknown"),
            "model_path": model_paths.get(name, ""),
            "x": float(row[f"obj{i}_x"]),
            "y": float(row[f"obj{i}_y"]),
            "z": float(row[f"obj{i}_z"]),
            # spawn_yaw, not the live quaternion: we want the pose the object
            # was created with, not wherever it had rolled to by this row.
            "yaw": float(row.get(f"obj{i}_spawn_yaw") or 0.0),
            "scale": float(row.get(f"obj{i}_scale") or 1.0),
        })

    episode = {
        "seed": meta["seed"],
        "mode": meta["mode"],
        "color_bin_mapping": meta["color_bin_mapping"],
        "objects": objects,
        "lighting": _lighting_from_row(row),
    }
    return episode


def build_model_paths(sim_cfg: dict) -> dict:
    return {o["name"]: o.get("model_path", "")
            for o in sim_cfg.get("objects", []) if o.get("role") == "object"}


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def run(sim_config_path, n_objects_override, replay_folders=None, replay_loop=True):
    sim_cfg, spawn = resolve_merged_config(sim_config_path)
    all_objects    = build_object_defs(sim_cfg)
    bin_mapping    = build_bin_mapping(sim_cfg)
    bin_positions  = build_bin_positions(sim_cfg)

    # --- replay mode: serve recorded scenes in order, one per request --------
    replay_queue = []
    if replay_folders:
        model_paths = build_model_paths(sim_cfg)
        for folder in replay_folders:
            try:
                ep = load_recorded_episode(Path(folder), model_paths)
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

    # n_objects: explicit CLI override > all available objects
    n_objects = n_objects_override if n_objects_override is not None else len(all_objects)
    n_objects = min(n_objects, len(all_objects))

    log.info("Loaded %d pickable objects, will spawn %d per episode.", len(all_objects), n_objects)
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
                    episode = sample_episode(all_objects, bin_mapping, bin_positions, n_objects, spawn, rng)
                    episode["seed"] = seed

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
                    "  lighting | main_pos=(%.2f,%.2f,%.2f) diffuse=(%.2f,%.2f,%.2f) fill=%.2f",
                    mp[0], mp[1], mp[2],
                    lt["main_diffuse"][0], lt["main_diffuse"][1], lt["main_diffuse"][2],
                    sum(lt["fill_diffuse"]) / 3,
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
    run(args.sim_config, args.n_objects, replay_folders=replay, replay_loop=not args.no_loop)
