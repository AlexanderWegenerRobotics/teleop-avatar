"""
episode_config_server.py
------------------------
UDP server that randomizes carton positions each episode and returns
full per-object spawn configs to the Avatar process.

Protocol (msgpack over UDP):
  Request  <- Avatar: {"type": "request_episode_config"}
  Response -> Avatar: {
      "seed":              int,
      "mode":              int,      # 0=unimanual, 1=bimanual
      "color_bin_mapping": str,      # JSON, e.g. '{"red":"bin_1","blue":"bin_2"}'
      "objects": [
          {"name": str, "color": str, "model_path": str, "x": float, "y": float, "z": float},
          ...
      ]
  }

Objects with role=bin are NOT included in the response; their positions are fixed
in the sim_config and the Avatar reads them directly from there at startup.
"""

import argparse
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

SPAWN_X_RANGE = (0.55, 0.85)
SPAWN_Y_RANGE = (-0.30, 0.30)
SPAWN_Z       = 0.725

MIN_OBJECT_DIST  = 0.12
MIN_BIN_DIST     = 0.18

MODE_WEIGHTS = {0: 0.5, 1: 0.5}


def load_sim_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


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
        log.warning("No objects with role=object found in sim_config — check your YAML.")
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
            pos = obj.get("pose", {}).get("position", None)
            if pos:
                positions.append((pos[0], pos[1]))
    return positions


def sample_positions(n, bin_positions, rng):
    positions = []
    for _ in range(n):
        for attempt in range(200):
            x = rng.uniform(*SPAWN_X_RANGE)
            y = rng.uniform(*SPAWN_Y_RANGE)
            far_from_objects = all(
                math.hypot(x - px, y - py) >= MIN_OBJECT_DIST
                for px, py, _ in positions
            )
            far_from_bins = all(
                math.hypot(x - bx, y - by) >= MIN_BIN_DIST
                for bx, by in bin_positions
            )
            if far_from_objects and far_from_bins:
                positions.append((x, y, SPAWN_Z))
                break
        else:
            x = rng.uniform(*SPAWN_X_RANGE)
            y = rng.uniform(*SPAWN_Y_RANGE)
            positions.append((x, y, SPAWN_Z))
            log.warning("Could not place object %d with min-distance constraint; placed anyway.", len(positions))
    return positions


def sample_episode(all_objects, bin_mapping, bin_positions, n_objects, rng):
    active = all_objects[:n_objects]
    positions = sample_positions(len(active), bin_positions, rng)

    spawned = []
    for obj, (x, y, z) in zip(active, positions):
        spawned.append({
            "name":       obj["name"],
            "color":      obj["color"],
            "model_path": obj["model_path"],
            "x": x, "y": y, "z": z,
        })

    modes = list(MODE_WEIGHTS.keys())
    weights = list(MODE_WEIGHTS.values())
    mode = rng.choices(modes, weights=weights, k=1)[0]

    return {
        "mode":              mode,
        "color_bin_mapping": json.dumps(bin_mapping),
        "objects":           spawned,
    }


def run(sim_config_path, n_objects):
    sim_cfg       = load_sim_config(sim_config_path)
    all_objects   = build_object_defs(sim_cfg)
    bin_mapping   = build_bin_mapping(sim_cfg)
    bin_positions = build_bin_positions(sim_cfg)

    if not all_objects:
        log.error("No pickable objects found — exiting.")
        sys.exit(1)

    n_objects = min(n_objects, len(all_objects))
    log.info("Loaded %d pickable objects, will spawn %d per episode.", len(all_objects), n_objects)
    log.info("Bin mapping: %s", bin_mapping)
    log.info("Bin positions (excluded zone): %s", bin_positions)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((LISTEN_HOST, LISTEN_PORT))
    log.info("Listening on %s:%d", LISTEN_HOST, LISTEN_PORT)

    while True:
        try:
            raw, addr = sock.recvfrom(4096)
            msg = msgpack.unpackb(raw, raw=False)

            if msg.get("type") != "request_episode_config":
                log.warning("Unknown message type: %s", msg.get("type"))
                continue

            seed = random.randint(0, 2**31 - 1)
            rng  = random.Random(seed)

            episode = sample_episode(all_objects, bin_mapping, bin_positions, n_objects, rng)
            episode["seed"] = seed

            sock.sendto(msgpack.packb(episode), addr)

            names = [o["name"] for o in episode["objects"]]
            log.info(
                "Episode sent | seed=%d mode=%s objects=%s",
                seed,
                "bimanual" if episode["mode"] else "unimanual",
                names,
            )
            for o in episode["objects"]:
                log.info("  %s (%s) -> (%.3f, %.3f, %.3f)", o["name"], o["color"], o["x"], o["y"], o["z"])

        except Exception as e:
            log.error("Error: %s", e, exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Episode config server")
    parser.add_argument(
        "--sim-config",
        default=str(Path(__file__).parent.parent / "config" / "sim_config_franka.yaml"),
        help="Path to sim_config_franka.yaml",
    )
    parser.add_argument(
        "--n-objects", type=int, default=4,
        help="Number of objects to spawn per episode (1=smoke test, 4=full dataset)",
    )
    args = parser.parse_args()
    run(args.sim_config, args.n_objects)
