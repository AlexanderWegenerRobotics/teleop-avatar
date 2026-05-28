#!/usr/bin/env python3
"""
shm_inject.py — Standalone stereo SHM test injector
=====================================================
Bypasses MuJoCo entirely.  Pumps solid-colour frames into the Windows named
file mappings that avatar_pipeline reads as its "mujoco" camera source.

Usage
-----
  # Mono mode  — inject green frames into avatar_cam
  python shm_inject.py

  # Stereo mode — inject red/blue frames into avatar_cam_left / avatar_cam_right
  python shm_inject.py --stereo

  # Stop after N frames (default: run until Ctrl-C)
  python shm_inject.py --frames 300

Then launch avatar_pipeline in another terminal:
  build\\Release\\avatar_pipeline.exe config\\config_local.yaml
and confirm it logs "Registering camera" for the expected channel(s).

SharedFrameBuffer layout (must match shared_memory.hpp)
--------------------------------------------------------
  uint32  write_idx     (atomic, treated as plain u32 by Python)
  uint32  frame_count   (atomic, treated as plain u32 by Python)
  uint32  width
  uint32  height
  uint8   slots[N_SLOTS][MAX_W * MAX_H * CHANNELS]

On Windows the shared name is "Local\\<name>" (leading slash stripped).
"""

import argparse
import mmap
import struct
import sys
import time

# ── Constants (must match shared_memory.hpp) ──────────────────────────────────

N_SLOTS   = 3
MAX_W     = 1280
MAX_H     = 960
CHANNELS  = 3

HEADER_FMT  = "<IIII"          # write_idx, frame_count, width, height
HEADER_SIZE = struct.calcsize(HEADER_FMT)
SLOT_SIZE   = MAX_W * MAX_H * CHANNELS
TOTAL_SIZE  = HEADER_SIZE + N_SLOTS * SLOT_SIZE


# ── SHM segment helper ────────────────────────────────────────────────────────

class ShmWriter:
    """
    Opens (or creates) a Windows named file mapping and exposes a write()
    method compatible with SharedMemoryWriter in shared_memory.hpp.
    """

    def __init__(self, posix_name: str, width: int, height: int):
        # Strip leading slash to get the Win32 name, e.g. "/avatar_cam" → "Local\avatar_cam"
        bare = posix_name.lstrip("/")
        win32_name = f"Local\\{bare}"

        self._mm = mmap.mmap(-1, TOTAL_SIZE, tagname=win32_name)
        self._width  = width
        self._height = height
        self._write_idx   = 0
        self._frame_count = 0

        # Initialise the header
        self._mm.seek(0)
        self._mm.write(struct.pack(HEADER_FMT, 0, 0, width, height))
        print(f"[ShmWriter] Opened '{win32_name}'  ({width}x{height})")

    def write(self, frame_bytes: bytes):
        slot = self._write_idx % N_SLOTS
        offset = HEADER_SIZE + slot * SLOT_SIZE
        self._mm.seek(offset)
        self._mm.write(frame_bytes[:SLOT_SIZE])

        self._write_idx   += 1
        self._frame_count += 1

        # Update header atomically enough for a test tool
        self._mm.seek(0)
        self._mm.write(struct.pack(HEADER_FMT,
                                   self._write_idx, self._frame_count,
                                   self._width, self._height))

    def close(self):
        self._mm.close()


def solid_frame(width: int, height: int, r: int, g: int, b: int) -> bytes:
    """Return a solid-colour RGB frame (top-down, 8-bit per channel)."""
    pixel = bytes([r, g, b])
    return pixel * (width * height)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Inject test frames into SHM for avatar_pipeline testing")
    parser.add_argument("--stereo",  action="store_true",
                        help="Stereo mode: write left (red) + right (blue) instead of mono (green)")
    parser.add_argument("--width",   type=int, default=1280)
    parser.add_argument("--height",  type=int, default=960)
    parser.add_argument("--fps",     type=int, default=30)
    parser.add_argument("--frames",  type=int, default=0,
                        help="Stop after N frames (0 = run forever)")
    args = parser.parse_args()

    W, H = args.width, args.height
    period = 1.0 / args.fps

    writers: list[tuple[ShmWriter, bytes]] = []

    try:
        if args.stereo:
            print("[shm_inject] Mode: STEREO  (left=red  right=blue)")
            writers.append((ShmWriter("/avatar_cam_left",  W, H), solid_frame(W, H, 200, 50,  50)))
            writers.append((ShmWriter("/avatar_cam_right", W, H), solid_frame(W, H, 50,  50, 200)))
        else:
            print("[shm_inject] Mode: MONO  (green)")
            writers.append((ShmWriter("/avatar_cam", W, H), solid_frame(W, H, 50, 200, 50)))

        print(f"[shm_inject] Pushing frames at {args.fps} fps — Ctrl-C to stop")

        count   = 0
        t_start = time.monotonic()
        t_next  = t_start

        while True:
            for writer, frame in writers:
                writer.write(frame)

            count += 1

            if args.frames and count >= args.frames:
                break

            # Rate limiting
            t_next += period
            now = time.monotonic()
            if t_next > now:
                time.sleep(t_next - now)

            # Status line every second
            elapsed = time.monotonic() - t_start
            if count % args.fps == 0:
                actual_fps = count / elapsed if elapsed > 0 else 0
                channels = "left+right" if args.stereo else "mono"
                print(f"\r[shm_inject] {count:6d} frames  {actual_fps:.1f} fps  channels={channels}    ",
                      end="", flush=True)

    except KeyboardInterrupt:
        print("\n[shm_inject] Interrupted.")
    finally:
        for writer, _ in writers:
            writer.close()

    elapsed = time.monotonic() - t_start
    print(f"[shm_inject] Done — {count} frames in {elapsed:.1f}s")


if __name__ == "__main__":
    if sys.platform != "win32":
        print("ERROR: This script uses Windows named file mappings (mmap tagname=).")
        print("       Run it on the Windows host where avatar_pipeline.exe lives.")
        sys.exit(1)

    main()
