#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <msgpack.hpp>

// Scene object geometry sent Avatar -> orchestrator each StateSnapshot (sim state now, perception later).
struct SceneObjectSlot {
    std::string        name;
    uint8_t             type;          // SlotType (intention_sample.hpp)
    std::vector<float>  position;      // world, (x, y, z)
    std::vector<float>  quaternion;    // world, (w, x, y, z)
    std::vector<float>  half_extents;  // zero if unknown / not a box

    MSGPACK_DEFINE_MAP(name, type, position, quaternion, half_extents)
};

struct SceneObjectsMsg {
    // frame_id = render frame (join key for gaze/images), tick_id = avatar control tick (orchestrator pacing)
    uint64_t                      frame_id     = 0;
    uint64_t                      tick_id      = 0;
    uint64_t                      timestamp_ns = 0;
    std::vector<SceneObjectSlot>  slots;
    // per-arm CommandAuthority by device name, resent every tick so a dropped packet fails closed
    std::map<std::string, uint8_t> authority;
    // SysState, 255 = UNDEFINED / old avatar
    uint8_t                        state = 255;
    // head pan/tilt, rad
    float                          head_pan  = 0.0f;
    float                          head_tilt = 0.0f;

    MSGPACK_DEFINE_MAP(frame_id, tick_id, timestamp_ns, slots, authority, state,
                       head_pan, head_tilt)
};
