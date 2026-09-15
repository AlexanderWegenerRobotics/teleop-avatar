#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <msgpack.hpp>

// Sent from Avatar -> orchestrator whenever a StateSnapshot is built (see
// Avatar::updateStateMachine), so an external consumer gets the same
// candidate/bin geometry the C++ intention pipeline uses, without depending
// on IntentionBuffer's gaze fusion. Privileged sim state today (queried via
// Simulation::getFreeBodyPose); a real perception system's equivalent
// publish on hardware, so the orchestrator should treat this as one
// interchangeable ObjectSource, not sim-specific.
struct SceneObjectSlot {
    std::string        name;
    uint8_t             type;          // SlotType (intention_sample.hpp)
    std::vector<float>  position;      // world position, len 3 (x, y, z)
    std::vector<float>  quaternion;    // world orientation, len 4 (w, x, y, z)
    std::vector<float>  half_extents;  // len 3; zero vector if unknown/not a box

    MSGPACK_DEFINE_MAP(name, type, position, quaternion, half_extents)
};

struct SceneObjectsMsg {
    // Two different clocks, deliberately separate.
    //   frame_id  visual frame counter (Simulation::stream_frame_count_, paced
    //             at rendering.fps). Gaze pixels and camera images only mean
    //             anything against the frame they were captured in, so this
    //             stays the join key for GazeSampleMsg and for the policy.
    //   tick_id   Avatar control-loop counter, one per avatar tick. This is the
    //             orchestrator's pacing signal. Keeping it apart from frame_id
    //             is what stops the command rate being hostage to the render
    //             rate -- they were the same number, so commands ran at
    //             rendering.fps (15 Hz) while the avatar looped at 100.
    uint64_t                      frame_id     = 0;
    uint64_t                      tick_id      = 0;
    uint64_t                      timestamp_ns = 0;
    std::vector<SceneObjectSlot>  slots;

    MSGPACK_DEFINE_MAP(frame_id, tick_id, timestamp_ns, slots)
};
