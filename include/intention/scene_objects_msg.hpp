#pragma once

#include <cstdint>
#include <map>
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
    // Per-arm CommandAuthority (common.hpp), keyed by device name
    // ("arm_left" / "arm_right"). A map rather than a vector so the consumer
    // never has to know arm_instances' order, and so an arm a given build does
    // not have simply is not a key.
    //
    // Re-sent every tick rather than published as an event, because this is a
    // safety mutex and the orchestrator has to fail closed on a dropped packet.
    // At 100 Hz the current value re-asserts a hundred times a second, so a
    // lost datagram costs 10 ms of staleness instead of leaving the policy
    // acting on an authority that was revoked.
    //
    // Consumers read it with a default -- "authority" absent means an avatar
    // that predates this field, not "nobody has authority".
    std::map<std::string, uint8_t> authority;
    // Avatar SysState, so a consumer can read it without the reliable command
    // channel. That channel is point-to-point (one remote_ip/remote_port), and
    // with the VR interface and the orchestrator both live it can only serve
    // one of them -- the other sees no heartbeat and concludes the avatar is
    // dead. This socket is fire-and-forget to its own configured host/port, so
    // it has no such contention.
    //
    // 255 = SysState::UNDEFINED, which is also what an avatar predating this
    // field looks like to a consumer using .get("state", 255).
    uint8_t                        state = 255;
    // Head pan/tilt, radians, for the same reason as `state`: the head's own
    // transmission is point-to-point too, and the VR interface needs it. A
    // consumer that only wants the head POSE (gaze projection geometry) can
    // take it from here and leave the head channel to the interface.
    float                          head_pan  = 0.0f;
    float                          head_tilt = 0.0f;

    MSGPACK_DEFINE_MAP(frame_id, tick_id, timestamp_ns, slots, authority, state,
                       head_pan, head_tilt)
};
