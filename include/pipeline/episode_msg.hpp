#pragma once

#include <cstdint>
#include <string>

#include <msgpack.hpp>

// Sent from Avatar → pipeline process over UDP whenever an episode boundary occurs.
// Both sender (avatar.cpp) and receiver (episode_controller.cpp) use this struct.
struct EpisodeEventMsg {
    std::string type;            // "episode_start" | "episode_end"
    std::string session_id;      // avatar session timestamp string
    int32_t     episode_index = -1;  // zero-based, matches avatar's episode folder index
    std::string reason;          // for episode_end: "operator_idle", "operator_pause", "reset_all", …
    std::string log_dir;         // absolute path to this episode's folder (set by avatar on episode_start)

    MSGPACK_DEFINE_MAP(type, session_id, episode_index, reason, log_dir)
};
