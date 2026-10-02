#pragma once

#include <cstdint>
#include <string>

#include <msgpack.hpp>

// Episode boundary message, avatar -> pipeline over UDP.
struct EpisodeEventMsg {
    std::string type;            // "episode_start" | "episode_end"
    std::string session_id;
    int32_t     episode_index = -1;  // zero-based
    std::string reason;          // episode_end only
    std::string log_dir;         // absolute path, set on episode_start

    MSGPACK_DEFINE_MAP(type, session_id, episode_index, reason, log_dir)
};
