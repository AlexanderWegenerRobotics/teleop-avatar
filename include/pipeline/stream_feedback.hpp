#pragma once

#include <cstdint>

struct StreamFeedbackMsg {
    uint64_t timestamp_ns;
    float    loss_rate;
    float    jitter_ms;
    float    rtt_ms;
    uint32_t packets_received;
    uint32_t packets_lost;
    uint32_t frames_received;
};

struct FrameTimestamp {
    uint64_t frame_number;
    uint64_t sender_timestamp_ns;
};

#ifdef _MSC_VER
#pragma pack(push, 1)
#endif
struct StreamStatusMsg {
    uint32_t magic;
    uint8_t  state;
    uint64_t timestamp_ns;
}
#ifndef _MSC_VER
__attribute__((packed))
#endif
;
#ifdef _MSC_VER
#pragma pack(pop)
#endif

static constexpr uint32_t kStreamStatusMagic = 0x53545354;