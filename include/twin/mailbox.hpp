#pragma once

// Correction handoff from reconciler thread (writer) to twin control loop (reader). Mutex is fine at ~100 Hz.

#include <cstdint>
#include <mutex>

constexpr int kTwinDof = 14;  // arm_left (7) + arm_right (7)

struct Correction {
    double   dq[kTwinDof]  = {};   // position delta to apply, radians
    double   ddq[kTwinDof] = {};   // velocity delta to apply, rad/s
    uint64_t computed_t_ns = 0;    // wall-clock ns
    uint8_t  regime        = 0;    // 0 = soft, 1 = hard resync
    bool     pending       = false;
};

class CorrectionMailbox {
public:
    void publish(const Correction& c) {
        std::lock_guard<std::mutex> lock(mtx_);
        slot_ = c;
        slot_.pending = true;
    }

    // takes the pending correction (if any) and clears it
    bool take(Correction& out) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!slot_.pending) return false;
        out = slot_;
        slot_.pending = false;
        return true;
    }

private:
    std::mutex mtx_;
    Correction slot_;
};
