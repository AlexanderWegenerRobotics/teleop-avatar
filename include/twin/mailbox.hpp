#pragma once

// Correction handoff between the reconciler thread (writer) and the twin's
// control loop (reader), applied at the top of the control loop's next tick
// (docs/twin_concept.md section 5: "the computed correction ... goes into a
// mailbox"). Guarded by a small mutex rather than hand-rolled multi-word
// lock-free atomics: both sides run at ~100 Hz with a sub-microsecond
// critical section, so contention is a non-issue and correctness is far
// easier to reason about than a lock-free compound slot would be.

#include <cstdint>
#include <mutex>

constexpr int kTwinDof = 14;  // arm_left (7) + arm_right (7)

struct Correction {
    double   dq[kTwinDof]  = {};   // position delta to apply, radians
    double   ddq[kTwinDof] = {};   // velocity delta to apply, rad/s
    uint64_t computed_t_ns = 0;    // wall-clock ns when this correction was computed
    uint8_t  regime        = 0;    // 0 = soft (dissolves into servo), 1 = hard resync + UI cue
    bool     pending       = false;
};

class CorrectionMailbox {
public:
    void publish(const Correction& c) {
        std::lock_guard<std::mutex> lock(mtx_);
        slot_ = c;
        slot_.pending = true;
    }

    // Consumer: takes the pending correction (if any) and clears it.
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
