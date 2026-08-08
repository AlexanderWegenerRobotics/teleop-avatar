#pragma once

// Fixed-capacity SPSC ring buffer B (docs/twin_concept.md section 3):
// "fixed array + two atomic indices". Exactly one producer thread (the
// twin's control loop, via Reconciler::pushTwinState) and exactly one
// consumer thread (the reconciler thread) -- never more, by construction in
// Reconciler. T must be trivially copyable. Capacity must be a power of two.
//
// The consumer indexes by position (oldest-first) rather than popping,
// since the reconciler needs random access for matched-phase lookup and
// windowed replay, not a queue. If the producer wraps around while the
// consumer is mid-scan of a very old index, that index silently reflects
// newer data instead -- acceptable here given Capacity is sized well past
// buffer_horizon_s at the expected push rate (see reconciler_config.yaml).

#include <array>
#include <atomic>
#include <cstddef>

template <typename T, std::size_t Capacity>
class SpscRingBuffer {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of two");

public:
    // Producer only.
    void push(const T& item) {
        std::size_t head = head_.load(std::memory_order_relaxed);
        buf_[head & (Capacity - 1)] = item;
        head_.store(head + 1, std::memory_order_release);
    }

    // Consumer only. Number of valid entries currently visible.
    std::size_t size() const {
        std::size_t head = head_.load(std::memory_order_acquire);
        return head < Capacity ? head : Capacity;
    }

    // Consumer only. index 0 = oldest currently-valid entry, size()-1 = newest.
    const T& at(std::size_t index) const {
        std::size_t head = head_.load(std::memory_order_acquire);
        std::size_t lo   = head > Capacity ? head - Capacity : 0;
        return buf_[(lo + index) & (Capacity - 1)];
    }

private:
    std::array<T, Capacity> buf_{};
    std::atomic<std::size_t> head_{0};
};
