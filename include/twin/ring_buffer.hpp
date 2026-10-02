#pragma once

// Fixed-size SPSC ring buffer with random access by index. One producer, one consumer. Capacity must be a power of two.

#include <array>
#include <atomic>
#include <cstddef>

template <typename T, std::size_t Capacity>
class SpscRingBuffer {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of two");

public:
    // producer only
    void push(const T& item) {
        std::size_t head = head_.load(std::memory_order_relaxed);
        buf_[head & (Capacity - 1)] = item;
        head_.store(head + 1, std::memory_order_release);
    }

    // consumer only
    std::size_t size() const {
        std::size_t head = head_.load(std::memory_order_acquire);
        return head < Capacity ? head : Capacity;
    }

    // consumer only. index 0 = oldest, size()-1 = newest
    const T& at(std::size_t index) const {
        std::size_t head = head_.load(std::memory_order_acquire);
        std::size_t lo   = head > Capacity ? head - Capacity : 0;
        return buf_[(lo + index) & (Capacity - 1)];
    }

private:
    std::array<T, Capacity> buf_{};
    std::atomic<std::size_t> head_{0};
};
