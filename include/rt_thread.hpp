#pragma once

#include <thread>

#ifdef _WIN32
#include <windows.h>
#elif !defined(WITH_FRANKA)
#include <pthread.h>
#include <sched.h>
#endif

// Sets real-time priority and CPU affinity on the given thread.
// Skipped entirely on Linux+Franka builds — libfranka owns RT scheduling.
inline void set_realtime(std::thread& t, int cpu_core) {
#ifdef _WIN32
    HANDLE h = static_cast<HANDLE>(t.native_handle());
    SetThreadAffinityMask(h, 1ULL << cpu_core);
    SetThreadPriority(h, THREAD_PRIORITY_TIME_CRITICAL);
#elif !defined(WITH_FRANKA)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_core, &cpuset);
    pthread_setaffinity_np(t.native_handle(), sizeof(cpuset), &cpuset);
    sched_param sp{};
    sp.sched_priority = 80;
    pthread_setschedparam(t.native_handle(), SCHED_FIFO, &sp);
#endif
}
