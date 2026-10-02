#pragma once

#include <thread>
#include <iostream>
#include <string>

#ifdef _WIN32
#include <windows.h>
#elif defined(__APPLE__)
#include <pthread.h>
#include <mach/mach_init.h>
#include <mach/mach_time.h>
#include <mach/thread_act.h>
#include <mach/thread_policy.h>
#else
#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#endif

// Pin a thread to a core and raise its priority where we own scheduling.
// With libfranka only affinity is set, libfranka owns the priority.
inline void set_realtime(std::thread& t, int cpu_core) {
    if (!t.joinable() || cpu_core < 0) return;

#ifdef _WIN32
    HANDLE h = static_cast<HANDLE>(t.native_handle());
    if (SetThreadAffinityMask(h, 1ULL << cpu_core) == 0)
        std::cout << "[WARN] rt: SetThreadAffinityMask(core " << cpu_core << ") failed." << std::endl;
    if (!SetThreadPriority(h, THREAD_PRIORITY_TIME_CRITICAL))
        std::cout << "[WARN] rt: SetThreadPriority(TIME_CRITICAL) failed." << std::endl;
#elif defined(__APPLE__)
    // no core pinning on macOS, time-constraint policy only (dev builds, not real-time)
    (void)cpu_core;

#ifndef WITH_FRANKA
    mach_timebase_info_data_t tb{};
    if (mach_timebase_info(&tb) == KERN_SUCCESS && tb.numer != 0) {
        const double ns_per_tick = static_cast<double>(tb.numer) / tb.denom;
        const auto to_ticks = [&](double ns) {
            return static_cast<uint32_t>(ns / ns_per_tick);
        };
        thread_time_constraint_policy_data_t pol{};
        pol.period      = to_ticks(1e6);   // ns
        pol.computation = to_ticks(5e5);
        pol.constraint  = to_ticks(1e6);
        pol.preemptible = 0;
        kern_return_t kr = thread_policy_set(
            pthread_mach_thread_np(t.native_handle()),
            THREAD_TIME_CONSTRAINT_POLICY,
            reinterpret_cast<thread_policy_t>(&pol),
            THREAD_TIME_CONSTRAINT_POLICY_COUNT);
        if (kr != KERN_SUCCESS)
            std::cout << "[WARN] rt: thread_policy_set(TIME_CONSTRAINT) failed (kr="
                      << kr << ") - loop will run at default priority." << std::endl;
    }
#endif

#else
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_core, &cpuset);
    int rc = pthread_setaffinity_np(t.native_handle(), sizeof(cpuset), &cpuset);
    if (rc != 0)
        std::cout << "[WARN] rt: pthread_setaffinity_np(core " << cpu_core
                  << ") failed (rc=" << rc << ")." << std::endl;

#ifndef WITH_FRANKA
    sched_param sp{};
    sp.sched_priority = 80;
    rc = pthread_setschedparam(t.native_handle(), SCHED_FIFO, &sp);
    if (rc != 0)
        std::cout << "[WARN] rt: pthread_setschedparam(SCHED_FIFO 80) failed (rc=" << rc
                  << ") - loop will run at SCHED_OTHER." << std::endl;
#endif
#endif
}

// Warn at startup if the process cannot get RT priority (kIgnore hides this).
inline void warn_if_no_realtime(const std::string& who) {
#if defined(__linux__)
    rlimit rl{};
    if (getrlimit(RLIMIT_RTPRIO, &rl) == 0 && rl.rlim_cur == 0) {
        std::cout << "[WARN] " << who
                  << ": RLIMIT_RTPRIO is 0 - this process cannot obtain real-time priority.\n"
                     "        libfranka is constructed with RealtimeConfig::kIgnore, so it will run the\n"
                     "        1 kHz control loop at SCHED_OTHER instead of failing. Expect\n"
                     "        control_command_success_rate < 1 and reflex aborts under load.\n"
                     "        Fix: add '<user> - rtprio 99' to /etc/security/limits.conf and re-login,\n"
                     "        and confirm a PREEMPT_RT kernel with: uname -v | grep PREEMPT_RT"
                  << std::endl;
    }
#else
    (void)who;
#endif
}
