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

// ─────────────────────────────────────────────────────────────────────────────
//  set_realtime — pin a thread to a core and, where we own scheduling, raise it
//  to real-time priority.
//
//  Windows                : affinity + THREAD_PRIORITY_TIME_CRITICAL (sim build).
//  Linux, no libfranka    : affinity + SCHED_FIFO 80. Nobody else sets priority.
//  Linux + libfranka      : affinity ONLY.
//  macOS                  : NO affinity (see below) + time-constraint policy.
//
//  The Linux+Franka case used to compile to nothing at all, on the assumption
//  that "libfranka owns RT scheduling". That is only half true: libfranka raises
//  the priority of whichever thread calls Robot::control(), but it never touches
//  affinity — and with RealtimeConfig::kIgnore it will silently continue at
//  SCHED_OTHER if it cannot raise priority at all. So the 1 kHz loop was left
//  both unpinned and (potentially) non-RT. We still must not call
//  pthread_setschedparam here or we would be fighting libfranka for the policy,
//  but pinning is ours to do and is the difference between the control loop
//  living on a quiet core and sharing one with the network stack.
// ─────────────────────────────────────────────────────────────────────────────
inline void set_realtime(std::thread& t, int cpu_core) {
    if (!t.joinable() || cpu_core < 0) return;

#ifdef _WIN32
    HANDLE h = static_cast<HANDLE>(t.native_handle());
    if (SetThreadAffinityMask(h, 1ULL << cpu_core) == 0)
        std::cout << "[WARN] rt: SetThreadAffinityMask(core " << cpu_core << ") failed." << std::endl;
    if (!SetThreadPriority(h, THREAD_PRIORITY_TIME_CRITICAL))
        std::cout << "[WARN] rt: SetThreadPriority(TIME_CRITICAL) failed." << std::endl;
#elif defined(__APPLE__)
    // macOS has neither cpu_set_t nor pthread_setaffinity_np. Its nearest
    // equivalent, THREAD_AFFINITY_POLICY, expresses "these threads share a
    // cache" rather than "run on core N", and on Apple Silicon it is ignored
    // outright. There is no way to pin a thread here, so we do not pretend to.
    //
    // What macOS does offer is THREAD_TIME_CONSTRAINT_POLICY, which is the real
    // analogue of SCHED_FIFO: it tells the scheduler this thread needs
    // `computation` time out of every `period`. That is worth setting for the
    // same reason the Linux branch sets SCHED_FIFO -- in a sim build we own the
    // loop timing.
    //
    // This path exists so the project builds and runs on a laptop for
    // development. It is NOT a real-time platform: expect the loop-rate
    // quantisation described in Robot::control to be worse here, not better.
    (void)cpu_core;

#ifndef WITH_FRANKA
    mach_timebase_info_data_t tb{};
    if (mach_timebase_info(&tb) == KERN_SUCCESS && tb.numer != 0) {
        // Express the 1 kHz control period in mach absolute-time units.
        const double ns_per_tick = static_cast<double>(tb.numer) / tb.denom;
        const auto to_ticks = [&](double ns) {
            return static_cast<uint32_t>(ns / ns_per_tick);
        };
        thread_time_constraint_policy_data_t pol{};
        pol.period      = to_ticks(1e6);   // 1 ms nominal loop period
        pol.computation = to_ticks(5e5);   // ask for 0.5 ms of it
        pol.constraint  = to_ticks(1e6);   // must finish within the period
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
    // Sim-only: we are the ones driving the loop, so we set the policy too.
    sched_param sp{};
    sp.sched_priority = 80;
    rc = pthread_setschedparam(t.native_handle(), SCHED_FIFO, &sp);
    if (rc != 0)
        std::cout << "[WARN] rt: pthread_setschedparam(SCHED_FIFO 80) failed (rc=" << rc
                  << ") - loop will run at SCHED_OTHER." << std::endl;
#endif
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
//  warn_if_no_realtime — loud check for the failure mode RealtimeConfig::kIgnore
//  hides.
//
//  kEnforce would make libfranka throw when the process cannot obtain real-time
//  priority. We deliberately run kIgnore (deployment constraint), which means
//  that failure is silent and shows up much later as
//  control_command_success_rate < 1 followed by a reflex abort. This gives us the
//  loud version of the same diagnosis at startup, without changing the config.
// ─────────────────────────────────────────────────────────────────────────────
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
