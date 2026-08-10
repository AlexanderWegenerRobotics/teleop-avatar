#pragma once

#include <thread>
#include <iostream>
#include <string>

#ifdef _WIN32
#include <windows.h>
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
