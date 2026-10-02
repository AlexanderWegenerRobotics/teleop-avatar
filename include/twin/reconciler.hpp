#pragma once

// Corrects the twin's predicted state against delayed hardware telemetry. Needs WITH_MUJOCO, otherwise the ctor throws.

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "network/platform_socket.hpp"
#include "twin/mailbox.hpp"
#include "twin/ring_buffer.hpp"
#include "twin/telemetry_msg.hpp"

class Simulation;

// Twin state + applied ctrl at one tick; ctrl is kept so the replay re-drives what was actually applied.
struct TwinStateSample {
    uint64_t t_ns              = 0;
    double   q[kTwinDof]       = {};
    double   dq[kTwinDof]      = {};
    double   ctrl[kTwinDof]    = {};
};

enum class ReconcileRegime : uint8_t { Soft = 0, Hard = 1 };

struct ReconcilerConfig {
    int    listen_port      = 7400;
    double buffer_horizon_s = 1.0;
    double correction_tau_s = 0.3;
    double epsilon_soft_rad = 0.035;
    double epsilon_hard_rad = 0.175;
    int    reconciler_hz    = 100;
    double mailbox_stale_s  = 0.05;

    static ReconcilerConfig load(const YAML::Node& reconciler_yaml);
};

class Reconciler {
public:
    // sim is not owned. device_names must add up to kTwinDof (2 arms x 7).
    Reconciler(const ReconcilerConfig& cfg, Simulation* sim,
               std::vector<std::string> device_names = {"arm_left", "arm_right"});
    ~Reconciler();

    Reconciler(const Reconciler&)            = delete;
    Reconciler& operator=(const Reconciler&) = delete;

    void start();
    void stop();

    // called from the control loop; lock-free and bounded-time
    void pushTwinState(uint64_t t_ns, const double q[kTwinDof],
                        const double dq[kTwinDof], const double ctrl[kTwinDof]);

    // called at top of the control loop tick; stale corrections are dropped
    void applyPendingCorrection();

    struct Stats {
        double          last_innovation_norm_rad = 0.0;
        double          measured_d_f_s           = 0.0;
        double          measured_d_b_s            = 0.0;
        ReconcileRegime last_regime               = ReconcileRegime::Soft;
        uint64_t        hard_resync_count          = 0;
        uint64_t        packets_received           = 0;
    };
    Stats getStats() const;

private:
    void runReconcilerThread();
    void handleTelemetry(const TwinTelemetryMsg& msg);

private:
    ReconcilerConfig          cfg_;
    Simulation*               sim_;
    std::vector<std::string>  device_names_;

    SpscRingBuffer<TwinStateSample, 512> buffer_;   // ~5s @ 100Hz
    CorrectionMailbox                     mailbox_;

    socket_t          sock_    = kInvalidSocket;
    std::thread       thread_;
    std::atomic<bool> running_{false};

    mutable std::mutex stats_mtx_;
    Stats               stats_;

    void* replay_data_ = nullptr;  // mjData*, only used WITH_MUJOCO
};
