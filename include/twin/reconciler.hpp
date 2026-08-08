#pragma once

// Reconciler: corrects the twin's predicted state against delayed real
// hardware telemetry (docs/twin_concept.md sections 4-5). Only meaningful
// for role: twin, and only functional in a build compiled WITH_MUJOCO (it
// needs a live Simulation to read/correct twin state and a private headless
// mjData to replay through).
//
// This header has zero MuJoCo dependency -- only a forward-declared
// Simulation and a type-erased `void*` for the scratch mjData -- so it is
// always includable and this class always compiles, in every build
// configuration. Constructing a Reconciler without WITH_MUJOCO throws a
// clear std::runtime_error instead of doing anything undefined; see
// reconciler.cpp's two implementations (WITH_MUJOCO / stub).

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

// One ring-buffer entry: the twin's own state and applied low-level ctrl at
// a given tick. ctrl is buffered alongside q/dq so the forward replay
// (Phi_fhat) can re-drive the scratch mjData with what the twin actually
// applied, not a re-derivation of the control law (section 4).
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
    // sim: the twin's own live Simulation. Reconciler does not own it and
    // only touches it through its public API (mjModelPtr/getDeviceState/
    // applyJointCorrection/replay*) -- section 5's single-source-of-truth
    // boundary: the reconciler's private replay mjData is a scratch
    // calculator, never an authority. device_names must total kTwinDof
    // (14) degrees of freedom at 7 each (2 arms), matching this codebase's
    // bimanual assumption.
    Reconciler(const ReconcilerConfig& cfg, Simulation* sim,
               std::vector<std::string> device_names = {"arm_left", "arm_right"});
    ~Reconciler();

    Reconciler(const Reconciler&)            = delete;
    Reconciler& operator=(const Reconciler&) = delete;

    void start();
    void stop();

    // Producer side (twin's control loop, ~reconciler_hz cadence): record
    // the twin's own just-computed state and applied ctrl this tick.
    // Lock-free, bounded-time -- safe to call from the control loop.
    void pushTwinState(uint64_t t_ns, const double q[kTwinDof],
                        const double dq[kTwinDof], const double ctrl[kTwinDof]);

    // Consumer side (twin's control loop, top of next tick): applies any
    // pending correction via Simulation::applyJointCorrection, honoring the
    // mailbox staleness guard (section 5) -- degrades gracefully to
    // open-loop prediction if the reconciler thread has stalled.
    void applyPendingCorrection();

    struct Stats {
        double          last_innovation_norm_rad = 0.0;
        double          measured_d_f_s           = 0.0;  // operational estimate, see reconciler.cpp
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

    SpscRingBuffer<TwinStateSample, 512> buffer_;   // ~5s @ 100Hz, well past buffer_horizon_s
    CorrectionMailbox                     mailbox_;

    socket_t          sock_    = kInvalidSocket;
    std::thread       thread_;
    std::atomic<bool> running_{false};

    mutable std::mutex stats_mtx_;
    Stats               stats_;

    void* replay_data_ = nullptr;  // mjData*, only valid/used when WITH_MUJOCO
};
