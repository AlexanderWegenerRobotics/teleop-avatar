#include "twin/reconciler.hpp"

#include <cstring>
#include <stdexcept>

ReconcilerConfig ReconcilerConfig::load(const YAML::Node& y) {
    ReconcilerConfig cfg;
    if (!y) return cfg;
    cfg.listen_port      = y["listen_port"].as<int>(cfg.listen_port);
    cfg.buffer_horizon_s = y["buffer_horizon_s"].as<double>(cfg.buffer_horizon_s);
    cfg.correction_tau_s = y["correction_tau_ms"].as<double>(cfg.correction_tau_s * 1000.0) / 1000.0;
    cfg.epsilon_soft_rad = y["epsilon_soft_rad"].as<double>(cfg.epsilon_soft_rad);
    cfg.epsilon_hard_rad = y["epsilon_hard_rad"].as<double>(cfg.epsilon_hard_rad);
    cfg.reconciler_hz    = y["reconciler_hz"].as<int>(cfg.reconciler_hz);
    cfg.mailbox_stale_s  = y["mailbox_stale_ms"].as<double>(cfg.mailbox_stale_s * 1000.0) / 1000.0;
    return cfg;
}

#ifndef WITH_MUJOCO

// ── No-MuJoCo stub ──────────────────────────────────────────────────────────
// Keeps the symbol set complete for every build configuration, but a
// Reconciler must never actually be constructed here: Avatar only builds one
// for role: twin, and role: twin requires WITH_MUJOCO. This throws instead
// of doing anything undefined, per the "don't crash without MuJoCo" build
// requirement -- no MuJoCo header is included in this branch at all.

Reconciler::Reconciler(const ReconcilerConfig&, Simulation*, std::vector<std::string>) {
    throw std::runtime_error(
        "Reconciler requires a MuJoCo build (WITH_MUJOCO). This binary was "
        "compiled without MuJoCo support, so role: twin is unavailable.");
}
Reconciler::~Reconciler() = default;
void Reconciler::start() {}
void Reconciler::stop() {}
void Reconciler::pushTwinState(uint64_t, const double*, const double*, const double*) {}
void Reconciler::applyPendingCorrection() {}
Reconciler::Stats Reconciler::getStats() const { return {}; }
void Reconciler::runReconcilerThread() {}
void Reconciler::handleTelemetry(const TwinTelemetryMsg&) {}

#else  // WITH_MUJOCO

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

#include <mujoco/mujoco.h>

#include "sim_env/simulation.hpp"

namespace {
constexpr int kArmDof = 7;

double secondsFromNs(uint64_t ns) { return static_cast<double>(ns) * 1e-9; }

uint64_t nowNs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}
}  // namespace

Reconciler::Reconciler(const ReconcilerConfig& cfg, Simulation* sim,
                        std::vector<std::string> device_names)
    : cfg_(cfg), sim_(sim), device_names_(std::move(device_names)) {
    if (!sim_) throw std::runtime_error("Reconciler: Simulation pointer is null");
    if (device_names_.size() * static_cast<size_t>(kArmDof) != static_cast<size_t>(kTwinDof))
        throw std::runtime_error("Reconciler: device_names must total kTwinDof (14) DoF at 7 each");

    const mjModel* model = sim_->mjModelPtr();
    if (!model) throw std::runtime_error("Reconciler: Simulation has no mjModel loaded");

    mjData* rd = mj_makeData(const_cast<mjModel*>(model));
    if (!rd) throw std::runtime_error("Reconciler: mj_makeData failed for replay scratch data");
    replay_data_ = rd;

    sock_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock_ == kInvalidSocket) {
        mj_deleteData(rd);
        replay_data_ = nullptr;
        throw std::runtime_error("Reconciler: failed to create telemetry listen socket");
    }

    sockaddr_in bind_addr{};
    bind_addr.sin_family      = AF_INET;
    bind_addr.sin_addr.s_addr = INADDR_ANY;
    bind_addr.sin_port        = htons(static_cast<uint16_t>(cfg_.listen_port));
    if (::bind(sock_, reinterpret_cast<sockaddr*>(&bind_addr), sizeof(bind_addr)) != 0) {
        close_socket(sock_);
        sock_ = kInvalidSocket;
        mj_deleteData(rd);
        replay_data_ = nullptr;
        throw std::runtime_error("Reconciler: failed to bind telemetry listen port " +
                                  std::to_string(cfg_.listen_port));
    }

    // Short recv timeout so runReconcilerThread() can re-check running_
    // promptly on stop() without a dedicated wakeup mechanism.
#ifdef _WIN32
    DWORD timeout_ms = 20;
    setsockopt(sock_, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&timeout_ms), sizeof(timeout_ms));
#else
    struct timeval tv{};
    tv.tv_sec  = 0;
    tv.tv_usec = 20000;
    setsockopt(sock_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif

    std::cout << "[RECONCILER-INFO] Listening for twin telemetry on port " << cfg_.listen_port << std::endl;
}

Reconciler::~Reconciler() {
    stop();
    if (replay_data_) mj_deleteData(static_cast<mjData*>(replay_data_));
    if (sock_ != kInvalidSocket) close_socket(sock_);
}

void Reconciler::start() {
    if (running_.exchange(true)) return;
    thread_ = std::thread(&Reconciler::runReconcilerThread, this);
}

void Reconciler::stop() {
    if (!running_.exchange(false)) return;
    if (thread_.joinable()) thread_.join();
}

void Reconciler::pushTwinState(uint64_t t_ns, const double q[kTwinDof],
                                const double dq[kTwinDof], const double ctrl[kTwinDof]) {
    TwinStateSample s;
    s.t_ns = t_ns;
    std::memcpy(s.q, q, sizeof(s.q));
    std::memcpy(s.dq, dq, sizeof(s.dq));
    std::memcpy(s.ctrl, ctrl, sizeof(s.ctrl));
    buffer_.push(s);
}

void Reconciler::applyPendingCorrection() {
    Correction c;
    if (!mailbox_.take(c)) return;

    uint64_t now = nowNs();
    double age_s = (now > c.computed_t_ns) ? secondsFromNs(now - c.computed_t_ns) : 0.0;
    if (age_s > cfg_.mailbox_stale_s) {
        // Stale guard (section 5): reconciler thread stalled -- degrade
        // gracefully to open-loop prediction instead of applying an
        // ancient correction.
        return;
    }

    for (size_t d = 0; d < device_names_.size(); ++d) {
        std::vector<double> dq (c.dq  + d * kArmDof, c.dq  + (d + 1) * kArmDof);
        std::vector<double> ddq(c.ddq + d * kArmDof, c.ddq + (d + 1) * kArmDof);
        sim_->applyJointCorrection(device_names_[d], dq, ddq);
    }
}

Reconciler::Stats Reconciler::getStats() const {
    std::lock_guard<std::mutex> lock(stats_mtx_);
    return stats_;
}

void Reconciler::runReconcilerThread() {
    uint8_t raw[sizeof(TwinTelemetryMsg) + 64];
    while (running_.load()) {
        sockaddr_in sender{};
        socklen_t   sender_len = sizeof(sender);
        int n = recvfrom(sock_, reinterpret_cast<char*>(raw), sizeof(raw), 0,
                          reinterpret_cast<sockaddr*>(&sender), &sender_len);
        if (n == static_cast<int>(sizeof(TwinTelemetryMsg))) {
            TwinTelemetryMsg msg;
            std::memcpy(&msg, raw, sizeof(msg));
            handleTelemetry(msg);
        }
        // n <= 0 on timeout/no-data -- loop back around and re-check running_.
    }
}

void Reconciler::handleTelemetry(const TwinTelemetryMsg& msg) {
    uint64_t now_ns = nowNs();
    uint64_t t_s_ns = msg.header.timestamp_ns;  // hardware sample time (docs/twin_concept.md section 2)

    {
        std::lock_guard<std::mutex> lock(stats_mtx_);
        stats_.packets_received++;
        stats_.measured_d_b_s = (now_ns > t_s_ns) ? secondsFromNs(now_ns - t_s_ns) : 0.0;
    }

    double y[kTwinDof], y_dot[kTwinDof];
    for (int i = 0; i < kArmDof; ++i) {
        y[i]               = msg.q_left[i];
        y[kArmDof + i]     = msg.q_right[i];
        y_dot[i]           = msg.dq_left[i];
        y_dot[kArmDof + i] = msg.dq_right[i];
    }

    std::size_t n = buffer_.size();
    if (n == 0) return;  // nothing buffered yet -- can't reconcile

    // ── Matched-phase lookup (section 4): find the buffered twin sample
    // whose own state is closest to y in joint space -- the reconciler's
    // best estimate of "twin state at t_s". We don't have a direct
    // timestamp-exchange channel to measure d_f independently (that needs
    // round-trip instrumentation at the avatar's command receiver -- a
    // natural follow-up); d_f is instead estimated operationally as
    // t_s - matched.t_ns, consistent with x(t_s) ~= x_hat(t_s - d_f).
    std::size_t best_idx  = 0;
    double      best_dist = -1.0;
    for (std::size_t i = 0; i < n; ++i) {
        const TwinStateSample& s = buffer_.at(i);
        double dist = 0.0;
        for (int k = 0; k < kTwinDof; ++k) {
            double diff = s.q[k] - y[k];
            dist += diff * diff;
        }
        if (best_dist < 0.0 || dist < best_dist) {
            best_dist = dist;
            best_idx  = i;
        }
    }
    const TwinStateSample& matched = buffer_.at(best_idx);
    double d_f_s = (t_s_ns > matched.t_ns) ? secondsFromNs(t_s_ns - matched.t_ns) : 0.0;

    // Innovation e = y - x_hat(t_s - d_f), joint space (catches
    // nullspace/elbow drift invisible to an EE-only comparison).
    double e_norm_sq = 0.0;
    for (int k = 0; k < kTwinDof; ++k) {
        double diff = y[k] - matched.q[k];
        e_norm_sq += diff * diff;
    }
    double e_norm = std::sqrt(e_norm_sq);

    // ── Forward replay Phi_fhat: seed the scratch mjData at y, then
    // re-step through the buffered ctrl history from the matched sample to
    // now -- reusing the twin's actually-applied low-level commands rather
    // than re-deriving the control law inside the reconciler.
    mjData* rd = static_cast<mjData*>(replay_data_);
    for (size_t d = 0; d < device_names_.size(); ++d) {
        std::vector<double> q0 (y     + d * kArmDof, y     + (d + 1) * kArmDof);
        std::vector<double> dq0(y_dot + d * kArmDof, y_dot + (d + 1) * kArmDof);
        sim_->replaySeed(rd, device_names_[d], q0, dq0);
    }
    for (std::size_t i = best_idx; i < n; ++i) {
        const TwinStateSample& s = buffer_.at(i);
        for (size_t d = 0; d < device_names_.size(); ++d) {
            std::vector<double> ctrl(s.ctrl + d * kArmDof, s.ctrl + (d + 1) * kArmDof);
            sim_->replaySetCtrl(rd, device_names_[d], ctrl);
        }
        sim_->replayAdvance(rd);  // one MuJoCo step per buffered sample in the window
    }

    double x_tilde[kTwinDof]     = {};
    double x_tilde_dot[kTwinDof] = {};
    for (size_t d = 0; d < device_names_.size(); ++d) {
        auto q  = sim_->replayReadQ(rd, device_names_[d]);
        auto dq = sim_->replayReadDq(rd, device_names_[d]);
        for (size_t i = 0; i < q.size()  && i < static_cast<size_t>(kArmDof); ++i)
            x_tilde[d * kArmDof + i] = q[i];
        for (size_t i = 0; i < dq.size() && i < static_cast<size_t>(kArmDof); ++i)
            x_tilde_dot[d * kArmDof + i] = dq[i];
    }

    // ── Correction (section 4): filtered pull of the twin's *current*
    // state toward x_tilde. The newest buffered sample is used as the
    // proxy for "current" twin state (it lags true-current by at most one
    // control tick -- negligible relative to correction_tau_s).
    const TwinStateSample& x_hat_now = buffer_.at(n - 1);

    bool hard = e_norm > cfg_.epsilon_hard_rad;
    // K = 1 - e^{-Delta_t/tau}; Delta_t ~= 1/reconciler_hz between ticks.
    // Below epsilon_hard, corrections dissolve into this servo pull and
    // fall below epsilon_soft/visual perception at steady state. Above
    // epsilon_hard the model broke (unexpected contact, safety stop, joint
    // limit) -- K=1 forces an immediate resync, flagged for a visible UI cue
    // via regime=Hard (see Stats::last_regime / hard_resync_count).
    double K = hard ? 1.0
                     : (1.0 - std::exp(-1.0 / (std::max(1, cfg_.reconciler_hz) * cfg_.correction_tau_s)));

    Correction corr;
    corr.computed_t_ns = now_ns;
    corr.regime         = hard ? 1 : 0;
    corr.pending         = true;
    for (int k = 0; k < kTwinDof; ++k) {
        corr.dq[k]  = K * (x_tilde[k]     - x_hat_now.q[k]);
        corr.ddq[k] = K * (x_tilde_dot[k] - x_hat_now.dq[k]);
    }
    mailbox_.publish(corr);

    std::lock_guard<std::mutex> lock(stats_mtx_);
    stats_.last_innovation_norm_rad = e_norm;
    stats_.measured_d_f_s            = d_f_s;
    stats_.last_regime               = hard ? ReconcileRegime::Hard : ReconcileRegime::Soft;
    if (hard) stats_.hard_resync_count++;
}

#endif  // WITH_MUJOCO
