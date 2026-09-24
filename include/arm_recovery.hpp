#pragma once

#include <atomic>
#include <mutex>
#include <string>
#include <Eigen/Dense>
#include "common.hpp"

enum class RecoveryMode : uint8_t {
    NONE            = 0,
    MOVING_TO_SAFE  = 1,   // executing autonomous joint motion
    WAITING_ACK     = 2,   // motion done, waiting for operator resume
};

enum class RecoveryTrigger : uint8_t {
    OPERATOR_RESET  = 0,
    JOINT_LIMIT     = 1,
    EXTERNAL_FORCE  = 2,
    VELOCITY_ERROR  = 3,
};

struct RecoveryRequest {
    bool        valid   = false;
    RecoveryTrigger trigger = RecoveryTrigger::OPERATOR_RESET;
    Vector7     target_q;   // joint config to move to
    bool        skip_motion = false;  // for velocity errors — go straight to WAITING_ACK
};

class ArmRecovery {
public:
    explicit ArmRecovery(const std::string& device_name)
        : name_(device_name) {}

    // --- avatar-side interface ---
    void requestRecovery(RecoveryTrigger trigger, const Vector7& target_q, bool skip_motion = false) {
        RecoveryRequest req;
        req.valid       = true;
        req.trigger     = trigger;
        req.target_q    = target_q;
        req.skip_motion = skip_motion;
        {
            std::lock_guard<std::mutex> lock(mtx_);
            pending_     = req;
            target_q_    = target_q;
        }
    }

    Vector7 targetQ() {
        std::lock_guard<std::mutex> lock(mtx_);
        return target_q_;
    }

    void confirmResume() {
        resume_requested_.store(true);
    }

    bool hasPending() {
        std::lock_guard<std::mutex> lock(mtx_);
        return pending_.valid;
    }

    bool abort() {
        if (!hasPending() && mode_.load() == RecoveryMode::NONE) return false;
        abort_requested_.store(true);
        return true;
    }

    bool consumeAbort() {
        return abort_requested_.exchange(false);
    }

    bool isActive() const {
        return mode_.load() != RecoveryMode::NONE;
    }

    bool isWaitingAck() const {
        return mode_.load() == RecoveryMode::WAITING_ACK;
    }

    // set once the avatar has sent the reset_complete notification
    void markNotified() {
        notified_.store(true);
    }

    bool needsNotification() const {
        return notify_pending_.load();
    }

    void clearNotification() {
        notify_pending_.store(false);
    }

    RecoveryMode mode() const {
        return mode_.load();
    }

    void pushBack(const RecoveryRequest& req) {
        std::lock_guard<std::mutex> lock(mtx_);
        pending_ = req;
    }

    // --- arm-side interface (called from runStateHandler) ---

    // returns a valid request and consumes it if one is pending
    RecoveryRequest consumePending() {
        std::lock_guard<std::mutex> lock(mtx_);
        RecoveryRequest req = pending_;
        pending_.valid = false;
        return req;
    }

    void setMode(RecoveryMode m) {
        mode_.store(m);
        if (m == RecoveryMode::WAITING_ACK) {
            notify_pending_.store(true);
            notified_.store(false);
        }
        if (m == RecoveryMode::NONE) {
            notify_pending_.store(false);
            notified_.store(false);
            resume_requested_.store(false);
        }
    }

    bool shouldResume() const {
        return resume_requested_.load();
    }

    const std::string& name() const { return name_; }

private:
    std::string         name_;
    std::mutex          mtx_;
    RecoveryRequest     pending_;
    std::atomic<RecoveryMode> mode_{RecoveryMode::NONE};
    std::atomic<bool>   notify_pending_{false};
    std::atomic<bool>   notified_{false};
    std::atomic<bool>   resume_requested_{false};
    std::atomic<bool>   abort_requested_{false};
    Vector7 target_q_ = Vector7::Zero();
};