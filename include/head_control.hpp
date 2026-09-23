#pragma once
#include <atomic>
#include <string>
#include <yaml-cpp/yaml.h>
#include <mutex>
#include <thread>

#include "sim_env/tum_head_driver.hpp"
#include "interpolator.hpp"
#include "network/udp_stream.hpp"
#include "data_logger.hpp"
#include "common.hpp"

struct HeadState {
    Vector2 q, dq;
    std::chrono::high_resolution_clock::time_point last_dq_update;

    HeadState() {
        q.setZero();
        dq.setZero();
        last_dq_update = std::chrono::high_resolution_clock::now();
    }
};

class HeadControl{
public:
    HeadControl(const YAML::Node& device_config, const std::string& session_id);
    ~HeadControl();

    void start();
    void stop();
    bool isRunning() const {return bRunning;}
    std::string getDeviceName() const {return name_;}
    void requestState(SysState state){cmd_state_ = state;}
    SysState getState() const {return state_;}
    void markEpisodeStart() { if (logger_) logger_->markEpisodeStart(); }
    void markEpisodeEnd(const std::string& reason) { if (logger_) logger_->markEpisodeEnd(reason); }
    void restartLogger(const std::string& path);
    void writeEpisodeConfig(int seed, int mode, const std::string& color_bin_mapping);

public:
    std::unique_ptr<franka_joint_driver::Driver> module;
    Interpolator interpolator_;
    using HeadStream = UdpStream<HeadCommandMsg, HeadStateMsg>;
    std::unique_ptr<HeadStream> transmission_;
    // Second, optional channel on its own port -- same HeadCommandMsg struct,
    // unchanged, but pan/tilt are read as ABSOLUTE joint targets instead of
    // transmission_'s home-relative semantics (which add q0_ on arrival).
    //
    // The same split the arms already have (transmission_absolute in
    // arm_control.hpp), for the same reason: the VR interface naturally speaks
    // deltas from an origin it captured, while an autonomous policy predicts
    // absolute joint targets, because absolute is what the logs it trained on
    // record (head.csv's q/q_cmd are post-q0_). Converting between the two in
    // the sender meant duplicating q0 into another repo's config, where
    // nothing could notice it drifting -- and on 2026-09-23 it was simply
    // missing, so q0 was applied twice and the neck sat 23 degrees too low.
    //
    // Config-gated (transmission_absolute in the head's device_config), so a
    // config without it behaves exactly as before.
    std::unique_ptr<HeadStream> transmission_absolute_;
    std::unique_ptr<DataLogger<HeadLogEntry>> logger_;
    std::chrono::high_resolution_clock::time_point startTime_;

private:
    std::atomic<bool> bRunning;
    std::thread control_thread;
    std::thread state_thread;
    std::mutex state_mtx;
    std::string name_;
    std::atomic<SysState> state_{SysState::OFFLINE};
    std::atomic<SysState> cmd_state_{SysState::OFFLINE};
    Vector2 q0_;
    HeadState current_state;
    double alpha_dq;
    Vector2 kp_, kd_;
    double q_min[2]     = {-2.78, -1.57};
    double q_max[2]     = { 2.78,  1.57};
    Vector2 tau_max_;
    Vector2 tau_rate_max_;

private:
    void runControlHandler();
    void runStateHandler();
    void updateStateMachine(SysState cmd_state);
    bool isHome();
};