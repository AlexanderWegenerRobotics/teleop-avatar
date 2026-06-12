#include "head_control.hpp"
#include "data_logger.hpp"
#include "rt_thread.hpp"
#include <iostream>

using namespace franka_joint_driver;

HeadControl::HeadControl(const YAML::Node& device_config, const std::string& session_id)
    : bRunning(false)
    , module(std::make_unique<franka_joint_driver::Driver>())
    , state_(SysState::OFFLINE)
    , alpha_dq(0.02)
    , interpolator_(InterpolatorConfig{
        .control_freq   = 1000,
        .comm_freq      = device_config["transmission"]["frequency"].as<int>(),
        .n_dof          = 2,
        .max_linear_vel = 0.1,
        .max_angular_vel = 8.0
    })
{
    name_ = device_config["name"].as<std::string>();

    q0_ = yamlToVector<2>(device_config["q0"]);
    tau_max_ = yamlToVector<2>(device_config["max_torque"]);
    tau_rate_max_ = yamlToVector<2>(device_config["max_torque_rate"]) / 1000.0;
    kp_ = yamlToVector<2>(device_config["control"]["kp_joint"]);
    kd_ = yamlToVector<2>(device_config["control"]["kd_joint"]);

    if (device_config["transmission"]) {
        UdpStreamConfig stream_cfg;
        stream_cfg.transport.remote_ip   = device_config["transmission"]["remote_ip"].as<std::string>();
        stream_cfg.transport.remote_port = device_config["transmission"]["send_port"].as<int>();
        stream_cfg.transport.bind_port   = device_config["transmission"]["receive_port"].as<int>();
        stream_cfg.send_rate_hz          = device_config["transmission"]["frequency"].as<int>();
        transmission_ = std::make_unique<HeadStream>(stream_cfg);
    }
    logger_ = std::make_unique<DataLogger<HeadLogEntry>>("../log/" + name_ + "_log.csv", headLogHeader, headLogRow, session_id);
}

HeadControl::~HeadControl(){
    stop();
}

void HeadControl::start(){
    bRunning = true;
    state_ = SysState::IDLE;
    Vector2 q_init = Vector2::Zero();
    interpolator_.planJoint(q_init, q_init, ProfileType::TRAPEZOIDAL);
    control_thread = std::thread(&HeadControl::runControlHandler, this);
    set_realtime(control_thread, 4);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    state_thread = std::thread(&HeadControl::runStateHandler, this);
    set_realtime(state_thread, 5);
    if (transmission_) transmission_->start();
    logger_->start();
    logger_->enable(true);
    startTime_ = std::chrono::high_resolution_clock::now();
}

void HeadControl::stop(){
    if (logger_) logger_->stop(); 
    bRunning = false;
    if (control_thread.joinable()) control_thread.join();
    if (state_thread.joinable()) state_thread.join();
    if (transmission_) transmission_->stop();
}

void HeadControl::runStateHandler(){
    constexpr std::chrono::microseconds control_period(static_cast<int>(1e6 / 200));
    auto next_control_time = std::chrono::high_resolution_clock::now();

    SysState prev_state = state_;
    Vector2 q_current = Vector2::Zero();
    bool has_cmd = false;
    HeadCommandMsg cmd;

    while(bRunning){

        if (transmission_ && transmission_->hasNew()) {
            cmd = transmission_->getRecvData();
            has_cmd = true;
        }

        updateStateMachine(cmd_state_);

        if (state_ == SysState::HOMING && prev_state != SysState::HOMING) {
            {
                std::lock_guard<std::mutex> lock(state_mtx);
                q_current = current_state.q;
            }
            interpolator_.planJoint(q_current, q0_, ProfileType::MINJERK);
        }
        else if (state_ == SysState::ENGAGED) {
            if (has_cmd) {
                Vector2 q_target;
                q_target(0) = static_cast<double>(cmd.pan);
                q_target(1) = static_cast<double>(cmd.tilt);
                q_target += q0_;
                interpolator_.planJoint(interpolator_.getCurrentJoint(), q_target, ProfileType::LINEAR);
                has_cmd = false;
            }
        }

        if (transmission_) {
            Vector2 q, dq;
            {
                std::lock_guard<std::mutex> lock(state_mtx);
                q  = current_state.q;
                dq = current_state.dq;
            }

            HeadStateMsg state_msg{};
            state_msg.pan   = static_cast<float>(q(0));
            state_msg.tilt  = static_cast<float>(q(1));
            transmission_->setSendData(state_msg);
        }

        prev_state = state_;
        next_control_time += control_period;
        std::this_thread::sleep_until(next_control_time);
    }
}

void HeadControl::updateStateMachine(SysState cmd_state){
    SysState prev = state_;
    if(cmd_state == SysState::STOP){
        state_ = SysState::STOP;
    }
    switch (state_) {
        case SysState::IDLE:
            if(cmd_state == SysState::HOMING){
                state_ = SysState::HOMING;
                std::cout << "[INFO]: " << name_ << " is homing." << std::endl;
            }
            break;
        case SysState::HOMING:
            if(isHome()){
                state_ = SysState::AWAITING;
                std::cout << "[INFO]: " << name_ << " is awaiting." << std::endl;
            }
            break;
        case SysState::AWAITING:
            if(cmd_state == SysState::IDLE){
                state_ = SysState::IDLE;
            }
            else if(cmd_state == SysState::ENGAGED){
                state_ = SysState::ENGAGED;
                std::cout << "[INFO]: " << name_ << " engaged." << std::endl;
            }
            break;
        case SysState::ENGAGED:
            if(cmd_state == SysState::IDLE){
                state_ = SysState::IDLE;
            }
            else if(cmd_state == SysState::PAUSED){
                state_ = SysState::PAUSED;
            }
            break;
        case SysState::PAUSED:
            if(cmd_state == SysState::IDLE){
                state_ = SysState::IDLE;
            }
            else if(cmd_state == SysState::ENGAGED){
                state_ = SysState::ENGAGED;
            }
            break;

        default:
            break;
    }
    if (state_ != prev && transmission_) {
        transmission_->setState(state_);
    }
}

void HeadControl::runControlHandler() {
    Vector2 tau_prev_ = Vector2::Zero();
    bool bInitDone = false;

    franka_joint_driver::Driver::CallbackFunctionTorque torque_control_callback =
        [&](const std::vector<Driver::State>& driver_state,
            std::vector<Driver::CommandTorque>& command) -> void {

            interpolator_.step();

            Vector2 q, tau_j;
            for (size_t i = 0; i < 2; ++i) {
                q(i)     = driver_state[i].theta;
                tau_j(i) = driver_state[i].tau_j;
            }

            if (!bInitDone) {
                interpolator_.planJoint(q, q, ProfileType::TRAPEZOIDAL);
                bInitDone = true;
            }

            auto now = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration<double>(now - current_state.last_dq_update).count();

            Vector2 dq = current_state.dq;
            if (dt > 1e-4) {
                Vector2 dq_raw = (q - current_state.q) / dt;    
                std::lock_guard<std::mutex> lock(state_mtx);
                dq = dq_raw * alpha_dq + (1.0 - alpha_dq) * current_state.dq;
                current_state.last_dq_update = now;
                current_state.q = q;
                current_state.dq = dq;
            }


            Vector2 q_cmd = interpolator_.getCurrentJoint();
            Vector2 e = q_cmd - q;
            Vector2 tau_cmd = kp_.cwiseProduct(e) - kd_.cwiseProduct(dq);

            // joint limit proximity — scale down torque that drives into limits
            for (int i = 0; i < 2; ++i) {
                if ((q(i) >= q_max[i] && tau_cmd(i) > 0.0) ||
                    (q(i) <= q_min[i] && tau_cmd(i) < 0.0))
                    tau_cmd(i) = 0.0;
            }

            // rate limit & torque limit
            tau_cmd = tau_prev_ + (tau_cmd - tau_prev_).cwiseMax(-tau_rate_max_).cwiseMin(tau_rate_max_);
            tau_cmd = tau_cmd.cwiseMax(-tau_max_).cwiseMin(tau_max_);
            tau_prev_ = tau_cmd;

            if (logger_) {
                double t = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - startTime_).count();

                HeadLogEntry entry{};
                entry.time          = t;
                entry.wall_clock_ns = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count());
                entry.state = state_;

                Eigen::Map<Eigen::Vector2d>(entry.q.data()) = q;
                Eigen::Map<Eigen::Vector2d>(entry.q_cmd.data()) = q_cmd;
                Eigen::Map<Eigen::Vector2d>(entry.dq.data()) = dq;
                Eigen::Map<Eigen::Vector2d>(entry.tau_J.data()) = tau_cmd;

                logger_->write(entry);
            }

            for (size_t i = 0; i < 2; ++i) {
                command[i].tau_j_d  = tau_cmd[i];
                command[i].finished = false;
            }
        };

    if (module->control(torque_control_callback) == Driver::Error::kCommunicationError) {
        std::cout << "[WARNING]: " << name_ << " has a communication error." << std::endl;
        state_    = SysState::FAULT;
        bRunning  = false;
    }
}

bool HeadControl::isHome() {
    Vector2 q, dq;
    {
        std::lock_guard<std::mutex> lock(state_mtx);
        q  = current_state.q;
        dq = current_state.dq;
    }
    return (q0_ - q).cwiseAbs().maxCoeff() < 0.05 
        && dq.cwiseAbs().maxCoeff() < 0.03;
}

void HeadControl::restartLogger(const std::string& path) {
    logger_->restart(path);
}

void HeadControl::writeEpisodeConfig(int seed, int mode, const std::string& color_bin_mapping) {
    logger_->writeEpisodeConfig(seed, mode, color_bin_mapping);
}
