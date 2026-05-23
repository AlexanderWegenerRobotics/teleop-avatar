#include <iostream>
#include <chrono>
#include <thread>

#include "sim_env/robot.hpp"
#include "sim_env/simulation.hpp"


using namespace franka;

Torques::Torques(const std::array<double, 7>& torques) noexcept : tau_J(torques) {}

Torques::Torques(std::initializer_list<double> torques) {
    std::copy(torques.begin(), torques.end(), tau_J.begin());
}

Robot::Robot() {
    r_      = Vector7::Zero();
    p_prev_ = Vector7::Zero();
}

Robot::~Robot() {}

void Robot::set_simulation(Simulation& _sim, const YAML::Node& sim_dev, const YAML::Node& robot_dev) {
    sim          = &_sim;
    name_        = sim_dev["name"].as<std::string>();
    ee_frame_name_ = sim_dev["urdf_ee_name"] ? sim_dev["urdf_ee_name"].as<std::string>() : "panda_link8";

    std::string urdf_path = sim_dev["urdf_path"].as<std::string>();
    auto ori = robot_dev["base_pose"]["orientation"].as<std::vector<double>>();
    std::array<double, 4> base_quat = {ori[0], ori[1], ori[2], ori[3]};
    model_ = std::make_unique<franka::Model>(urdf_path, base_quat, ee_frame_name_);
}

Model& Robot::loadModel() {
    return *model_;
}

RobotState Robot::readOnce() {
    return robot_state_;
}

void Robot::updateGMO(const std::array<double, 7>& q, const std::array<double, 7>& dq, const std::array<double, 7>& tau_cmd, double dt) {
    Vector7 tau_eig = Eigen::Map<const Vector7>(tau_cmd.data());
    auto [p, tau_model] = model_->computeGMOInputs(q, dq);
    r_ += K_GMO * (p - p_prev_ - (tau_eig - tau_model + r_) * dt);
    p_prev_ = p;
    std::array<double, 7> tau_ext;
    Eigen::Map<Vector7>(tau_ext.data()) = r_;
    robot_state_.tau_ext_hat_filtered   = tau_ext;
    robot_state_.O_F_ext_hat_K          = model_->cartesianWrench(q, tau_ext);
}

void Robot::populateRobotState(const DeviceState& ds, double dt) {
    for (size_t i = 0; i < 7 && i < ds.q.size(); ++i) {
        robot_state_.q[i]     = ds.q[i];
        robot_state_.dq[i]    = ds.dq[i];
        robot_state_.tau_J[i] = ds.tau_J[i];
    }
    robot_state_.O_T_EE = model_->EEPose(robot_state_.q);
    updateGMO(robot_state_.q, robot_state_.dq, robot_state_.tau_J_d, dt);
}

void Robot::checkFrankaErrors(const Vector7& tau_cmd, const Vector7& dq, const Vector7& q) {
    static const std::array<double, 7> kMaxTorqueRate    = {1000, 1000, 1000, 1000, 1000, 1000, 1000};
    static const std::array<double, 7> kMaxTorque        = {87, 87, 87, 87, 12, 12, 12};
    static const std::array<double, 7> kMaxJointVelocity = {2.150, 2.150, 2.150, 2.150, 2.580, 2.580, 2.580};
    static const std::array<std::pair<double,double>, 7> kJointLimits = {{
        {-2.7437 + 0.01,  2.7437 - 0.01},
        {-1.7837 + 0.01,  1.7837 - 0.01},
        {-2.9007 + 0.01,  2.9007 - 0.01},
        {-3.0421 + 0.01, -0.1518 - 0.01},
        {-2.8065 + 0.01,  2.8065 - 0.01},
        { 0.5445 + 0.01,  4.5169 - 0.01},
        {-3.0159 + 0.01,  3.0159 - 0.01}
    }};

    constexpr double dt = 1.0 / 1000.0;

    for (int i = 0; i < 7; ++i) {
        if (std::abs(tau_cmd(i)) > kMaxTorque[i])
            std::cout << "[FRANKA ERROR] " << name_ << " joint " << i
                      << ": tau_J_range_violation - tau=" << tau_cmd(i)
                      << " Nm (limit=" << kMaxTorque[i] << " Nm)\n";

        double tau_rate = std::abs(tau_cmd(i) - tau_prev_(i)) / dt;
        if (tau_rate > kMaxTorqueRate[i])
            std::cout << "[FRANKA ERROR] " << name_ << " joint " << i
                      << ": torque_discontinuity - rate=" << tau_rate
                      << " Nm/s (limit=" << kMaxTorqueRate[i] << " Nm/s)\n";

        if (std::abs(dq(i)) > kMaxJointVelocity[i])
            std::cout << "[FRANKA ERROR] " << name_ << " joint " << i
                      << ": joint_velocity_violation - dq=" << dq(i)
                      << " rad/s (limit=" << kMaxJointVelocity[i] << " rad/s)\n";

        if (q(i) < kJointLimits[i].first || q(i) > kJointLimits[i].second)
            std::cout << "[FRANKA ERROR] " << name_ << " joint " << i
                      << ": joint_position_limits_violation - q=" << q(i)
                      << " rad (limits=[" << kJointLimits[i].first
                      << ", " << kJointLimits[i].second << "] rad)\n";
    }

    tau_prev_ = tau_cmd;
}

void Robot::control(std::function<Torques(const RobotState&, Duration)> control_callback) {
    constexpr double dt = 1.0 / 1000.0;
    constexpr std::chrono::microseconds control_period(static_cast<int>(1e6 / 1000.0));

    constexpr double filter_cutoff_hz = 500.0;
    constexpr double omega = 2.0 * M_PI * filter_cutoff_hz;
    constexpr double alpha = (omega * dt) / (1.0 + omega * dt);

    auto next_control_time = std::chrono::high_resolution_clock::now();
    Duration dur;

    if (sim == nullptr) {
        std::cout << "You need to set the simulator first" << std::endl;
        return;
    }

    sim->setDeviceActive(name_, true);
    tau_filtered_ = Vector7::Zero();
    tau_prev_     = Vector7::Zero();
    bRunning = true;

    while (bRunning) {
        if (!sim->isRunning()) {
            std::cout << "Simulation stopped" << std::endl;
            bRunning = false;
            break;
        }

        DeviceState device_state = sim->getDeviceState(name_);
        populateRobotState(device_state, dt);

        Torques tau_cmd = control_callback(robot_state_, dur);

        std::array<double, 7> gravity = model_->gravity(robot_state_.q);

        Vector7 tau_raw;
        for (int i = 0; i < 7; ++i)
            tau_raw[i] = tau_cmd.tau_J[i] + gravity[i];

        Vector7 dq_eig = Eigen::Map<const Vector7>(robot_state_.dq.data());
        Vector7 q_eig  = Eigen::Map<const Vector7>(robot_state_.q.data());
        Vector7 tau_cmd_eig = Eigen::Map<const Vector7>(tau_cmd.tau_J.data());

        checkFrankaErrors(tau_cmd_eig, dq_eig, q_eig);

        tau_filtered_ = alpha * tau_raw + (1.0 - alpha) * tau_filtered_;

        if (tau_cmd.motion_finished) {
            std::cout << "Stopped robot arm control loop" << std::endl;
            bRunning = false;
        }

        if (bRunning) {
            std::array<double, 7> tau_out;
            Eigen::Map<Vector7>(tau_out.data()) = tau_filtered_;
            robot_state_.tau_J_d = tau_out;
            sim->setCtrl(name_, std::vector<double>(tau_out.begin(), tau_out.end()));
            next_control_time += control_period;
            std::this_thread::sleep_until(next_control_time);
        }
    }
    sim->setDeviceActive(name_, false);
}