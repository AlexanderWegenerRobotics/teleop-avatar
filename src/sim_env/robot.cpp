#include <iostream>
#include <chrono>
#include <thread>
#include <algorithm>   // std::min/std::max, for the dt guard in control()

#include "sim_env/robot.hpp"
#include "sim_env/simulation.hpp"


using namespace franka;

Torques::Torques(const std::array<double, 7>& torques) noexcept : tau_J(torques) {}

Torques::Torques(std::initializer_list<double> torques) {
    std::copy(torques.begin(), torques.end(), tau_J.begin());
}

Robot::Robot() {
    r_            = Vector7::Zero();
    p_prev_       = Vector7::Zero();
    tau_filtered_ = Vector7::Zero();
    tau_prev_     = Vector7::Zero();
}

Robot::~Robot() {}

void Robot::set_simulation(Simulation& _sim, const YAML::Node& sim_dev, const YAML::Node& robot_dev) {
    sim          = &_sim;
    name_        = sim_dev["name"].as<std::string>();
    ee_frame_name_ = sim_dev["urdf_ee_name"] ? sim_dev["urdf_ee_name"].as<std::string>() : "panda_link8";

    std::string urdf_path = sim_dev["urdf_path"].as<std::string>();
    auto ori = robot_dev["base_pose"]["orientation"].as<std::vector<double>>();
    std::array<double, 4> base_quat = {ori[0], ori[1], ori[2], ori[3]};

    // defaults from menagerie fr3_torque.xml, must match the MJCF or friction shows up in tau_ext
    auto read7 = [&sim_dev](const char* key, const Vector7& fallback) {
        Vector7 v = fallback;
        if (sim_dev[key]) {
            auto raw = sim_dev[key].as<std::vector<double>>();
            for (size_t i = 0; i < 7 && i < raw.size(); ++i) v[static_cast<int>(i)] = raw[i];
        }
        return v;
    };
    const Vector7 joint_damping = read7("joint_damping",
        (Vector7() << 0.21, 0.21, 0.21, 0.21, 0.21, 0.21, 0.21).finished());
    const Vector7 joint_coulomb = read7("joint_friction",
        (Vector7() << 1.137, 1.137, 1.137, 1.137, 0.763, 0.44, 0.248).finished());
    const Vector7 rotor_inertia = read7("rotor_inertia",
        (Vector7() << 0.195, 0.195, 0.195, 0.195, 0.074, 0.074, 0.074).finished());

    model_ = std::make_unique<franka::Model>(urdf_path, base_quat, ee_frame_name_,
                                             joint_damping, joint_coulomb, rotor_inertia);

    if (robot_dev["safety"]) {
        const auto& s = robot_dev["safety"];
        if (s["collision_reflex_enabled"])
            collision_reflex_enabled_ = s["collision_reflex_enabled"].as<bool>();
        if (s["collision_persist_ticks"])
            collision_persist_ticks_ = std::max(1, s["collision_persist_ticks"].as<int>());
    }
    std::cout << "[SIM] " << name_ << ": collision reflex "
              << (collision_reflex_enabled_ ? "ENABLED" : "DISABLED") << ", persist "
              << collision_persist_ticks_ << " ticks, upper force ["
              << upper_force_thresholds_[0] << ", " << upper_force_thresholds_[1] << ", "
              << upper_force_thresholds_[2] << "] N." << std::endl;

    if (robot_dev["q_min"] && robot_dev["q_max"]) {
        auto qmin_vec = robot_dev["q_min"].as<std::vector<double>>();
        auto qmax_vec = robot_dev["q_max"].as<std::vector<double>>();
        for (size_t i = 0; i < 7 && i < qmin_vec.size(); ++i) q_min_[i] = qmin_vec[i];
        for (size_t i = 0; i < 7 && i < qmax_vec.size(); ++i) q_max_[i] = qmax_vec[i];
    }
}

Model& Robot::loadModel() {
    return *model_;
}

void Robot::setCollisionBehavior(
    const std::array<double, 7>& lower_torque_thresholds,
    const std::array<double, 7>& upper_torque_thresholds,
    const std::array<double, 6>& lower_force_thresholds,
    const std::array<double, 6>& upper_force_thresholds) {
    lower_torque_thresholds_ = lower_torque_thresholds;
    upper_torque_thresholds_ = upper_torque_thresholds;
    lower_force_thresholds_  = lower_force_thresholds;
    upper_force_thresholds_  = upper_force_thresholds;
    // runs before set_simulation(), so name_ is not set yet
    std::cout << "[SIM] setCollisionBehavior() - upper force ["
              << upper_force_thresholds_[0] << ", " << upper_force_thresholds_[1] << ", "
              << upper_force_thresholds_[2] << "] N, upper wrist torque ["
              << upper_torque_thresholds_[4] << ", " << upper_torque_thresholds_[5] << ", "
              << upper_torque_thresholds_[6] << "] Nm." << std::endl;
}

void Robot::setJointImpedance(const std::array<double, 7>& K_theta) {
    joint_impedance_ = K_theta;
}

void Robot::setCartesianImpedance(const std::array<double, 6>& K_x) {
    cartesian_impedance_ = K_x;
}

void Robot::automaticErrorRecovery() {
    r_            = Vector7::Zero();
    p_prev_       = Vector7::Zero();
    gmo_seed_pending_ = true;
    // observer is reseeded, old streaks would re-trip the reflex
    joint_reflex_streak_.fill(0);
    cart_reflex_streak_.fill(0);
    robot_state_.joint_contact.fill(0.0);
    robot_state_.cartesian_contact.fill(0.0);
    robot_state_.joint_collision.fill(0.0);
    robot_state_.cartesian_collision.fill(0.0);
    std::cout << "[SIM] " << name_ << ": automaticErrorRecovery()" << std::endl;
}

RobotState Robot::readOnce() {
    // outside control() read live sim state, without touching the GMO
    if (!bRunning.load() && sim != nullptr) {
        DeviceState ds = sim->getDeviceState(name_);
        if (ds.q.size() >= 7) {
            for (size_t i = 0; i < 7; ++i) {
                robot_state_.q[i]     = ds.q[i];
                robot_state_.dq[i]    = ds.dq[i];
                robot_state_.tau_J[i] = ds.tau_J[i];
            }
            robot_state_.sim_time = ds.time;
            robot_state_.O_T_EE   = model_->EEPose(robot_state_.q);
        }
    }
    return robot_state_;
}

void Robot::updateGMO(const std::array<double, 7>& q, const std::array<double, 7>& dq, const std::array<double, 7>& tau_cmd, double dt) {
    Vector7 tau_eig = Eigen::Map<const Vector7>(tau_cmd.data());
    auto [p, tau_model] = model_->computeGMOInputs(q, dq);
    // seed p_prev_ instead of differencing against zero, arm may still be moving
    if (gmo_seed_pending_) {
        gmo_seed_pending_ = false;
        p_prev_ = p;
    }
    r_ += K_GMO * (p - p_prev_ - (tau_eig - tau_model + r_) * dt);
    p_prev_ = p;
    std::array<double, 7> tau_ext;
    Eigen::Map<Vector7>(tau_ext.data()) = r_;
    robot_state_.tau_ext_hat_filtered   = tau_ext;
    robot_state_.O_F_ext_hat_K          = model_->cartesianWrench(q, tau_ext);

    // base -> stiffness frame, O_T_EE is column-major
    Eigen::Map<const Eigen::Matrix4d> T(robot_state_.O_T_EE.data());
    const Eigen::Matrix3d R = T.topLeftCorner<3, 3>();
    Eigen::Map<const Eigen::Vector3d> f_O(robot_state_.O_F_ext_hat_K.data());
    Eigen::Map<const Eigen::Vector3d> m_O(robot_state_.O_F_ext_hat_K.data() + 3);
    Eigen::Map<Eigen::Vector3d>(robot_state_.K_F_ext_hat_K.data())     = R.transpose() * f_O;
    Eigen::Map<Eigen::Vector3d>(robot_state_.K_F_ext_hat_K.data() + 3) = R.transpose() * m_O;
}

void Robot::checkCollisionReflex() {
    // FR3 reflex: lower thresholds set contact flags, upper ones stop the arm
    auto& rs = robot_state_;

    for (int i = 0; i < 7; ++i) {
        const double t = std::abs(rs.tau_ext_hat_filtered[i]);
        rs.joint_contact[i]   = (t > lower_torque_thresholds_[i]) ? 1.0 : 0.0;
        rs.joint_collision[i] = (t > upper_torque_thresholds_[i]) ? 1.0 : 0.0;
    }
    for (int i = 0; i < 6; ++i) {
        const double f = std::abs(rs.K_F_ext_hat_K[i]);
        rs.cartesian_contact[i]   = (f > lower_force_thresholds_[i]) ? 1.0 : 0.0;
        rs.cartesian_collision[i] = (f > upper_force_thresholds_[i]) ? 1.0 : 0.0;
    }

    if (!collision_reflex_enabled_) {
        joint_reflex_streak_.fill(0);
        cart_reflex_streak_.fill(0);
        return;
    }

    // must persist collision_persist_ticks_, not a single sample
    static const char* kAxis[6] = {"Fx", "Fy", "Fz", "Mx", "My", "Mz"};

    for (int i = 0; i < 6; ++i) {
        if (rs.cartesian_collision[i] != 0.0) {
            if (++cart_reflex_streak_[i] >= collision_persist_ticks_) {
                const double f = rs.K_F_ext_hat_K[i];
                std::cout << "[FRANKA ERROR] " << name_ << ": cartesian_reflex - "
                          << kAxis[i] << "=" << f << (i < 3 ? " N" : " Nm")
                          << " (limit=" << upper_force_thresholds_[i]
                          << (i < 3 ? " N)" : " Nm)") << "\n";
                cart_reflex_streak_.fill(0);
                joint_reflex_streak_.fill(0);
                throw ControlException("sim robot (" + name_ + "): cartesian_reflex on " +
                                       kAxis[i]);
            }
        } else {
            cart_reflex_streak_[i] = 0;
        }
    }

    for (int i = 0; i < 7; ++i) {
        if (rs.joint_collision[i] != 0.0) {
            if (++joint_reflex_streak_[i] >= collision_persist_ticks_) {
                std::cout << "[FRANKA ERROR] " << name_ << " joint " << i
                          << ": joint_reflex - tau_ext=" << rs.tau_ext_hat_filtered[i]
                          << " Nm (limit=" << upper_torque_thresholds_[i] << " Nm)\n";
                cart_reflex_streak_.fill(0);
                joint_reflex_streak_.fill(0);
                throw ControlException("sim robot (" + name_ + "): joint_reflex on joint " +
                                       std::to_string(i));
            }
        } else {
            joint_reflex_streak_[i] = 0;
        }
    }
}

void Robot::populateRobotState(const DeviceState& ds, double dt) {
    for (size_t i = 0; i < 7 && i < ds.q.size(); ++i) {
        robot_state_.q[i]     = ds.q[i];
        robot_state_.dq[i]    = ds.dq[i];
        robot_state_.tau_J[i] = ds.tau_J[i];
    }
    robot_state_.sim_time = ds.time;
    robot_state_.O_T_EE = model_->EEPose(robot_state_.q);
    updateGMO(robot_state_.q, robot_state_.dq, robot_state_.tau_J_d, dt);
}

void Robot::checkFrankaErrors(const Vector7& tau_cmd, const Vector7& dq, const Vector7& q) {
    // mimics the FR3 hard limits so the sim can fault like the real arm
    static const std::array<double, 7> kMaxTorqueRate    = {1000, 1000, 1000, 1000, 1000, 1000, 1000};
    static const std::array<double, 7> kMaxTorque        = {87, 87, 87, 87, 12, 12, 12};
    // FR3 datasheet rad/s, A6 uses libfranka's tighter 4.18
    static const std::array<double, 7> kMaxJointVelocity = {2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26};
    // rad, inside q_min_/q_max_
    constexpr double kJointLimitMargin = 0.01;

    constexpr double dt = 1.0 / 1000.0;

    // seed tau_prev_ on first tick after (re)entering control(), otherwise the rate check spikes
    if (tau_rate_seed_pending_) {
        tau_rate_seed_pending_ = false;
        tau_prev_ = tau_cmd;
    }

    // throws ControlException like libfranka
    for (int i = 0; i < 7; ++i) {
        if (std::abs(tau_cmd(i)) > kMaxTorque[i]) {
            std::cout << "[FRANKA ERROR] " << name_ << " joint " << i
                      << ": tau_J_range_violation - tau=" << tau_cmd(i)
                      << " Nm (limit=" << kMaxTorque[i] << " Nm)\n";
            tau_prev_ = tau_cmd;
            throw ControlException("sim robot (" + name_ + "): tau_J_range_violation on joint " +
                                    std::to_string(i));
        }

        double tau_rate = std::abs(tau_cmd(i) - tau_prev_(i)) / dt;
        if (tau_rate > kMaxTorqueRate[i]) {
            std::cout << "[FRANKA ERROR] " << name_ << " joint " << i
                      << ": torque_discontinuity - rate=" << tau_rate
                      << " Nm/s (limit=" << kMaxTorqueRate[i] << " Nm/s)\n";
            tau_prev_ = tau_cmd;
            throw ControlException("sim robot (" + name_ + "): torque_discontinuity on joint " +
                                    std::to_string(i));
        }

        if (std::abs(dq(i)) > kMaxJointVelocity[i]) {
            std::cout << "[FRANKA ERROR] " << name_ << " joint " << i
                      << ": joint_velocity_violation - dq=" << dq(i)
                      << " rad/s (limit=" << kMaxJointVelocity[i] << " rad/s)\n";
            throw ControlException("sim robot (" + name_ + "): joint_velocity_violation on joint " +
                                    std::to_string(i));
        }

        // report only the entry into violation, then latch so the joint can move back out
        const double q_lo = q_min_[i] + kJointLimitMargin;
        const double q_hi = q_max_[i] - kJointLimitMargin;
        if (q(i) < q_lo || q(i) > q_hi) {
            if (!joint_limit_tripped_[i]) {
                joint_limit_tripped_[i] = true;
                std::cout << "[FRANKA ERROR] " << name_ << " joint " << i
                          << ": joint_position_limits_violation - q=" << q(i)
                          << " rad (limits=[" << q_lo << ", " << q_hi << "] rad)\n";
                throw ControlException("sim robot (" + name_ + "): joint_position_limits_violation on joint " +
                                        std::to_string(i));
            }
        } else if (joint_limit_tripped_[i]) {
            joint_limit_tripped_[i] = false;
            std::cout << "[SIM] " << name_ << " joint " << i
                      << ": back inside position limits (q=" << q(i) << " rad)\n";
        }
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
    tau_rate_seed_pending_ = true;
    gmo_seed_pending_      = true;
    // don't reset tau_filtered_/tau_prev_ here, re-entered after every fault
    bRunning = true;

    try {
        while (bRunning) {
            if (!sim->isRunning()) {
                std::cout << "Simulation stopped" << std::endl;
                bRunning = false;
                break;
            }

            DeviceState device_state = sim->getDeviceState(name_);

            // sim time, not wall clock, since the GMO differentiates mjData state
            double dt_sim = (sim_time_prev_ < 0.0) ? 0.0
                                                   : device_state.time - sim_time_prev_;
            sim_time_prev_ = device_state.time;
            if (dt_sim < 0.0 || dt_sim > 0.1) dt_sim = 0.0;   // reset/seek guard

            populateRobotState(device_state, dt_sim);

            Torques tau_cmd = control_callback(robot_state_, dur);

            std::array<double, 7> gravity = model_->gravity(robot_state_.q);

            Vector7 tau_raw;
            for (int i = 0; i < 7; ++i)
                tau_raw[i] = tau_cmd.tau_J[i] + gravity[i];

            Vector7 dq_eig = Eigen::Map<const Vector7>(robot_state_.dq.data());
            Vector7 q_eig  = Eigen::Map<const Vector7>(robot_state_.q.data());
            Vector7 tau_cmd_eig = Eigen::Map<const Vector7>(tau_cmd.tau_J.data());

            // may throw ControlException, like libfranka
            checkFrankaErrors(tau_cmd_eig, dq_eig, q_eig);
            checkCollisionReflex();

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

                // sleep + spin, sleep_until alone rounds up to the OS tick and halves the rate.
                // kSpinMargin must exceed the timer granularity (~1 ms on Windows)
                constexpr auto kSpinMargin = std::chrono::microseconds(1200);
                const auto sleep_until_tp = next_control_time - kSpinMargin;
                if (std::chrono::high_resolution_clock::now() < sleep_until_tp)
                    std::this_thread::sleep_until(sleep_until_tp);
                while (std::chrono::high_resolution_clock::now() < next_control_time)
                    std::this_thread::yield();

                // resync if far behind instead of catching up
                const auto now_tp = std::chrono::high_resolution_clock::now();
                if (now_tp - next_control_time > std::chrono::milliseconds(50))
                    next_control_time = now_tp;
            }
        }
    } catch (...) {
        bRunning = false;
        sim->setDeviceActive(name_, false);
        throw;
    }
    sim->setDeviceActive(name_, false);
}