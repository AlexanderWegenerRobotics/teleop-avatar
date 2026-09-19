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

    // Defaults are the mujoco_menagerie FR3 values in fr3_torque.xml. Override
    // per device in sim_config when the MJCF (or an identified arm) differs --
    // these must track the plant or the GMO reports friction as tau_ext.
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
    // Called from ArmControl's constructor, i.e. BEFORE set_simulation(), so
    // name_ and the reflex settings are not populated yet -- those are logged
    // there instead.
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
    // The observer is being re-seeded, so any streak counted against the old
    // residual is meaningless. Not clearing these re-trips the reflex on the
    // first tick after the resume, before the arm has moved.
    joint_reflex_streak_.fill(0);
    cart_reflex_streak_.fill(0);
    robot_state_.joint_contact.fill(0.0);
    robot_state_.cartesian_contact.fill(0.0);
    robot_state_.joint_collision.fill(0.0);
    robot_state_.cartesian_collision.fill(0.0);
    std::cout << "[SIM] " << name_ << ": automaticErrorRecovery()" << std::endl;
}

RobotState Robot::readOnce() {
    // Outside control() nothing refreshes robot_state_, so a caller planning a
    // recovery from it would see the pose at the moment of the fault, not where
    // the arm has coasted to since. Pull the live sim state instead, without
    // touching the momentum observer (its dt bookkeeping belongs to control()).
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
    // Seed rather than difference against a zeroed p_prev_. Robot() and
    // automaticErrorRecovery() both zero it, so the first update after either
    // one evaluated r_ += K_GMO * p -- 50x the momentum, reported as external
    // torque. At rest p is zero and it did not show; after a fault the arm is
    // still coasting (waitForRest only gets it under 0.1 rad/s) and it is not.
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

    // Rotate the base-frame wrench into the stiffness frame. O_T_EE is
    // column-major, so the leading 3x3 block of the Map is the rotation.
    Eigen::Map<const Eigen::Matrix4d> T(robot_state_.O_T_EE.data());
    const Eigen::Matrix3d R = T.topLeftCorner<3, 3>();
    Eigen::Map<const Eigen::Vector3d> f_O(robot_state_.O_F_ext_hat_K.data());
    Eigen::Map<const Eigen::Vector3d> m_O(robot_state_.O_F_ext_hat_K.data() + 3);
    Eigen::Map<Eigen::Vector3d>(robot_state_.K_F_ext_hat_K.data())     = R.transpose() * f_O;
    Eigen::Map<Eigen::Vector3d>(robot_state_.K_F_ext_hat_K.data() + 3) = R.transpose() * m_O;
}

void Robot::checkCollisionReflex() {
    // The real FR3 reflex, reproduced: the LOWER thresholds only raise the
    // contact flags, the UPPER ones stop the arm. Without this the twin will
    // happily push tens of newtons into a fixture that would have reflex-stopped
    // the plant, so anything learned or recorded against it is a habit that
    // faults on hardware.
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

    // Persistence, not a single sample. See collision_persist_ticks_ in the
    // header for why: this observer's free-motion noise reaches ~5 N p95.
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
    // RE-ENABLED 2026-08-20. Was disabled on 2026-08-08 because the retry/FAULT
    // path it feeds locked up desk-config runs on joint-velocity violations
    // that motion_gen_ did not back off from on retry.
    //
    // The retry path has since been fixed (b7f63c8: rearmFromMeasuredState()
    // re-plans from the measured pose and zeroes tau_prev_, so a restart no
    // longer steps straight back to the pre-fault torque), which removes the
    // cascade that made this unusable.
    //
    // Leaving it off has a cost that only became clear after the 2026-08-09
    // desk test: with these checks bypassed the twin CANNOT fault, so it
    // silently continues through conditions that stop the real arm. During
    // that run the avatar faulted at t=404.7 s while the twin ran on to
    // t=422.7 s, and no statement about the twin's safety behaviour was
    // supportable. A digital twin that cannot fail the way the plant fails is
    // not a safety model.
    //
    // If this needs disabling again, gate it behind a config flag that is
    // logged, so the analysis can see it was off.

    static const std::array<double, 7> kMaxTorqueRate    = {1000, 1000, 1000, 1000, 1000, 1000, 1000};
    static const std::array<double, 7> kMaxTorque        = {87, 87, 87, 87, 12, 12, 12};
    // FR3, matching models/mujoco/robots/franka_fr3. The datasheet gives
    // A1-A4 150 deg/s (2.62 rad/s) and A5-A7 301 deg/s (5.26 rad/s); libfranka
    // publishes 4.18 for A6, so take the tighter value there.
    //
    // These were previously Panda's limits (2.175 / 2.610) rounded down, on an
    // FR3 model. That faulted the wrist at half its real ceiling: every
    // joint_velocity_violation in session 002 was joint 5 at 2.58-2.61 rad/s,
    // against a true limit of 4.18. See claude/arm-fault-root-cause-001.md.
    static const std::array<double, 7> kMaxJointVelocity = {2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26};
    // 0.01 rad (~0.6 deg) numerical safety margin inside the per-device
    // q_min_/q_max_ (set in set_simulation() from this robot's own config --
    // the SAME range ArmControl's IK plans/brakes against). Previously this
    // was an independent hardcoded array that didn't match a given device's
    // actual configured range, tripping this check before ArmControl's own
    // joint-limit braking ever needed to engage.
    constexpr double kJointLimitMargin = 0.01;

    constexpr double dt = 1.0 / 1000.0;

    // First tick after entering or re-entering control(): seed tau_prev_ from
    // the current command instead of differencing against a torque that is
    // seconds old. The loop is parked in enterFaultAndWaitForReset() while
    // faulted, so no ticks run, but the rate below still divides by a nominal
    // 1 ms -- which turned an ordinary 1.8 Nm resume command into a reported
    // 1800 Nm/s and re-faulted the arm on the tick after every reset.
    if (tau_rate_seed_pending_) {
        tau_rate_seed_pending_ = false;
        tau_prev_ = tau_cmd;
    }

    // Mirrors libfranka's behavior: a reflex-worthy condition throws
    // franka::ControlException out of control(), rather than merely logging.
    // This lets us exercise the same catch/retry/FAULT path in sim that the
    // real robot forces us to handle on hardware.
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

        // Report the ENTRY into violation, once, then latch until the joint is
        // back inside. A joint that is already past its limit has to be allowed
        // to travel out again, or the fault is permanent: this runs on the first
        // tick after control resumes, before the recovery trajectory has moved
        // anything, and the arm is at rest there. Anything that keys off the
        // sign of dq re-faults immediately, because dq is zero.
        //
        // The escape itself already exists -- an operator reset calls
        // requestRecovery(OPERATOR_RESET, getQ0()) and updateRecovery() plans a
        // MINJERK move to q0, which is inside the range on every joint. It just
        // needs permission to execute.
        //
        // This is also what the hardware does: the reflex is on being driven
        // into the stop, and moving the joint back out is what clears it.
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
    auto tick_prev = next_control_time;
    // Rolling mean of the achieved period, reported every kRateReportTicks. The
    // nominal 1 ms is a request, not a guarantee -- this loop measured 2.001 ms
    // on Windows before the hybrid sleep+spin below (see the note there).
    constexpr int kRateReportTicks = 5000;
    double dt_sum  = 0.0;   // wall seconds accumulated
    double sim_sum = 0.0;   // simulated seconds accumulated
    int    dt_n    = 0;

    Duration dur;

    if (sim == nullptr) {
        std::cout << "You need to set the simulator first" << std::endl;
        return;
    }

    sim->setDeviceActive(name_, true);
    tau_rate_seed_pending_ = true;
    gmo_seed_pending_      = true;
    // tau_filtered_/tau_prev_ deliberately NOT reset here -- see Robot::Robot()
    // and automaticErrorRecovery() comments. This function is re-entered by
    // arm_control.cpp's retry loop after every caught fault; zeroing either on
    // each entry manufactures a spurious torque_discontinuity out of a normal
    // torque on the very next tick.
    bRunning = true;

    try {
        while (bRunning) {
            if (!sim->isRunning()) {
                std::cout << "Simulation stopped" << std::endl;
                bRunning = false;
                break;
            }

            DeviceState device_state = sim->getDeviceState(name_);

            // SIMULATED elapsed time, not wall clock. The momentum observer
            // differentiates mjData state, so its dt is the plant's integration
            // time; using a wall clock biases it by (M*qdd)*(dt_sim/dt_wall - 1),
            // which is zero at rest and grows with acceleration. The sim thread
            // runs slower than real time (measured 0.50x), so the two differ by
            // a factor of two here.
            double dt_sim = (sim_time_prev_ < 0.0) ? 0.0
                                                   : device_state.time - sim_time_prev_;
            sim_time_prev_ = device_state.time;
            // dt_sim == 0 means no new sim step since the last tick; p is then
            // unchanged too, so the observer correctly leaves r_ alone.
            if (dt_sim < 0.0 || dt_sim > 0.1) dt_sim = 0.0;   // reset/seek guard

            const auto tick_now = std::chrono::high_resolution_clock::now();
            dt_sum += std::chrono::duration<double>(tick_now - tick_prev).count();
            sim_sum += dt_sim;
            tick_prev = tick_now;
            if (++dt_n >= kRateReportTicks) {
                std::cout << "[SIM] " << name_ << ": control loop " << (dt_sum / dt_n) * 1e3
                          << " ms wall / " << (sim_sum / dt_n) * 1e3 << " ms sim per tick ("
                          << (dt_sum > 0 ? sim_sum / dt_sum : 0.0) << "x real time)" << std::endl;
                dt_sum = 0.0;
                sim_sum = 0.0;
                dt_n   = 0;
            }

            populateRobotState(device_state, dt_sim);

            Torques tau_cmd = control_callback(robot_state_, dur);

            std::array<double, 7> gravity = model_->gravity(robot_state_.q);

            Vector7 tau_raw;
            for (int i = 0; i < 7; ++i)
                tau_raw[i] = tau_cmd.tau_J[i] + gravity[i];

            Vector7 dq_eig = Eigen::Map<const Vector7>(robot_state_.dq.data());
            Vector7 q_eig  = Eigen::Map<const Vector7>(robot_state_.q.data());
            Vector7 tau_cmd_eig = Eigen::Map<const Vector7>(tau_cmd.tau_J.data());

            // May throw franka::ControlException, same as real hardware hitting a
            // reflex stop - propagates out of control() below, exactly like libfranka.
            checkFrankaErrors(tau_cmd_eig, dq_eig, q_eig);
            // Contact and collision against the setCollisionBehavior thresholds,
            // on the momentum observer's estimate. Separate from the hard limits
            // above: those are about what we COMMAND, this is about what the
            // world is doing back to the arm.
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

                // Hybrid sleep + spin, instead of sleep_until(deadline).
                //
                // This loop asks for 1 kHz and delivered 479 Hz on 2026-08-09:
                // a median dt of 2.001 ms, i.e. exactly twice the requested
                // period. It is not compute-bound -- dt was identical in IDLE
                // (2.001 ms) and ENGAGED (2.001 ms), so the loop body is not
                // the constraint. It is the OS timer: sleep_until wakes on the
                // next scheduler tick, so a sub-millisecond deadline is
                // rounded up to the following one and every period doubles.
                //
                // The real robot does not have this problem because libfranka's
                // control() is clocked by the FCI's own 1 ms tick. The
                // consequence was a twin running at half the avatar's rate with
                // three times the jitter, which is not a fair basis for
                // comparing the two.
                //
                // Sleep until slightly before the deadline, then busy-wait the
                // remainder. kSpinMargin must exceed the platform's timer
                // granularity (~1 ms on Windows without timeBeginPeriod).
                constexpr auto kSpinMargin = std::chrono::microseconds(1200);
                const auto sleep_until_tp = next_control_time - kSpinMargin;
                if (std::chrono::high_resolution_clock::now() < sleep_until_tp)
                    std::this_thread::sleep_until(sleep_until_tp);
                while (std::chrono::high_resolution_clock::now() < next_control_time)
                    std::this_thread::yield();

                // If we have fallen far behind (debugger, host contention),
                // resynchronise rather than sprinting to catch up -- a burst of
                // zero-dt ticks corrupts every rate statistic downstream.
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