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
    r_            = Vector7::Zero();
    p_prev_       = Vector7::Zero();
    // One-time init only -- deliberately NOT repeated in control() (see there):
    // the retry loop in arm_control.cpp re-enters control() after every caught
    // fault, and tau_prev_ needs to keep holding the last actually-applied
    // torque across that re-entry, not reset to zero, or the discontinuity
    // check below manufactures a fault out of a normal torque on the very next
    // tick (see automaticErrorRecovery()'s comment for the full story).
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
    model_ = std::make_unique<franka::Model>(urdf_path, base_quat, ee_frame_name_);

    // Same q_min/q_max ArmControl itself reads from this device's config and
    // plans/brakes against -- checkFrankaErrors' hard joint-limit check uses
    // these (see below) instead of an independent hardcoded range, so the two
    // layers agree on where the limit actually is. Falls back to the FR3
    // factory range (this class's member-initializer default) if a config
    // omits them, though ArmControl requires these keys unconditionally so
    // that shouldn't happen for any device that's actually running.
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
    std::cout << "[SIM] " << name_ << ": setCollisionBehavior() applied (stored only - sim has no "
                 "separate collision reflex yet; hard torque/velocity/position limits are still "
                 "enforced via checkFrankaErrors)." << std::endl;
}

void Robot::setJointImpedance(const std::array<double, 7>& K_theta) {
    joint_impedance_ = K_theta;
}

void Robot::setCartesianImpedance(const std::array<double, 6>& K_x) {
    cartesian_impedance_ = K_x;
}

void Robot::automaticErrorRecovery() {
    // Real hardware clears its reflex/lockout state here. Sim has no persistent
    // lockout, so this just resets GMO history so the next control() call's
    // external-force estimate starts clean, matching the "fresh start" behavior
    // after a real recovery.
    //
    // Deliberately NOT resetting tau_prev_ (or tau_filtered_) to zero: it's the
    // reference checkFrankaErrors' torque_discontinuity check diffs the next
    // commanded torque against. Zeroing it means the very next control tick
    // compares a normal torque (e.g. a few Nm) against 0, producing a huge
    // apparent rate purely as an artifact of the reset -- not a real
    // discontinuity -- which manufactures a second (and third, and fourth...)
    // fault out of a single genuine one, cascading straight into FAULT. Leaving
    // it as whatever was last actually applied keeps the check meaningful.
    r_            = Vector7::Zero();
    p_prev_       = Vector7::Zero();
    std::cout << "[SIM] " << name_ << ": automaticErrorRecovery()" << std::endl;
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
    static const std::array<double, 7> kMaxJointVelocity = {2.150, 2.150, 2.150, 2.150, 2.580, 2.580, 2.580};
    // 0.01 rad (~0.6 deg) numerical safety margin inside the per-device
    // q_min_/q_max_ (set in set_simulation() from this robot's own config --
    // the SAME range ArmControl's IK plans/brakes against). Previously this
    // was an independent hardcoded array that didn't match a given device's
    // actual configured range, tripping this check before ArmControl's own
    // joint-limit braking ever needed to engage.
    constexpr double kJointLimitMargin = 0.01;

    constexpr double dt = 1.0 / 1000.0;

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

        const double q_lo = q_min_[i] + kJointLimitMargin;
        const double q_hi = q_max_[i] - kJointLimitMargin;
        if (q(i) < q_lo || q(i) > q_hi) {
            std::cout << "[FRANKA ERROR] " << name_ << " joint " << i
                      << ": joint_position_limits_violation - q=" << q(i)
                      << " rad (limits=[" << q_lo << ", " << q_hi << "] rad)\n";
            throw ControlException("sim robot (" + name_ + "): joint_position_limits_violation on joint " +
                                    std::to_string(i));
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
            populateRobotState(device_state, dt);

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