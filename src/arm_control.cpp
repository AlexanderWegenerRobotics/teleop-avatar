#include "arm_control.hpp"
#include "common.hpp"
#include "MotionGenerator.hpp"
#include "self_collision_protection.hpp"
#include "sim_env/gripper.hpp"
#include "sim_env/model.hpp"
#include "rt_thread.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <thread>

namespace {
constexpr double kGripperMaxWidth = 0.08;

// Bounds on the measured command interval used by validateTargetPose. Low guards
// against a zero/negative interval; high caps the jump allowed after a gap in the
// command stream (0.05 s x max_command_velocity = 50 mm at 1.0 m/s).
constexpr double kMinCmdDt = 0.0005;
constexpr double kMaxCmdDt = 0.05;

// Well below the float32 resolution of a command on the wire, so this only ever
// matches a genuinely repeated target.
constexpr double kTargetEpsM   = 1e-6;
constexpr double kTargetEpsRad = 1e-6;

bool targetsEqual(const Eigen::Isometry3d& a, const Eigen::Isometry3d& b) {
    if ((a.translation() - b.translation()).norm() > kTargetEpsM) return false;
    return Eigen::Quaterniond(a.rotation()).angularDistance(
               Eigen::Quaterniond(b.rotation())) <= kTargetEpsRad;
}

template<size_t N>
std::array<double, N> toArray(const std::vector<double>& v) {
    std::array<double, N> a{};
    for (size_t i = 0; i < N && i < v.size(); ++i) a[i] = v[i];
    return a;
}
}

ArmControl::ArmControl(const YAML::Node& device_config, const std::string& session_id)
#ifdef WITH_FRANKA
    : robot(std::make_unique<franka::Robot>(device_config["franka_ip"].as<std::string>("192.168.3.100"), franka::RealtimeConfig::kIgnore))
    //, gripper(std::make_unique<franka::Gripper>(device_config["franka_ip"].as<std::string>("192.168.3.100")))
#else
    : robot(std::make_unique<franka::Robot>())
    , gripper(std::make_unique<franka::Gripper>())
#endif
    , bRunning(false)
    , state_(SysState::OFFLINE)
    , motion_gen_(InterpolatorConfig{
        .control_freq   = 1000,
        .comm_freq      = device_config["transmission"]["frequency"].as<int>(),
        .n_dof          = 7,
        // The interpolator's speed caps MUST match the safety limiter applied in
        // validateTargetPose, never be tighter than it. They were hardcoded at
        // 0.5 m/s and 0.8 rad/s while safety allowed 1.0 m/s and 4.0 rad/s --
        // 2x tighter in translation, 5x in rotation. planCartesian replans from
        // the current interpolated waypoint each command, so above the cap the
        // reference simply cannot move fast enough and lag accumulates without
        // bound until the operator slows down. It is a cliff, not a gradient:
        // below the cap the cost is ~4.5 ms (half of min_steps), at it the
        // reference saturates. logs/079 and logs/080 show the reference angular
        // rate pinned at p95 = 0.86 rad/s against the old 0.8 cap for much of
        // the episode -- rotation was saturated during ordinary use. Reading
        // both from the same config keys the limiter uses keeps them from
        // drifting apart again.
        .max_linear_vel  = device_config["safety"]["max_command_velocity"].as<double>(),
        .max_angular_vel = device_config["safety"]["max_command_angular_velocity"].as<double>(),
        // Joint plans were timed off max_angular_vel above, which is the
        // END-EFFECTOR rotational cap. Nothing there bounds a joint, and MINJERK
        // peaks at 1.875x the average it was sized for, so a recovery plan
        // reached 7.5 rad/s and faulted the arm mid-homing. Same numbers the
        // robot's own check uses; absent from the config it falls back to the
        // old behaviour and says so.
        .max_joint_vel   = [&device_config]() -> std::vector<double> {
            if (device_config["dq_max"])
                return device_config["dq_max"].as<std::vector<double>>();
            return {};
        }()
    })
    , recovery_(device_config["name"].as<std::string>())
{
    name_ = device_config["name"].as<std::string>();

    #ifdef WITH_FRANKA
        try {
            gripper = std::make_unique<franka::Gripper>(device_config["franka_ip"].as<std::string>());
        } catch (const franka::Exception& e) {
            std::cout << "[WARN] " << name_ << ": gripper connection failed (" << e.what()
                    << ") - continuing without gripper control." << std::endl;
            gripper.reset();
        }
    #endif
    
    auto pos  = device_config["base_pose"]["position"].as<std::vector<double>>();
    auto ori  = device_config["base_pose"]["orientation"].as<std::vector<double>>();

    base_position_ = Eigen::Vector3d(pos[0], pos[1], pos[2]);
    base_orientation_ = Eigen::Quaterniond(ori[0], ori[1], ori[2], ori[3]);

    // Controller-frame -> EE/flange-frame axis remap for body-frame orientation
    // retargeting. Defaults to identity (passthrough). Fill from the single-axis test.
    R_ctrl_to_ee_ = Eigen::Matrix3d::Identity();
    if (device_config["controller_axis_map"]) {
        auto rows = device_config["controller_axis_map"].as<std::vector<std::vector<double>>>();
        if (rows.size() == 3 && rows[0].size() == 3 && rows[1].size() == 3 && rows[2].size() == 3) {
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    R_ctrl_to_ee_(i, j) = rows[i][j];
        } else {
            std::cout << "[WARN] " << device_config["name"].as<std::string>()
                      << ": controller_axis_map must be 3x3 - using identity." << std::endl;
        }
    }
    {
        double det = R_ctrl_to_ee_.determinant();
        Eigen::Matrix3d orth_err = R_ctrl_to_ee_.transpose() * R_ctrl_to_ee_ - Eigen::Matrix3d::Identity();
        if (std::abs(det - 1.0) > 1e-6 || orth_err.cwiseAbs().maxCoeff() > 1e-6) {
            std::cout << "[WARN] " << device_config["name"].as<std::string>()
                      << ": controller_axis_map is not a proper rotation (det=" << det
                      << "). Check signs/handedness - orientation retargeting will be wrong."
                      << std::endl;
        }
    }

    T_base_ = Eigen::Isometry3d::Identity();
    T_base_.translation() = base_position_;
    T_base_.linear()      = base_orientation_.toRotationMatrix();
    target_pose_raw_ = Eigen::Isometry3d::Identity();

    q0_ = yamlToVector<7>(device_config["q0"]);
    q_min_ = yamlToVector<7>(device_config["q_min"]);
    q_max_ = yamlToVector<7>(device_config["q_max"]);
    tau_max_ = yamlToVector<7>(device_config["max_torque"]);
    tau_rate_max_ = yamlToVector<7>(device_config["max_torque_rate"]) / 1000.0;

    // Headroom on the torque-rate limiter. max_torque_rate stays whatever the
    // config says (it is the physical envelope); this is how much of it we let
    // ourselves spend, so the safety margin can be tuned against the jitter of a
    // given host without touching the envelope itself.
    if (device_config["control"]["torque_rate_margin"])
        torque_rate_margin_ = device_config["control"]["torque_rate_margin"].as<double>();
    torque_rate_margin_ = std::clamp(torque_rate_margin_, 0.05, 1.0);

    kp_joint_ = yamlToVector<7>(device_config["control"]["kp_joint"]);
    kd_joint_ = yamlToVector<7>(device_config["control"]["kd_joint"]);

    // IDLE hold gains -- explicit config if present, otherwise a scaled-down
    // fraction of the tracking gains. Deliberately soft: the IDLE target is
    // latched at the *current* pose, so initial error (and therefore initial
    // torque) is zero, and steady-state error is only tau_residual/kp -- well
    // under a degree even at reduced stiffness.
    constexpr double kIdleStiffFrac = 0.40;
    constexpr double kIdleDampFrac  = 0.63;   // ~sqrt(0.40), keeps damping ratio
    if (device_config["control"]["kp_idle"])
        kp_idle_ = yamlToVector<7>(device_config["control"]["kp_idle"]);
    else
        kp_idle_ = kp_joint_ * kIdleStiffFrac;
    if (device_config["control"]["kd_idle"])
        kd_idle_ = yamlToVector<7>(device_config["control"]["kd_idle"]);
    else
        kd_idle_ = kd_joint_ * kIdleDampFrac;

    kp_cart_ = yamlToVector<6>(device_config["control"]["kp_cart"]);
    kd_cart_ = yamlToVector<6>(device_config["control"]["kd_cart"]);
    const YAML::Node ctrl = device_config["control"];
    if (ctrl["eta_lin"])       eta_lin_       = ctrl["eta_lin"].as<double>();
    if (ctrl["eta_rot"])       eta_rot_       = ctrl["eta_rot"].as<double>();
    if (ctrl["ff_force_max"])  ff_force_max_  = ctrl["ff_force_max"].as<double>();
    if (ctrl["ff_torque_max"]) ff_torque_max_ = ctrl["ff_torque_max"].as<double>();
    eta_lin_ = std::clamp(eta_lin_, 0.0, 1.0);
    eta_rot_ = std::clamp(eta_rot_, 0.0, 1.0);
    if (eta_lin_ > 0.0 || eta_rot_ > 0.0)
        std::cout << "[INFO] " << name_ << ": velocity feedforward eta_lin=" << eta_lin_
                  << " eta_rot=" << eta_rot_ << " (cap " << ff_force_max_ << " N / "
                  << ff_torque_max_ << " Nm)" << std::endl;
    kp_null_ = yamlToVector<7>(device_config["control"]["kp_null"]);
    kd_null_ = yamlToVector<7>(device_config["control"]["kd_null"]);
    kd_joint_limit_ = yamlToVector<7>(device_config["control"]["kd_joint_limit"]);
    kp_joint_limit_ = yamlToVector<7>(device_config["control"]["kp_joint_limit"]);
    joint_limit_buffer_  = device_config["safety"]["joint_limit_buffer"].as<double>();
    joint_limit_torque_frac_ = device_config["safety"]["joint_limit_torque_frac"].as<double>();

    // ── Collision behavior / impedance ──────────────────────────────────────
    // Applied once at startup, before the control thread ever calls robot->control().
    // Left optional and explicit: if not present in config we deliberately keep
    // whatever the robot/Desk already has configured rather than guessing values.
    if (device_config["safety"]["collision_lower_torque"] && device_config["safety"]["collision_upper_torque"] &&
        device_config["safety"]["collision_lower_force"]  && device_config["safety"]["collision_upper_force"]) {
        auto lower_torque = toArray<7>(device_config["safety"]["collision_lower_torque"].as<std::vector<double>>());
        auto upper_torque = toArray<7>(device_config["safety"]["collision_upper_torque"].as<std::vector<double>>());
        auto lower_force  = toArray<6>(device_config["safety"]["collision_lower_force"].as<std::vector<double>>());
        auto upper_force  = toArray<6>(device_config["safety"]["collision_upper_force"].as<std::vector<double>>());
        robot->setCollisionBehavior(lower_torque, upper_torque, lower_force, upper_force);
        std::cout << "[INFO] " << name_ << ": collision behavior thresholds applied from config." << std::endl;
    } else {
        std::cout << "[INFO] " << name_ << ": no safety.collision_* thresholds in config - keeping "
                                            "robot/Desk defaults." << std::endl;
    }

    if (device_config["control"]["joint_impedance"]) {
        robot->setJointImpedance(toArray<7>(device_config["control"]["joint_impedance"].as<std::vector<double>>()));
    }
    if (device_config["control"]["cartesian_impedance"]) {
        robot->setCartesianImpedance(toArray<6>(device_config["control"]["cartesian_impedance"].as<std::vector<double>>()));
    }

    if (device_config["transmission"]) {
        UdpStreamConfig stream_cfg;
        stream_cfg.transport.remote_ip   = device_config["transmission"]["remote_ip"].as<std::string>();
        stream_cfg.transport.remote_port = device_config["transmission"]["send_port"].as<int>();
        stream_cfg.transport.bind_port   = device_config["transmission"]["receive_port"].as<int>();
        stream_cfg.send_rate_hz          = device_config["transmission"]["frequency"].as<int>();
        transmission_ = std::make_unique<ArmStream>(stream_cfg);
    }

    // Optional second channel, own port, same ArmCommandMsg struct -- for an
    // autonomous policy's absolute world-frame pose commands (see
    // worldAbsoluteToBase), kept fully separate from transmission_'s
    // delta-from-origin VR path above so nothing sending there is affected.
    // Its outgoing ArmStateMsg echo isn't consumed by anything -- state is
    // already published on transmission_ -- so remote_ip/send_port here just
    // need to be valid, not actually listened to.
    if (device_config["transmission_absolute"]) {
        UdpStreamConfig stream_cfg;
        stream_cfg.transport.remote_ip   = device_config["transmission_absolute"]["remote_ip"].as<std::string>();
        stream_cfg.transport.remote_port = device_config["transmission_absolute"]["send_port"].as<int>();
        stream_cfg.transport.bind_port   = device_config["transmission_absolute"]["receive_port"].as<int>();
        stream_cfg.send_rate_hz          = device_config["transmission_absolute"]["frequency"].as<int>();
        transmission_absolute_ = std::make_unique<ArmStream>(stream_cfg);
    }

    workspace_min_ = Eigen::Vector3d(
        device_config["safety"]["workspace_min"][0].as<double>(),
        device_config["safety"]["workspace_min"][1].as<double>(),
        device_config["safety"]["workspace_min"][2].as<double>()
    );
    workspace_max_ = Eigen::Vector3d(
        device_config["safety"]["workspace_max"][0].as<double>(),
        device_config["safety"]["workspace_max"][1].as<double>(),
        device_config["safety"]["workspace_max"][2].as<double>()
    );
    table_height_world_ = device_config["safety"]["table_height_world"].as<double>();
    table_safety_margin_ = device_config["safety"]["table_safety_margin"].as<double>();
    max_command_velocity_ = device_config["safety"]["max_command_velocity"].as<double>();
    max_command_angular_velocity_ = device_config["safety"]["max_command_angular_velocity"].as<double>();
    if (device_config["safety"]["max_target_lead"]) {
        max_target_lead_ = device_config["safety"]["max_target_lead"].as<double>();
        std::cout << "[INFO] " << name_ << ": target leash " << max_target_lead_ * 1000.0
                  << " mm" << std::endl;
    } else {
        std::cout << "[INFO] " << name_ << ": no safety.max_target_lead in config - "
                  << "target may lead the measured pose without bound." << std::endl;
    }
    ee_fingertip_length_ = device_config["safety"]["ee_fingertip_length"].as<double>();
    max_tilt_angle_ = device_config["safety"]["max_tilt_angle"].as<double>();
    cmd_dt_ = 1.0 / static_cast<double>(device_config["transmission"]["frequency"].as<int>());

    // ── Control mode ─────────────────────────────────────────────────────────
    control_mode_ = ControlMode::CARTESIAN_IMPEDANCE;
    if (device_config["control_mode"]) {
        std::string mode_str = device_config["control_mode"].as<std::string>();
        if (mode_str == "joint_ik") {
            control_mode_ = ControlMode::JOINT_IK;
            std::cout << "[INFO] " << name_ << ": control mode = JOINT_IK" << std::endl;
        } else {
            std::cout << "[INFO] " << name_ << ": control mode = CARTESIAN_IMPEDANCE" << std::endl;
        }
    }

    {
        IkConfig ik_cfg;
        ik_cfg.q0    = q0_;
        ik_cfg.q_min = q_min_;
        ik_cfg.q_max = q_max_;
        ik_cfg.qd_max = (Eigen::Matrix<double,7,1>() << 2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610).finished();

        if (device_config["ik"]) {
            const auto& ik = device_config["ik"];
            if (ik["kp_p"]) {
                auto v = ik["kp_p"].as<std::vector<double>>();
                ik_cfg.Kp_p = Eigen::Vector3d(v[0], v[1], v[2]);
            }
            if (ik["kp_o"])       ik_cfg.Kp_o     = ik["kp_o"].as<double>();
            if (ik["v_lin_max"])  ik_cfg.v_lin_max = ik["v_lin_max"].as<double>();
            if (ik["v_ang_max"])  ik_cfg.v_ang_max = ik["v_ang_max"].as<double>();
            if (ik["lambda"])     ik_cfg.lambda    = ik["lambda"].as<double>();
            if (ik["kp_posture"]) ik_cfg.Kp_posture = yamlToVector<7>(ik["kp_posture"]);
            if (ik["qd_max"])     ik_cfg.qd_max    = yamlToVector<7>(ik["qd_max"]);
            if (ik["t_brake"])    ik_cfg.T_brake   = ik["t_brake"].as<double>();
            if (ik["a_max"])      ik_cfg.a_max     = ik["a_max"].as<double>();
            if (ik["gamma"])      ik_cfg.gamma     = ik["gamma"].as<double>();
            if (ik["wtask"])      ik_cfg.Wtask     = yamlToVector<6>(ik["wtask"]);
        }
        motion_gen_.setIkConfig(ik_cfg);
    }

    if (device_config["gripper"]) {
        const auto& gripper_cfg = device_config["gripper"];
        if (gripper_cfg["grasp_confirm_tolerance_m"])
            grasp_confirm_tolerance_m_ = gripper_cfg["grasp_confirm_tolerance_m"].as<double>();
        if (gripper_cfg["grasp_confirm_time_s"])
            grasp_confirm_time_s_ = gripper_cfg["grasp_confirm_time_s"].as<double>();
    }

    // ── Thread placement ────────────────────────────────────────────────────
    // Defaults reproduce the previous hard-coded mapping (arm_left -> 0/1,
    // arm_right -> 2/3). Overridable because the right cores are a property of
    // the machine, not of the arm: on a P/E-core host you want the 1 kHz loop on
    // a P-core, and on any Linux box you want it off core 0.
    rt_control_core_ = (name_ == "arm_right") ? 2 : 0;
    rt_state_core_   = (name_ == "arm_right") ? 3 : 1;
    if (device_config["rt"]) {
        if (device_config["rt"]["control_core"]) rt_control_core_ = device_config["rt"]["control_core"].as<int>();
        if (device_config["rt"]["state_core"])   rt_state_core_   = device_config["rt"]["state_core"].as<int>();
        if (device_config["rt"]["state_rate_hz"]) state_rate_hz_ = device_config["rt"]["state_rate_hz"].as<double>();
    }
    if (state_rate_hz_ < 1.0) {
        std::cout << "[WARN] " << name_ << ": rt.state_rate_hz = " << state_rate_hz_
                  << " is not usable, falling back to 200." << std::endl;
        state_rate_hz_ = 200.0;
    }

    logger_ = std::make_unique<DataLogger<ArmLogEntry>>("../log/" + name_ + "_log.csv", armLogHeader, armLogRow, session_id);
    state_trace_ = std::make_unique<DataLogger<ArmStateTraceEntry>>(
        "../log/" + name_ + "_state_trace.csv", armStateTraceHeader, armStateTraceRow, session_id);
}

ArmControl::~ArmControl(){
    stop();
}

void ArmControl::start(){
    // We run libfranka with RealtimeConfig::kIgnore, which turns "cannot get RT
    // priority" from an exception into silence. Say it out loud at startup instead.
    warn_if_no_realtime(name_);

    bRunning = true;
    state_ = SysState::IDLE;
    cmd_state_ = SysState::IDLE;
    current_state = robot->readOnce();
#ifdef WITH_FRANKA
    franka_owned_model_ = std::make_unique<franka::Model>(robot->loadModel());
    model = franka_owned_model_.get();
#else
    model = &robot->loadModel();
#endif
    Eigen::Map<const Vector7> q_init(current_state.q.data());
    motion_gen_.planJoint(q_init, q_init, ProfileType::TRAPEZOIDAL);
    control_thread = std::thread(&ArmControl::runControlHandler, this);
    set_realtime(control_thread, rt_control_core_);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    state_thread = std::thread(&ArmControl::runStateHandler, this);
    set_realtime(state_thread, rt_state_core_);
    // Wake the state thread the moment a command lands rather than letting it
    // sit until the next periodic tick. Must be set before start() -- the
    // callback is read unlocked on the receive thread.
    if (transmission_) transmission_->setOnReceive([this]{ notifyCommandArrived(); });
    if (transmission_absolute_) transmission_absolute_->setOnReceive([this]{ notifyCommandArrived(); });
    if (transmission_) transmission_->start();
    if (transmission_absolute_) transmission_absolute_->start();
    logger_->start();
    logger_->enable(true);
    if (state_trace_) { state_trace_->start(); state_trace_->enable(true); }
    startTime_ = std::chrono::high_resolution_clock::now();
}

void ArmControl::stop(){
    if (logger_) logger_->stop();
    if (state_trace_) state_trace_->stop();
    bRunning = false;
    state_ = SysState::OFFLINE;
    // Release runStateHandler if it is parked in waitForCommandOrDeadline;
    // without this the join below waits out one full period.
    notifyCommandArrived();
    if (control_thread.joinable()) control_thread.join();
    if (state_thread.joinable()) state_thread.join();
    if (transmission_) transmission_->stop();
    if (transmission_absolute_) transmission_absolute_->stop();
}

void ArmControl::runStateHandler(){
    // Loop period and the dt handed to stepIk now come from one number.
    // They used to be independently hardcoded -- a 200 Hz period sitting next to
    // dt_state = 1/500 -- so stepIk integrated q_ref += u*dt with a dt 2.5x too
    // small: the IK reference advanced at 40% of the commanded joint velocity
    // and the a_max*dt acceleration bound was 2.5x tighter than configured.
    // JOINT_IK was running 2.5x slower than it was tuned for. Dead code while
    // control_mode is cartesian_impedance, but silent and confusing the moment
    // the IK path is revisited.
    const auto control_period =
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(1.0 / state_rate_hz_));
    const double dt_state = 1.0 / state_rate_hz_;
    auto next_control_time = std::chrono::steady_clock::now();
    SysState prev_state = SysState::OFFLINE;
    Eigen::VectorXd q_current = Eigen::VectorXd::Zero(7);
    bool has_cmd = false;
    bool has_cmd_abs = false;
    ArmCommandMsg cmd;
    ArmCommandMsg cmd_abs;
    Eigen::Quaterniond prev_cmd_quat_ = Eigen::Quaterniond::Identity();

    while(bRunning){

        if (transmission_ && transmission_->hasNew()) {
            cmd = transmission_->getRecvData();
            has_cmd = true;
            desired_gripper_closed_.store(cmd.gripper > 0.5f);
        }
        if (transmission_absolute_ && transmission_absolute_->hasNew()) {
            cmd_abs = transmission_absolute_->getRecvData();
            has_cmd_abs = true;
            desired_gripper_closed_.store(cmd_abs.gripper > 0.5f);
        }

        // ── Early wake ───────────────────────────────────────────────────────
        // A command landed before the periodic deadline. Push it straight
        // through to the target and go back to waiting instead of holding it
        // until the next tick. Only the command -> target path runs here: the
        // state machine, telemetry publish, gripper read and state trace stay on
        // the periodic tick, so none of their rates follow the command rate when
        // comms move to 500 Hz. prev_state is deliberately not touched, so a
        // fast iteration can never swallow a state-entry block.
        if (std::chrono::steady_clock::now() < next_control_time) {
            if (state_ == SysState::ENGAGED && (has_cmd || has_cmd_abs))
                applyOperatorCommand(cmd, cmd_abs, has_cmd, has_cmd_abs, prev_cmd_quat_);
            waitForCommandOrDeadline(next_control_time);
            continue;
        }

        updateRecovery();
        updateStateMachine(cmd_state_);

        if (state_ != SysState::ENGAGED) {
            has_cmd = false;
            has_cmd_abs = false;
        }

        // ── HOMING entry ──────────────────────────────────────────────────────
        if (state_ == SysState::HOMING && prev_state != SysState::HOMING) {
            {
                std::lock_guard<std::mutex> lock(state_mtx);
                q_current = Eigen::Map<const Vector7>(current_state.q.data());
            }
            motion_gen_.planJoint(q_current, q0_, ProfileType::MINJERK);
        }

        // ── IDLE entry: latch the current configuration as the hold target ────
        // Without this the arm is commanded zero torque, i.e. gravity-compensated
        // float. That is neutral equilibrium -- no restoring term anywhere -- so
        // any model residual or leftover velocity integrates into unbounded drift
        // rather than being corrected.
        else if (state_ == SysState::IDLE && prev_state != SysState::IDLE) {
            {
                std::lock_guard<std::mutex> lock(state_mtx);
                q_current = Eigen::Map<const Vector7>(current_state.q.data());
            }
            // Zero-length plan: getCurrentJoint() parks at q_current, so the
            // impedance target is FIXED rather than tracking the live pose.
            // Re-reading the live pose every tick would recreate neutral equilibrium
            // and drift exactly as before.
            motion_gen_.planJoint(q_current, q_current, ProfileType::MINJERK);
            idle_hold_valid_.store(true, std::memory_order_release);
            std::cout << "[INFO]: " << name_ << " idle hold latched." << std::endl;
        }

        // ── ENGAGED entry: seed IK to current robot state ─────────────────────
        else if (state_ == SysState::ENGAGED && prev_state != SysState::ENGAGED
                 && control_mode_ == ControlMode::JOINT_IK) {
            franka::RobotState rs;
            {
                std::lock_guard<std::mutex> lock(state_mtx);
                rs = current_state;
            }
            Vector7 q_seed = Eigen::Map<const Vector7>(rs.q.data());
            Eigen::Isometry3d T_ee(Eigen::Map<const Eigen::Matrix4d>(rs.O_T_EE.data()));
            motion_gen_.seedJointReference(q_seed);
            motion_gen_.setCartesianGoal(T_ee);
            has_planned_target_ = false;
        }

        // ── ENGAGED tick ──────────────────────────────────────────────────────
        else if (state_ == SysState::ENGAGED) {
            if (has_cmd || has_cmd_abs) {
                applyOperatorCommand(cmd, cmd_abs, has_cmd, has_cmd_abs, prev_cmd_quat_);

            } else if (control_mode_ == ControlMode::CARTESIAN_IMPEDANCE) {
                Eigen::Isometry3d T_current_target = motion_gen_.getCurrentCartesian();
                Eigen::Isometry3d T_filtered = T_current_target;
                applySelfCollisionFilter(T_filtered);

                double pos_change = (T_filtered.translation() - T_current_target.translation()).norm();
                if (pos_change > 1e-6) {
                    motion_gen_.planCartesian(motion_gen_.getCurrentCartesian(), T_filtered, ProfileType::LINEAR);
                    target_pose_ = T_base_ * T_filtered;
                }
            }

            if (control_mode_ == ControlMode::JOINT_IK) {
                franka::RobotState rs;
                {
                    std::lock_guard<std::mutex> lock(state_mtx);
                    rs = current_state;
                }
                Vector7 q_ref = motion_gen_.getJointReference();
                franka::RobotState rs_ref = rs;
                Eigen::Map<Vector7>(rs_ref.q.data()) = q_ref;

                auto J_array = model->zeroJacobian(franka::Frame::kEndEffector, rs_ref);
                Matrix6x7 J  = Eigen::Map<Matrix6x7>(J_array.data());

                std::array<double, 16> pose_arr;
#ifdef WITH_FRANKA
                pose_arr = model->pose(franka::Frame::kEndEffector, rs_ref.q, rs_ref.F_T_EE, rs_ref.EE_T_K);
#else
                pose_arr = model->EEPose(rs_ref.q);
#endif
                Eigen::Isometry3d x_ref(Eigen::Map<const Eigen::Matrix4d>(pose_arr.data()));

                motion_gen_.stepIk(q_ref, J, x_ref, dt_state);
            }
        }

        if (transmission_) {
            franka::RobotState rs;
            uint64_t rs_sample_ns;
            {
                std::lock_guard<std::mutex> lock(state_mtx);
                rs = current_state;
                // Read under the same lock as the state itself, so the stamp
                // always belongs to the snapshot we just took rather than to a
                // newer one that landed in between.
                rs_sample_ns = state_sample_ns_.load(std::memory_order_relaxed);
            }
            Eigen::Isometry3d T_ee(Eigen::Map<const Eigen::Matrix4d>(rs.O_T_EE.data()));

            if (scp_ && (state_ == SysState::ENGAGED || state_ == SysState::AWAITING)) {
                auto J_array = model->zeroJacobian(franka::Frame::kEndEffector, rs);
                Matrix6x7 J  = Eigen::Map<Matrix6x7>(J_array.data());
                Eigen::Map<const Vector7> dq(rs.dq.data());
                Eigen::Vector3d ee_vel = (J * dq).head<3>();

                CollisionState ds;
                ds.ee_position = T_base_.rotation() * T_ee.translation() + T_base_.translation();
                ds.ee_velocity = T_base_.rotation() * ee_vel;
                ds.weight      = scp_state_.weight;
                scp_->publishState(ds);
            }

            Eigen::Isometry3d T_ee_world = T_base_ * T_ee;
            Eigen::Quaterniond q_ee_world(T_ee_world.rotation());
            ArmStateMsg state_msg{};
            state_msg.position[0] = static_cast<float>(T_ee_world.translation().x());
            state_msg.position[1] = static_cast<float>(T_ee_world.translation().y());
            state_msg.position[2] = static_cast<float>(T_ee_world.translation().z());
            state_msg.quaternion[0] = static_cast<float>(q_ee_world.w());
            state_msg.quaternion[1] = static_cast<float>(q_ee_world.x());
            state_msg.quaternion[2] = static_cast<float>(q_ee_world.y());
            state_msg.quaternion[3] = static_cast<float>(q_ee_world.z());
            state_msg.recovering    = (state_ == SysState::RECOVERING) ? 1 : 0;
            state_msg.gripper_width = static_cast<float>(gripper_width_.load());
            state_msg.grasp_state   = grasp_state_.load();
            // When the CONTROL thread last read the robot. doSend() fills in
            // sequence/timestamp_ns at send time; this is the one field that
            // stops advancing if the control loop dies, which is the whole
            // point of it (see MsgHeader in common.hpp).
            state_msg.header.sample_time_ns = rs_sample_ns;
            // Echo the last operator command we acted on, so the interface can
            // difference it against its own send timestamp (see common.hpp).
            state_msg.applied_cmd_sequence = applied_cmd_seq_.load(std::memory_order_relaxed);
            transmission_->setSendData(state_msg);
        }

        const bool grasp_allowed = (state_ == SysState::ENGAGED || state_ == SysState::PAUSED);
        grasp_allowed_.store(grasp_allowed);
        applyGripper(grasp_allowed && desired_gripper_closed_.load());

        const double width = gripper ? gripper->readOnce().width : 0.0;
        gripper_width_.store(width);
        updateGraspConfirmation(width);

        // ── state trace ───────────────────────────────────────────────────────
        // Written here rather than in the control callback so it keeps going
        // through faults, automaticErrorRecovery() and the blocking FAULT wait
        // -- the three situations where arm.csv goes silent and where knowing
        // what happened matters most.
        if (state_trace_) {
            const uint64_t now_ns    = timestamp_ns();
            const uint64_t sample_ns = state_sample_ns_.load(std::memory_order_relaxed);
            ArmStateTraceEntry tr{};
            tr.time = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - startTime_).count();
            tr.wall_clock_ns        = now_ns;
            tr.control_sample_ns    = sample_ns;
            // Grows without bound while the control thread is not running. This
            // is the single column to plot when asking "was the robot alive?".
            tr.control_age_ms       = (sample_ns == 0 || now_ns < sample_ns)
                                        ? -1.0
                                        : static_cast<double>(now_ns - sample_ns) / 1e6;
            tr.control_loop_entries = control_loop_entries_.load(std::memory_order_relaxed);
            tr.fault_count          = fault_streak_.load(std::memory_order_relaxed);
            tr.state                = state_;
            tr.recovering           = (state_ == SysState::RECOVERING) ? 1 : 0;
            state_trace_->write(tr);
        }

        prev_state = state_;
        next_control_time += control_period;
        // Fell far behind (host contention, a long gripper call): resynchronise
        // rather than sprint through a burst of catch-up ticks. Same rationale
        // as Robot::control.
        const auto now_tp = std::chrono::steady_clock::now();
        if (now_tp - next_control_time > std::chrono::milliseconds(50))
            next_control_time = now_tp;
        waitForCommandOrDeadline(next_control_time);
    }
}

// Sleep to the periodic deadline, or return the instant a command arrives.
// Replaces sleep_until(next_control_time): the command path used to sit behind
// two independent polls in series -- UdpStream polling a non-blocking socket at
// send_rate_hz, then this thread polling hasNew() -- for 0-10 ms of pure
// waiting on a packet that had already arrived.
void ArmControl::waitForCommandOrDeadline(
        const std::chrono::steady_clock::time_point& deadline) {
    std::unique_lock<std::mutex> lock(cmd_wake_mtx_);
    cmd_wake_cv_.wait_until(lock, deadline, [this] {
        return cmd_wake_flag_ || !bRunning.load(std::memory_order_relaxed);
    });
    cmd_wake_flag_ = false;
}

void ArmControl::notifyCommandArrived() {
    {
        std::lock_guard<std::mutex> lock(cmd_wake_mtx_);
        cmd_wake_flag_ = true;
    }
    cmd_wake_cv_.notify_one();
}

// Body lifted verbatim out of the ENGAGED tick so the early-wake path and the
// periodic path apply a command identically. Called only from the state thread,
// so target_pose_/target_pose_raw_ and prev_cmd_quat_ keep their single-writer
// invariant.
void ArmControl::applyOperatorCommand(const ArmCommandMsg& cmd, const ArmCommandMsg& cmd_abs,
                                      bool& has_cmd, bool& has_cmd_abs,
                                      Eigen::Quaterniond& prev_cmd_quat_) {
    // Absolute (autonomous policy) takes priority if both arrived this tick --
    // shouldn't happen in practice, since SystemArbitrator/policy mode gating
    // means only one sender is ever actually active, but this keeps it
    // deterministic rather than order-of-arrival dependent.
    bool absolute = has_cmd_abs;
    const ArmCommandMsg& src = absolute ? cmd_abs : cmd;

    // Record which command we are about to ACT on, for the echo in the outgoing
    // ArmStateMsg. Deliberately set here rather than where the packet is
    // received: a command that arrived but was dropped (not ENGAGED, superseded
    // within the same tick) never moved the robot, and echoing it would
    // understate the latency the operator actually experiences.
    //
    // Only the relative/operator stream is echoed. The absolute stream comes
    // from an autonomous policy, not from the VR interface, so there is no send
    // timestamp on the operator side to difference against.
    if (!absolute) {
        applied_cmd_seq_.store(src.header.sequence, std::memory_order_relaxed);
    }

    Eigen::Isometry3d T_cmd = Eigen::Isometry3d::Identity();
    Eigen::Vector3d pos(src.position[0], src.position[1], src.position[2]);
    Eigen::Quaterniond q(src.quaternion[0], src.quaternion[1], src.quaternion[2], src.quaternion[3]);
    q.normalize();
    if (q.dot(prev_cmd_quat_) < 0.0) q.coeffs() *= -1.0;
    prev_cmd_quat_ = q;
    T_cmd.translation() = pos;
    T_cmd.linear() = q.toRotationMatrix();

    // Absolute: world-frame target, no origin/controller-remap involved
    // (worldAbsoluteToBase). Otherwise: existing delta-from-origin VR semantics
    // (transformCommandToBase), unchanged.
    Eigen::Isometry3d T_target = absolute ? worldAbsoluteToBase(T_cmd) : transformCommandToBase(T_cmd);
    target_pose_raw_ = T_base_ * T_target;
    applySelfCollisionFilter(T_target);
    validateTargetPose(T_target);

    if (control_mode_ == ControlMode::JOINT_IK) {
        // IK goal update -- goal is frozen when commands stop
        motion_gen_.setCartesianGoal(T_target);
    } else if (!has_planned_target_ || !targetsEqual(T_target, last_planned_target_)) {
        motion_gen_.setCommandInterval(last_cmd_dt_);
        motion_gen_.planCartesian(motion_gen_.getCurrentCartesian(), T_target, ProfileType::LINEAR);
        last_planned_target_ = T_target;
        has_planned_target_  = true;
    }

    target_pose_ = T_base_ * T_target;
    has_cmd = false;
    has_cmd_abs = false;
}

void ArmControl::updateRecovery() {
    RecoveryRequest req = recovery_.consumePending();
    if (req.valid) {
        Vector7 q_current;
        {
            std::lock_guard<std::mutex> lock(state_mtx);
            q_current = Eigen::Map<const Vector7>(current_state.q.data());
        }
        if (q_current.norm() < 1e-6) {
            recovery_.pushBack(req);
            return;
        }
        // Under state_mtx because runControlHandler's rearmFromMeasuredState reads
        // it from the control thread when restarting a faulted loop.
        {
            std::lock_guard<std::mutex> lock(state_mtx);
            recovery_target_q_ = req.target_q;
        }
        // Drop the pre-fault operator target. rearmFromMeasuredState already
        // re-plans the motion generator from the measured pose, but these two
        // survived it: the first command after a reset was rate-limited against
        // a prev_valid_target_pos_ from before the fault, and targetsEqual()
        // against a stale last_planned_target_ could suppress the replan
        // entirely, leaving the arm running the recovery trajectory. Written on
        // the state thread, same as validateTargetPose. Safe to clear now that
        // max_target_lead_ bounds the first unlimited command.
        has_prev_valid_target_ = false;
        has_planned_target_    = false;

        motion_gen_.planJoint(q_current, req.target_q, ProfileType::MINJERK);
        recovery_.setMode(RecoveryMode::MOVING_TO_SAFE);
        state_ = SysState::RECOVERING;
        recovery_start_time_ = std::chrono::steady_clock::now();
        if (transmission_) transmission_->setState(state_);
        std::cout << "[INFO]: " << name_ << " recovery motion started." << std::endl;
        return;
    }

    switch (recovery_.mode()) {
        case RecoveryMode::MOVING_TO_SAFE: {
            Vector7 q, dq;
            {
                std::lock_guard<std::mutex> lock(state_mtx);
                q  = Eigen::Map<const Vector7>(current_state.q.data());
                dq = Eigen::Map<const Vector7>(current_state.dq.data());
            }
            Vector7 q_final = recovery_target_q_;
            bool trajectory_done = motion_gen_.isDone();
            bool arrived = (q_final - q).cwiseAbs().maxCoeff() < 0.3;
            bool settled = dq.cwiseAbs().maxCoeff() < 0.07;
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - recovery_start_time_).count();
            bool timed_out = trajectory_done && elapsed > 5;
            if ((trajectory_done && arrived && settled) || timed_out) {
                Eigen::Isometry3d T_ee;
                {
                    std::lock_guard<std::mutex> lock(state_mtx);
                    T_ee = Eigen::Isometry3d(Eigen::Map<const Eigen::Matrix4d>(current_state.O_T_EE.data()));
                }
                motion_gen_.planCartesian(T_ee, T_ee);
                recovery_.setMode(RecoveryMode::WAITING_ACK);
                if (timed_out) {
                    std::cout << "[WARN]: " << name_ << " recovery timed out, residual joint err: "
                              << (q_final - q).cwiseAbs().maxCoeff() << " rad" << std::endl;
                } else {
                    std::cout << "[INFO]: " << name_ << " recovery motion done, awaiting operator." << std::endl;
                }
            }
            break;
        }
        case RecoveryMode::WAITING_ACK:
            if (recovery_.shouldResume()) {
                recovery_.setMode(RecoveryMode::NONE);
                state_ = SysState::AWAITING;
                if (transmission_) transmission_->setState(state_);
                std::cout << "[INFO]: " << name_ << " recovery complete, awaiting engagement." << std::endl;
            }
            break;
        default:
            break;
    }
}

void ArmControl::updateStateMachine(SysState cmd_state){
    if (state_ == SysState::RECOVERING) return;

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
                motion_gen_.planCartesian(T_origin_, T_origin_);
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
                has_prev_valid_target_ = false;
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
    // Invalidate the IDLE hold the instant we transition INTO idle, so the 1 kHz
    // control loop cannot hold against a stale motion_gen_ target (e.g. q0 left
    // over from HOMING) in the window before runStateHandler latches a new one.
    if (state_ != prev && state_ == SysState::IDLE) {
        idle_hold_valid_.store(false, std::memory_order_release);
    }
    if (state_ != prev && transmission_) {
        transmission_->setState(state_);
    }
}

void ArmControl::runControlHandler(){
    Vector7 tau_prev_ = Vector7::Zero();

    // Hoisted out of the callback: one multiply we do not need to repeat 1000x/s,
    // and it makes the effective limit visible in one place.
    const Vector7 tau_rate_step = tau_rate_max_ * torque_rate_margin_;

    std::function<franka::Torques(const franka::RobotState&, franka::Duration)>
        control_callback = [&](const franka::RobotState& robot_state, franka::Duration) -> franka::Torques {
            
            motion_gen_.step();
            {
                std::lock_guard<std::mutex> lock(state_mtx);
                current_state = robot_state;
                // Freshness stamp for outgoing telemetry. Written here and
                // nowhere else: this is the only place the robot is actually
                // read, so if this loop stops advancing, so does the stamp,
                // and every consumer can see it.
                state_sample_ns_.store(timestamp_ns(), std::memory_order_relaxed);
            }
            Vector7 ctrl_torque = Vector7::Zero();

            switch(state_){
                case SysState::HOMING:
                case SysState::RECOVERING:
                    // Joint impedance tracking planJoint trajectory
                    ctrl_torque = jointImpedanceControl(robot_state);
                    break;

                case SysState::IDLE:
                    // Hold the configuration latched on IDLE entry. Gated on
                    // idle_hold_valid_ so we never impedance-track an empty or
                    // stale motion_gen_ buffer (see arm_control.hpp).
                    if (idle_hold_valid_.load(std::memory_order_acquire))
                        ctrl_torque = jointImpedanceControl(robot_state);
                    break;

                case SysState::AWAITING:
                    // Always hold Cartesian pose while awaiting engagement
                    ctrl_torque = cartesianImpedanceControl(robot_state);
                    break;

                case SysState::ENGAGED:
                    if (control_mode_ == ControlMode::JOINT_IK)
                        ctrl_torque = jointImpedanceControl(robot_state);
                    else
                        ctrl_torque = cartesianImpedanceControl(robot_state);
                    break;

                default:
                    break;
            }

            ctrl_torque = tau_prev_ + (ctrl_torque - tau_prev_).cwiseMax(-tau_rate_step).cwiseMin(tau_rate_step);
            ctrl_torque = ctrl_torque.cwiseMax(-tau_max_).cwiseMin(tau_max_);
            tau_prev_ = ctrl_torque;

            if (logger_) {
                double t = std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - startTime_).count();
                Vector7 q_target = Vector7::Zero();
                Matrix4 T_target = Matrix4::Identity();
                if(state_ == SysState::HOMING || state_ == SysState::RECOVERING ||
                   (state_ == SysState::IDLE && idle_hold_valid_.load(std::memory_order_acquire))){
                    q_target = motion_gen_.getCurrentJoint();
                }
                else if(state_ == SysState::ENGAGED && control_mode_ == ControlMode::JOINT_IK){
                    q_target = motion_gen_.getJointReference();
                    T_target = motion_gen_.getCartesianGoal().matrix();
                }
                else if(state_ == SysState::ENGAGED || state_ == SysState::AWAITING){
                    Eigen::Isometry3d T_ee_target = motion_gen_.getCurrentCartesian();
                    T_target = T_ee_target.matrix();
                }

                ArmLogEntry entry{};
                entry.time          = t;
                entry.wall_clock_ns = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count());
#ifdef WITH_FRANKA
                entry.sim_time      = 0.0;                  // no sim clock on hardware
#else
                entry.sim_time      = robot_state.sim_time; // mjData::time, see ArmLogEntry
#endif
                entry.state = state_;
                entry.gripper_width = gripper_width_.load();
                entry.gripper_cmd   = (grasp_allowed_.load() && desired_gripper_closed_.load()) ? 0.0 : 0.08;
                entry.grasp_state = static_cast<uint8_t>(grasp_state_.load());
                std::copy(robot_state.q.begin(),                    robot_state.q.end(),                    entry.q.begin());
                Eigen::Map<Vector7>(entry.q_cmd.data()) = q_target;
                std::copy(robot_state.dq.begin(),                   robot_state.dq.end(),                   entry.dq.begin());
                std::copy(robot_state.tau_J.begin(),                robot_state.tau_J.end(),                entry.tau_J.begin());
                // Post rate-limit, post-saturation: exactly the vector handed to
                // franka::Torques below. diff() this per tick and compare against
                // max_torque_rate to see a discontinuity instead of guessing at one.
                Eigen::Map<Vector7>(entry.tau_cmd.data()) = ctrl_torque;
                std::copy(robot_state.tau_ext_hat_filtered.begin(), robot_state.tau_ext_hat_filtered.end(), entry.tau_ext.begin());
                std::copy(robot_state.O_T_EE.begin(),               robot_state.O_T_EE.end(),               entry.O_T_EE.begin());
                Eigen::Map<Matrix4>(entry.O_T_EE_cmd.data()) = T_target;
                // World-frame counterparts of the two above (T_base_ * local), additive --
                // same T_base_ * T composition already used live for ArmStateMsg (see
                // publishArmState) and target_pose_, just also logged per-tick here so
                // future policy training can consume world-frame poses directly instead
                // of a base-frame pose tied to this arm's mounting calibration.
                Eigen::Isometry3d T_ee_raw(Eigen::Map<const Eigen::Matrix4d>(robot_state.O_T_EE.data()));
                Eigen::Map<Matrix4>(entry.O_T_EE_world.data())     = (T_base_ * T_ee_raw).matrix();
                Eigen::Map<Matrix4>(entry.O_T_EE_cmd_world.data()) = (T_base_ * Eigen::Isometry3d(T_target)).matrix();
                std::copy(robot_state.O_F_ext_hat_K.begin(),        robot_state.O_F_ext_hat_K.end(),        entry.F_ext.begin());
                logger_->write(entry);
            }
            std::array<double, 7> ctrl_array;
            Eigen::Map<Vector7>(ctrl_array.data()) = ctrl_torque;

            franka::Torques tau(ctrl_array);
            tau.motion_finished = !bRunning;
            return tau;
        };

    constexpr int kMaxConsecutiveFaults = 3;
    constexpr auto kFaultStreakWindow   = std::chrono::seconds(5);
    int  fault_count      = 0;
    bool first_attempt    = true;
    bool have_prior_fault = false;
    std::chrono::steady_clock::time_point last_fault_time{};

    // ── Re-arm against the robot's MEASURED state before handing control back ──
    //
    // Two things change under us whenever robot->control() returns: the robot's
    // internal tau_J_d drops to zero, and the arm has usually moved (reflex stop,
    // then automaticErrorRecovery).
    //
    // Restarting without accounting for that is what turned a single reflex into
    // a fault loop. tau_prev_ lives outside this retry loop, so the first command
    // of the new control loop was tau_prev_ +/- one rate-limiter step -- i.e. it
    // jumped from 0 straight back to whatever torque was being commanded when the
    // reflex fired. At 15 Nm that is a 15000 Nm/s step on tick one, well past the
    // 1000 Nm/s FCI limit, so it tripped controller_torque_discontinuity before a
    // single command landed. That is the control_command_success_rate: 0 signature
    // on retries #2+.
    //
    // Zeroing tau_prev_ makes the existing rate limiter double as a soft-start
    // (~0.9 Nm/tick, so ~17 ms to climb back to 15 Nm), and re-planning from the
    // measured pose stops the impedance error from being large to begin with.
    auto rearmFromMeasuredState = [this, &tau_prev_]() {
        tau_prev_.setZero();

        Vector7 q, recovery_goal;
        Eigen::Isometry3d T_ee;
        {
            std::lock_guard<std::mutex> lock(state_mtx);
            q             = Eigen::Map<const Vector7>(current_state.q.data());
            T_ee          = Eigen::Isometry3d(Eigen::Map<const Eigen::Matrix4d>(current_state.O_T_EE.data()));
            recovery_goal = recovery_target_q_;
        }
        // No usable robot state yet -- leave the existing plan alone rather than
        // latching onto zeros (that would command a full-speed move to q = 0).
        if (!q.allFinite() || q.norm() < 1e-9) return;

        switch (state_.load()) {
            case SysState::HOMING:
                // Still going to q0, just re-planned from where the arm actually is.
                motion_gen_.planJoint(q, q0_, ProfileType::MINJERK);
                break;
            case SysState::RECOVERING:
                motion_gen_.planJoint(q, recovery_goal, ProfileType::MINJERK);
                break;
            case SysState::IDLE:
                motion_gen_.planJoint(q, q, ProfileType::MINJERK);
                idle_hold_valid_.store(true, std::memory_order_release);
                break;
            default:
                // AWAITING / ENGAGED / PAUSED: hold the measured pose. The operator
                // has to re-engage the stream anyway, and starting from zero error
                // is the whole point of this function.
                if (control_mode_ == ControlMode::JOINT_IK) {
                    motion_gen_.seedJointReference(q);
                    motion_gen_.setCartesianGoal(T_ee);
                } else {
                    motion_gen_.planCartesian(T_ee, T_ee);
                }
                break;
        }
    };

    // Enter FAULT and block the control thread (not exit it) until an operator
    // clears it. The existing arm_reset / reset_all commands already drive
    // ArmRecovery -> updateRecovery() on the state thread, which moves state_
    // out of FAULT into RECOVERING on its own - we just wait for that to happen.
    auto enterFaultAndWaitForReset = [this]() {
        state_ = SysState::FAULT;
        if (transmission_) transmission_->setState(state_, FaultCode::INTERNAL_ERROR);
        std::cout << "[WARN] " << name_
                  << ": control loop faulted - holding in FAULT until an operator reset "
                     "(arm_reset / reset_all)." << std::endl;
        while (bRunning && state_ == SysState::FAULT) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        if (bRunning) {
            std::cout << "[INFO] " << name_ << ": FAULT cleared, resuming control." << std::endl;
        }
    };

    while (bRunning) {
        // Only on a restart: on the very first entry start() has already seeded
        // motion_gen_ and tau_prev_ is zero by construction.
        if (!first_attempt) rearmFromMeasuredState();
        first_attempt = false;

        try {
            // Counted so the state trace can distinguish "ran clean" from
            // "faulted and silently retried". A bump here with no matching gap
            // in arm.csv means a fault was absorbed without the operator ever
            // being told.
            control_loop_entries_.fetch_add(1, std::memory_order_relaxed);
            robot->control(control_callback);
            break;  // clean stop: control_callback set motion_finished from ArmControl::stop()
        } catch (const franka::ControlException& e) {
            const auto now = std::chrono::steady_clock::now();
            // "Consecutive" should mean consecutive. If the loop ran clean for a
            // while before this fault, start a fresh streak instead of carrying
            // stale counts from an unrelated incident half an hour ago.
            if (have_prior_fault && (now - last_fault_time) > kFaultStreakWindow)
                fault_count = 0;
            last_fault_time  = now;
            have_prior_fault = true;

            ++fault_count;
            fault_streak_.store(static_cast<uint32_t>(fault_count), std::memory_order_relaxed);
            std::cout << "[WARN] " << name_ << ": franka::ControlException (#" << fault_count
                      << "/" << kMaxConsecutiveFaults << "): " << e.what() << std::endl;

            // Tell the operator NOW, on the first fault, not only once the
            // streak threshold is crossed.
            //
            // Previously the only fault ever transmitted came from
            // enterFaultAndWaitForReset(), i.e. after kMaxConsecutiveFaults.
            // automaticErrorRecovery() on real hardware takes several hundred
            // milliseconds, so a single fault meant roughly a second of a
            // motionless robot with the interface showing a fully healthy
            // link and an ENGAGED remote state. On 2026-08-09 the first fault
            // was at t=404.717 s and the arm never moved again, yet the
            // operator kept commanding until 408.139 s.
            //
            // RECOVERING (not FAULT) is deliberate: this is transient and
            // self-clearing, and it must not latch the interface into the
            // operator-reset path that FAULT triggers. It restores itself
            // below once control() is successfully re-entered.
            if (transmission_) transmission_->setState(SysState::RECOVERING, FaultCode::INTERNAL_ERROR);

            try {
                robot->automaticErrorRecovery();
            } catch (const franka::Exception& recovery_err) {
                std::cout << "[WARN] " << name_ << ": automaticErrorRecovery() failed: "
                          << recovery_err.what() << std::endl;
            }
            // Was '>', which is why the log showed a fourth attempt numbered "#4/3".
            if (fault_count >= kMaxConsecutiveFaults) {
                enterFaultAndWaitForReset();
                fault_count      = 0;
                fault_streak_.store(0, std::memory_order_relaxed);
                have_prior_fault = false;
            } else {
                // Recovered within the streak budget: clear the transient
                // RECOVERING published above so the interface stops warning,
                // then fall through and re-enter control().
                if (transmission_) transmission_->setState(state_);
            }
        } catch (const franka::Exception& e) {
            // Non-control franka errors (e.g. connection-level) aren't something a
            // retry loop can paper over - surface as FAULT and wait for the operator
            // rather than spinning or terminating the process.
            std::cout << "[ERROR] " << name_ << ": franka::Exception: " << e.what() << std::endl;
            enterFaultAndWaitForReset();
            fault_count      = 0;
            have_prior_fault = false;
        }
    }
}


Vector7 ArmControl::jointImpedanceControl(const franka::RobotState& rs) {
    Eigen::Map<const Vector7> q(rs.q.data());
    Eigen::Map<const Vector7> dq(rs.dq.data());


    Vector7 q_target;
    Vector7 dq_ff = Vector7::Zero();   // velocity feedforward
    if (control_mode_ == ControlMode::JOINT_IK &&
        (state_ == SysState::ENGAGED || state_ == SysState::AWAITING)) {
        q_target = motion_gen_.getJointReference();
        dq_ff = motion_gen_.getVelocityReference();
    } else {
        q_target = motion_gen_.getCurrentJoint();
    }

    Vector7 e  = q_target - q;
    Vector7 de = dq_ff - dq;   // velocity error: feedforward ref minus actual

    auto coriolis_array = model->coriolis(rs);
    Vector7 tau_coriolis = Eigen::Map<Vector7>(coriolis_array.data());

    // IDLE holds with reduced stiffness; every other state uses tracking gains.
    const bool idle_hold  = (state_ == SysState::IDLE);
    const Vector7& kp_sel = idle_hold ? kp_idle_  : kp_joint_;
    const Vector7& kd_sel = idle_hold ? kd_idle_  : kd_joint_;

    Vector7 tau = kp_sel.cwiseProduct(e) + kd_sel.cwiseProduct(de) + tau_coriolis;

    return tau;
}


Vector7 ArmControl::cartesianImpedanceControl(const franka::RobotState& rs) {
    Eigen::Map<const Vector7> q(rs.q.data());
    Eigen::Map<const Vector7> dq(rs.dq.data());

    Eigen::Isometry3d T_ee(Eigen::Map<const Eigen::Matrix4d>(rs.O_T_EE.data()));
    Eigen::Isometry3d T_ee_target = motion_gen_.getCurrentCartesian();

    Eigen::Vector3d pos_error = T_ee_target.translation() - T_ee.translation();

    Eigen::Quaterniond q_target(T_ee_target.rotation());
    Eigen::Quaterniond q_current(T_ee.rotation());
    if (q_target.dot(q_current) < 0.0) q_target.coeffs() *= -1.0;
    // Full rotation vector (axis * angle), not vec(q_err).
    //
    // vec(q_err) = n*sin(theta/2) ~ n*theta/2 for small theta -- HALF the
    // rotation vector -- while the damping term below uses the true angular
    // velocity from J*dq. The two halves of the impedance were in different
    // units, so the effective rotational stiffness was kp_cart/2 and the
    // steady-state lag was 2*kd/kp = 120 ms, not the 60 ms the raw ratio
    // suggests. Measured orientation lag in logs/079 and logs/080 was 158 and
    // 168 ms, the worst axis in the system by a wide margin.
    //
    // vec() also saturates at theta = 180 deg and reverses past it, so restoring
    // torque collapses exactly where it is needed most; axis*angle does not.
    //
    // The rotational kp_cart entries are HALVED in config alongside this change,
    // so closed-loop behaviour is unchanged on day one. This commit makes the
    // gain mean what it says; raising it is a separate, deliberate step.
    Eigen::Quaterniond q_error = (q_target * q_current.inverse()).normalized();
    Eigen::AngleAxisd  aa_error(q_error);
    Eigen::Vector3d    ori_error = aa_error.axis() * aa_error.angle();

    Eigen::Matrix<double, 6, 1> error;
    error << pos_error, ori_error;

    auto J_array = model->zeroJacobian(franka::Frame::kEndEffector, rs);
    Matrix6x7 J = Eigen::Map<Matrix6x7>(J_array.data());

    Eigen::Matrix<double, 6, 1> ee_vel = J * dq;

    Eigen::Matrix<double, 6, 1> F = kp_cart_.cwiseProduct(error) - kd_cart_.cwiseProduct(ee_vel)
                                  + feedforwardWrench(motion_gen_.getCurrentCartesianVelocity());
    Vector7 tau_task = J.transpose() * F;

    auto mass_array = model->mass(rs);
    Matrix7 M = Eigen::Map<Matrix7>(mass_array.data());

    auto coriolis_array = model->coriolis(rs);
    Vector7 tau_coriolis = Eigen::Map<Vector7>(coriolis_array.data());

    Eigen::LDLT<Matrix7> M_ldlt(M);
    Eigen::Matrix<double, 7, 7> M_inv = M_ldlt.solve(Matrix7::Identity());
    Eigen::Matrix<double, 6, 6> JMinvJt = J * M_inv * J.transpose();

    double lambda_sq = 0.01;
    Eigen::Matrix<double, 6, 6> JMinvJt_damped = JMinvJt + lambda_sq * Eigen::Matrix<double, 6, 6>::Identity();
    Eigen::Matrix<double, 7, 6> J_pinv = M_inv * J.transpose() * JMinvJt_damped.ldlt().solve(Eigen::Matrix<double, 6, 6>::Identity());
    Eigen::Matrix<double, 7, 7> N = Matrix7::Identity() - J_pinv * J;
    Vector7 tau_null = N * (kp_null_.cwiseProduct(q0_ - q) - kd_null_.cwiseProduct(dq));

    static const Vector7 kMaxDq = (Vector7() << 2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610).finished();
    static const double kVelDampOnset = 0.05;
    Vector7 tau_vel_damp = Vector7::Zero();
    for (int i = 0; i < 7; ++i) {
        double excess = std::abs(dq(i)) - (kMaxDq(i) - kVelDampOnset);
        if (excess > 0.0)
            tau_vel_damp(i) = -80.0 * excess * (dq(i) > 0 ? 1.0 : -1.0);
    }
    return tau_task + tau_null + tau_coriolis + jointLimitAvoidanceTorque(q, dq) + tau_vel_damp;
}

Eigen::Matrix<double, 6, 1> ArmControl::feedforwardWrench(
        const Eigen::Matrix<double, 6, 1>& v_ref) const {
    Eigen::Matrix<double, 6, 1> F = Eigen::Matrix<double, 6, 1>::Zero();
    F.head<3>() = eta_lin_ * kd_cart_.head<3>().cwiseProduct(v_ref.head<3>());
    F.tail<3>() = eta_rot_ * kd_cart_.tail<3>().cwiseProduct(v_ref.tail<3>());

    // Norm-clamped per block so the direction survives. Without this the
    // rotational term reaches 0.9 * 15 * 4 = 54 Nm at the configured angular
    // rate limit, against a 12 Nm wrist joint limit.
    const double f = F.head<3>().norm();
    if (f > ff_force_max_)  F.head<3>() *= ff_force_max_ / f;
    const double m = F.tail<3>().norm();
    if (m > ff_torque_max_) F.tail<3>() *= ff_torque_max_ / m;
    return F;
}

bool ArmControl::isHome() {
    Vector7 q, dq;
    Eigen::Isometry3d T_ee;
    {
        std::lock_guard<std::mutex> lock(state_mtx);
        q    = Eigen::Map<const Vector7>(current_state.q.data());
        dq   = Eigen::Map<const Vector7>(current_state.dq.data());
        T_ee = Eigen::Isometry3d(Eigen::Map<const Eigen::Matrix4d>(current_state.O_T_EE.data()));
    }

    bool position_reached = (q0_ - q).cwiseAbs().maxCoeff() < 0.1;
    bool velocity_settled = dq.cwiseAbs().maxCoeff() < 0.01;

    if (position_reached && velocity_settled) {
        T_origin_ = T_ee;
        return true;
    }
    return false;
}


Eigen::Isometry3d ArmControl::transformCommandToBase(const Eigen::Isometry3d& T_cmd_world) const {
    Eigen::Matrix3d R_w2b = T_base_.rotation().transpose();

    Eigen::Isometry3d T_target = Eigen::Isometry3d::Identity();

    T_target.translation() = T_origin_.translation() + R_w2b * T_cmd_world.translation();

    const Eigen::Matrix3d& M = R_ctrl_to_ee_;
    T_target.linear() = T_origin_.rotation() * (M * T_cmd_world.rotation() * M.transpose());

    return T_target;
}

Eigen::Isometry3d ArmControl::worldAbsoluteToBase(const Eigen::Isometry3d& T_world_abs) const {
    // Exact inverse of the T_base_ * T_local composition used everywhere else
    // for state/logging (see O_T_EE_world in the ArmLogEntry write site) --
    // no T_origin_/controller-remap involved, since this path is for an
    // absolute target, not a delta from wherever homing last landed.
    return T_base_.inverse() * T_world_abs;
}

Eigen::Isometry3d ArmControl::transformBaseToWorld(const Eigen::Isometry3d& T_base) const {
    Eigen::Matrix3d R_b2w = T_base_.rotation();

    Eigen::Isometry3d T_world = Eigen::Isometry3d::Identity();
    T_world.translation() = R_b2w * (T_base.translation() - T_origin_.translation());
    T_world.linear() = T_origin_.rotation().transpose() * T_base.rotation();

    return T_world;
}

Eigen::Isometry3d ArmControl::getTargetPose() const{
    return target_pose_;
}

Eigen::Isometry3d ArmControl::getRawTargetPose() const{
    return target_pose_raw_;
}

void ArmControl::getJointState(Vector7& q, Vector7& dq) const {
    std::lock_guard<std::mutex> lock(state_mtx);
    q  = Eigen::Map<const Vector7>(current_state.q.data());
    dq = Eigen::Map<const Vector7>(current_state.dq.data());
}

void ArmControl::applySelfCollisionFilter(Eigen::Isometry3d& T_target) {
    if (!scp_ || !scp_->config().enabled) return;

    Eigen::Isometry3d T_ee;
    Eigen::Vector3d ee_vel_base = Eigen::Vector3d::Zero();
    {
        std::lock_guard<std::mutex> lock(state_mtx);
        T_ee = Eigen::Isometry3d(Eigen::Map<const Eigen::Matrix4d>(current_state.O_T_EE.data()));
        auto J_array = model->zeroJacobian(franka::Frame::kEndEffector, current_state);
        Matrix6x7 J = Eigen::Map<Matrix6x7>(J_array.data());
        Eigen::Map<const Vector7> dq(current_state.dq.data());
        ee_vel_base = (J * dq).head<3>();
    }

    const Eigen::Matrix3d& R_b2w = T_base_.rotation();
    Eigen::Vector3d ee_pos_world = R_b2w * T_ee.translation() + T_base_.translation();
    Eigen::Vector3d ee_vel_world = R_b2w * ee_vel_base;
    Eigen::Vector3d target_pos_world = R_b2w * T_target.translation() + T_base_.translation();

    Eigen::Vector3d displacement = target_pos_world - ee_pos_world;
    double dist = displacement.norm();

    Eigen::Vector3d nominal_vel;
    if (dist < 1e-6) {
        nominal_vel = Eigen::Vector3d::Zero();
    } else {
        constexpr double vel_gain = 5.0;
        double desired_speed = std::min(vel_gain * dist, scp_->config().max_velocity);
        nominal_vel = (displacement / dist) * desired_speed;
    }

    CollisionState own_state;
    own_state.ee_position = ee_pos_world;
    own_state.ee_velocity = ee_vel_world;
    own_state.weight      = scp_state_.weight;

    constexpr double dt = 1.0 / 100.0;
    CorrectionResult cr = scp_->computeCorrection(nominal_vel, own_state, dt);

    if (cr.active) {
        Eigen::Vector3d safe_vel = nominal_vel + cr.velocity_correction;
        double safe_norm = safe_vel.norm();
        if (safe_norm > scp_->config().max_velocity) {
            safe_vel *= scp_->config().max_velocity / safe_norm;
        }

        Eigen::Vector3d safe_target_world = ee_pos_world + safe_vel * dt;
        T_target.translation() = R_b2w.transpose() * (safe_target_world - T_base_.translation());
    }
}

void ArmControl::validateTargetPose(Eigen::Isometry3d& T_target) {
    Eigen::Vector3d p_target = T_target.translation();
    Eigen::Quaterniond q_target(T_target.rotation());

    if (!p_target.allFinite() || !q_target.coeffs().allFinite()) {
        std::cout << "[WARN] " << name_ << ": non-finite command - discarding packet.\n";
        T_target.translation() = has_prev_valid_target_ ? prev_valid_target_pos_ : motion_gen_.getCurrentCartesian().translation();
        T_target.linear() = has_prev_valid_target_ ? prev_valid_target_rot_.toRotationMatrix() : motion_gen_.getCurrentCartesian().rotation();
        return;
    }

    // Elapsed time since the last accepted command, MEASURED rather than assumed
    // from transmission.frequency.
    //
    // This used cmd_dt_ = 1/transmission.frequency while the function itself runs
    // once per state-thread tick. Those are the same quantity only when the two
    // rates match. Raising comms 200 -> 500 Hz shrank cmd_dt_ to 2 ms while the
    // tick stayed at 5 ms, so the bound became 2 mm per 5 ms = 0.4 m/s against a
    // configured 1.0 m/s: the guard silently got 2.5x more aggressive purely from
    // a comms change, which is the opposite of what raising the command rate is
    // for. Measuring the interval makes this a true velocity limit at any command
    // rate, and robust to dropped packets and jitter.
    const auto cmd_now = std::chrono::steady_clock::now();
    double cmd_dt = cmd_dt_;   // nominal, for the very first command
    if (has_prev_valid_target_) {
        cmd_dt = std::chrono::duration<double>(cmd_now - prev_valid_target_time_).count();
        // Clamp low against a zero or negative interval (clock jitter, two
        // commands inside one tick), and high so a long gap -- re-engage, a
        // stalled sender, an operator who stopped moving -- cannot hand out an
        // effectively unbounded jump.
        cmd_dt = std::clamp(cmd_dt, kMinCmdDt, kMaxCmdDt);
        last_cmd_dt_ = cmd_dt;
    }

    if (has_prev_valid_target_) {
        Eigen::Vector3d dp = p_target - prev_valid_target_pos_;
        double jump_norm   = dp.norm();
        double max_step    = max_command_velocity_ * cmd_dt;

        if (jump_norm > max_step && jump_norm > 1e-9)
            p_target = prev_valid_target_pos_ + (max_step / jump_norm) * dp;

        if (q_target.dot(prev_valid_target_rot_) < 0.0)
            q_target.coeffs() *= -1.0;

        double angle     = prev_valid_target_rot_.angularDistance(q_target);
        double max_angle = max_command_angular_velocity_ * cmd_dt;

        if (angle > max_angle && angle > 1e-9)
            q_target = prev_valid_target_rot_.slerp(max_angle / angle, q_target);
    }

    // Leash the target to the MEASURED pose.
    //
    // Everything above bounds target-against-target: how fast the setpoint may
    // move. None of it looks at where the robot actually is, so a command far
    // enough away is not rejected -- it is walked toward at max_command_velocity
    // for as long as it takes, and the impedance spring stretches the whole way.
    // On 2026-09-16 a 143 mm command glitch became 150 ms at 1 m/s, 113 mm of
    // error, ~113 N at kp_cart 1000, and both arms latched FAULT. The rate limit
    // did not prevent that; it was the mechanism.
    //
    // Capping the lead converts an unreachable command into bounded force: the
    // target sits max_target_lead_ ahead, pulls with kp_cart * lead, and advances
    // only as the robot advances. The operator sees lag instead of a fault, and
    // the arm still gets there.
    if (max_target_lead_ > 0.0) {
        Eigen::Vector3d ee_pos;
        {
            std::lock_guard<std::mutex> lock(state_mtx);
            ee_pos = Eigen::Isometry3d(
                Eigen::Map<const Eigen::Matrix4d>(current_state.O_T_EE.data())).translation();
        }
        // Zero before the first state arrives -- leashing to the base frame would
        // yank the target to the robot's origin.
        if (ee_pos.norm() > 1e-6) {
            Eigen::Vector3d lead = p_target - ee_pos;
            double lead_norm = lead.norm();
            if (lead_norm > max_target_lead_ && lead_norm > 1e-9) {
                p_target = ee_pos + (max_target_lead_ / lead_norm) * lead;
                if (cmd_now - last_leash_log_time_ > std::chrono::seconds(1)) {
                    last_leash_log_time_ = cmd_now;
                    std::cout << "[WARN] " << name_ << ": target leashed - operator "
                              << lead_norm * 1000.0 << " mm ahead of the arm, capped at "
                              << max_target_lead_ * 1000.0 << " mm.\n";
                }
            }
        }
    }

    Eigen::Vector3d ee_z_world = (T_base_.rotation() * q_target.toRotationMatrix()).col(2);
    double tilt = std::acos(std::clamp(-ee_z_world.z(), -1.0, 1.0));
    if (tilt > max_tilt_angle_) {
        Eigen::Vector3d axis = ee_z_world.cross(Eigen::Vector3d(0.0, 0.0, -1.0));
        double axis_norm = axis.norm();
        if (axis_norm > 1e-9) {
            Eigen::Quaterniond q_correction(Eigen::AngleAxisd(tilt - max_tilt_angle_, axis / axis_norm));
            q_target = (Eigen::Quaterniond(T_base_.rotation()).inverse() * q_correction * Eigen::Quaterniond(T_base_.rotation()) * q_target).normalized();
        }
    }

    Eigen::Vector3d p_world = T_base_.rotation() * p_target + T_base_.translation();
    p_world = p_world.cwiseMax(workspace_min_).cwiseMin(workspace_max_);
    double min_world_z = table_height_world_ + table_safety_margin_ + ee_fingertip_length_;
    if (p_world.z() < min_world_z)
        p_world.z() = min_world_z;
    p_target = T_base_.rotation().transpose() * (p_world - T_base_.translation());

    T_target.translation() = p_target;
    T_target.linear()      = q_target.toRotationMatrix();
    prev_valid_target_pos_  = p_target;
    prev_valid_target_rot_  = q_target;
    prev_valid_target_time_ = cmd_now;
    has_prev_valid_target_  = true;
}

void ArmControl::reOrigin() {
    std::lock_guard<std::mutex> lock(state_mtx);
    T_origin_ = Eigen::Isometry3d(Eigen::Map<const Eigen::Matrix4d>(current_state.O_T_EE.data()));
}

void ArmControl::applyGripper(bool close) {
    if (!gripper) { gripper_close_applied_ = close; return; }
    if (close == gripper_close_applied_) return;
#ifdef WITH_FRANKA
    if (gripper_busy_.exchange(true)) return;
    const double width = close ? 0.0 : kGripperMaxWidth;
    std::thread([this, close, width]() {
        try {
            if (close) gripper->grasp(width, 0.1, 40.0);
            else       gripper->move(width, 0.1);
        } catch (...) {}
        gripper_busy_.store(false);
    }).detach();
    gripper_close_applied_ = close;
#else
    gripper->setWidth(close ? 0.0 : kGripperMaxWidth);
    gripper_close_applied_ = close;
#endif
}

void ArmControl::updateGraspConfirmation(double width) {
    const bool commanding_close = grasp_allowed_.load() && desired_gripper_closed_.load();
    const auto now = std::chrono::steady_clock::now();

    if (!commanding_close) {
        grasp_track_active_ = false;
        grasp_state_.store(GraspState::OPEN);
        return;
    }

    if (grasp_state_.load() == GraspState::LOST) {
        if (now < grasp_lost_latch_until_) return;
        grasp_state_.store(GraspState::OPEN);
    }

    const bool near_open   = width > (kGripperMaxWidth - grasp_confirm_tolerance_m_);
    const bool near_closed = width < grasp_confirm_tolerance_m_;
    if (near_open || near_closed) {
        if (grasp_state_.load() == GraspState::HELD) {
            grasp_state_.store(GraspState::LOST);
            grasp_lost_latch_until_ = now + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(grasp_lost_latch_s_));
        } else {
            grasp_state_.store(GraspState::OPEN);
        }
        grasp_track_active_ = false;
        return;
    }

    if (grasp_state_.load() == GraspState::HELD) return;

    if (!grasp_track_active_ || std::abs(width - grasp_track_width_) > grasp_confirm_tolerance_m_) {
        grasp_track_active_ = true;
        grasp_track_width_  = width;
        grasp_track_start_  = now;
        return;
    }

    if (std::chrono::duration<double>(now - grasp_track_start_).count() >= grasp_confirm_time_s_)
        grasp_state_.store(GraspState::HELD);
}

void ArmControl::restartLogger(const std::string& path) {
    logger_->restart(path);
}

void ArmControl::writeEpisodeConfig(int seed, int mode, const std::string& color_bin_mapping) {
    logger_->writeEpisodeConfig(seed, mode, color_bin_mapping);
}

Vector7 ArmControl::jointLimitAvoidanceTorque(const Vector7& q, const Vector7& dq) {
    Vector7 tau = Vector7::Zero();

    for (int i = 0; i < 7; ++i) {
        const double qmin = q_min_(i);
        const double qmax = q_max_(i);
        const double dmin = q(i) - qmin;
        const double dmax = qmax - q(i);

        double tau_i = 0.0;

        if (dmin < joint_limit_buffer_) {
            double w = std::clamp((joint_limit_buffer_ - dmin) / joint_limit_buffer_, 0.0, 1.0);
            w *= w;
            tau_i += kp_joint_limit_(i) * w * ((qmin + joint_limit_buffer_) - q(i));
            if (dq(i) < 0.0)
                tau_i -= kd_joint_limit_(i) * w * dq(i);
        }

        if (dmax < joint_limit_buffer_) {
            double w = std::clamp((joint_limit_buffer_ - dmax) / joint_limit_buffer_, 0.0, 1.0);
            w *= w;
            tau_i += kp_joint_limit_(i) * w * ((qmax - joint_limit_buffer_) - q(i));
            if (dq(i) > 0.0)
                tau_i -= kd_joint_limit_(i) * w * dq(i);
        }

        const double sat = joint_limit_torque_frac_ * tau_max_(i);
        tau(i) = std::clamp(tau_i, -sat, sat);
    }

    return tau;
}
