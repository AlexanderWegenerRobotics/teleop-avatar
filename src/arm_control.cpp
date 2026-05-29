#include "arm_control.hpp"
#include "common.hpp"
#include "interpolator.hpp"
#include "self_collision_protection.hpp"
#include "sim_env/gripper.hpp"
#include "sim_env/model.hpp"
#include <chrono>
#include <iostream>

ArmControl::ArmControl(const YAML::Node& device_config, const std::string& session_id)
#ifdef WITH_FRANKA
    : robot(std::make_unique<franka::Robot>(device_config["franka_ip"].as<std::string>()))
    , gripper(std::make_unique<franka::Gripper>(device_config["franka_ip"].as<std::string>()))
#else
    : robot(std::make_unique<franka::Robot>())
    , gripper(std::make_unique<franka::Gripper>())
#endif
    , bRunning(false)
    , state_(SysState::OFFLINE)
    , interpolator_(InterpolatorConfig{
        .control_freq   = 1000,
        .comm_freq      = device_config["transmission"]["frequency"].as<int>(),
        .n_dof          = 7,
        .max_linear_vel = 0.5,
        .max_angular_vel = 0.8
    })
    , recovery_(device_config["name"].as<std::string>())
{
    name_ = device_config["name"].as<std::string>();
    
    auto pos  = device_config["base_pose"]["position"].as<std::vector<double>>();
    auto ori  = device_config["base_pose"]["orientation"].as<std::vector<double>>();

    base_position_ = Eigen::Vector3d(pos[0], pos[1], pos[2]);
    base_orientation_ = Eigen::Quaterniond(ori[0], ori[1], ori[2], ori[3]);

    if (name_.find("right") != std::string::npos) {
        R_tool <<  0, -1,  0,
                   0,  0,  1,
                  -1,  0,  0;
    } else {
        R_tool <<  0, -1,  0,
                0,  0,  1,
                -1,  0,  0;
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
    kp_joint_ = yamlToVector<7>(device_config["control"]["kp_joint"]);
    kd_joint_ = yamlToVector<7>(device_config["control"]["kd_joint"]);
    kp_cart_ = yamlToVector<6>(device_config["control"]["kp_cart"]);
    kd_cart_ = yamlToVector<6>(device_config["control"]["kd_cart"]);
    kp_null_ = yamlToVector<7>(device_config["control"]["kp_null"]);
    kd_null_ = yamlToVector<7>(device_config["control"]["kd_null"]);
    kd_joint_limit_ = yamlToVector<7>(device_config["control"]["kd_joint_limit"]);
    kp_joint_limit_ = yamlToVector<7>(device_config["control"]["kp_joint_limit"]);
    joint_limit_buffer_  = device_config["safety"]["joint_limit_buffer"].as<double>();
    joint_limit_torque_frac_ = device_config["safety"]["joint_limit_torque_frac"].as<double>();

    if (device_config["transmission"]) {
        UdpStreamConfig stream_cfg;
        stream_cfg.transport.remote_ip   = device_config["transmission"]["remote_ip"].as<std::string>();
        stream_cfg.transport.remote_port = device_config["transmission"]["send_port"].as<int>();
        stream_cfg.transport.bind_port   = device_config["transmission"]["receive_port"].as<int>();
        stream_cfg.send_rate_hz          = device_config["transmission"]["frequency"].as<int>();
        transmission_ = std::make_unique<ArmStream>(stream_cfg);
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
    ee_fingertip_length_ = device_config["safety"]["ee_fingertip_length"].as<double>();
    max_tilt_angle_ = device_config["safety"]["max_tilt_angle"].as<double>();
    cmd_dt_ = 1.0 / static_cast<double>(device_config["transmission"]["frequency"].as<int>());

    logger_ = std::make_unique<DataLogger<ArmLogEntry>>("../log/" + name_ + "_log.csv", armLogHeader, armLogRow, session_id);

}

ArmControl::~ArmControl(){
    stop();
}

void ArmControl::start(){
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
    interpolator_.planJoint(q_init, q_init, ProfileType::TRAPEZOIDAL);
    control_thread = std::thread(&ArmControl::runControlHandler, this);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    state_thread = std::thread(&ArmControl::runStateHandler, this);
    if (transmission_) transmission_->start();
    logger_->start();
    logger_->enable(true);
    startTime_ = std::chrono::high_resolution_clock::now();
}

void ArmControl::stop(){
    if (logger_) logger_->stop();
    bRunning = false;
    state_ = SysState::OFFLINE;
    if (control_thread.joinable()) control_thread.join();
    if (state_thread.joinable()) state_thread.join();
    if (transmission_) transmission_->stop();
}

void ArmControl::runStateHandler(){
    constexpr std::chrono::microseconds control_period(static_cast<int>(1e6 / 100));
    auto next_control_time = std::chrono::high_resolution_clock::now();
    SysState prev_state = state_;
    Eigen::VectorXd q_current = Eigen::VectorXd::Zero(7);
    bool has_cmd = false;
    ArmCommandMsg cmd;
    Eigen::Quaterniond prev_cmd_quat_ = Eigen::Quaterniond::Identity();

    while(bRunning){

        if (transmission_ && transmission_->hasNew()) {
            cmd = transmission_->getRecvData();
            has_cmd = true;
        }

        updateRecovery();
        updateStateMachine(cmd_state_);

        if (state_ != SysState::ENGAGED) {
            has_cmd = false;
        }

        if (state_ == SysState::HOMING && prev_state != SysState::HOMING) {
            {
                std::lock_guard<std::mutex> lock(state_mtx);
                q_current = Eigen::Map<const Vector7>(current_state.q.data());
            }
            interpolator_.planJoint(q_current, q0_, ProfileType::MINJERK);
        }
        else if (state_ == SysState::ENGAGED) {
            if (has_cmd) {
                Eigen::Isometry3d T_cmd = Eigen::Isometry3d::Identity();
                Eigen::Vector3d pos(cmd.position[0], cmd.position[1], cmd.position[2]);
                Eigen::Quaterniond q(cmd.quaternion[0], cmd.quaternion[1], cmd.quaternion[2], cmd.quaternion[3]);
                q.normalize();
                if (q.dot(prev_cmd_quat_) < 0.0) q.coeffs() *= -1.0;
                prev_cmd_quat_ = q;
                T_cmd.translation() = pos;
                T_cmd.linear() = q.toRotationMatrix();
                Eigen::Isometry3d T_target = transformCommandToBase(T_cmd);
                target_pose_raw_ = T_base_ * T_target;
                applySelfCollisionFilter(T_target);
                validateTargetPose(T_target);
                interpolator_.planCartesian(interpolator_.getCurrentCartesian(), T_target, ProfileType::LINEAR);
                double target_width = (1.0 - static_cast<double>(cmd.gripper)) * 0.08;
#ifdef WITH_FRANKA
                gripper->move(target_width, 0.08);
#else
                gripper->setWidth(target_width);
#endif
                target_pose_ = T_base_ * T_target;
                has_cmd = false;
            } else {
                Eigen::Isometry3d T_current_target = interpolator_.getCurrentCartesian();
                Eigen::Isometry3d T_filtered = T_current_target;
                applySelfCollisionFilter(T_filtered);

                double pos_change = (T_filtered.translation() - T_current_target.translation()).norm();
                if (pos_change > 1e-6) {
                    interpolator_.planCartesian(interpolator_.getCurrentCartesian(), T_filtered, ProfileType::LINEAR);
                    target_pose_ = T_base_ * T_filtered;
                }
            }
        }

        if (transmission_) {
            franka::RobotState rs;
            {
                std::lock_guard<std::mutex> lock(state_mtx);
                rs = current_state;
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
            transmission_->setSendData(state_msg);
        }

        gripper_width_.store(gripper->readOnce().width);

        prev_state = state_;
        next_control_time += control_period;
        std::this_thread::sleep_until(next_control_time);
    }
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
        recovery_target_q_ = req.target_q;
        interpolator_.planJoint(q_current, req.target_q, ProfileType::MINJERK);
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
            bool trajectory_done = interpolator_.isDone();
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
                interpolator_.planCartesian(T_ee, T_ee);
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
                state_ = SysState::AWAITING;
                interpolator_.planCartesian(T_origin_, T_origin_);
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
    if (state_ != prev && transmission_) {
        transmission_->setState(state_);
    }
}

void ArmControl::runControlHandler(){
    Vector7 tau_prev_ = Vector7::Zero();

    std::function<franka::Torques(const franka::RobotState&, franka::Duration)>
        control_callback = [&](const franka::RobotState& robot_state, franka::Duration) -> franka::Torques {
            
            interpolator_.step();
            {
                std::lock_guard<std::mutex> lock(state_mtx);
                current_state = robot_state;
            }
            Vector7 ctrl_torque = Vector7::Zero();

            switch(state_){
                case SysState::HOMING:
                case SysState::RECOVERING:
                    ctrl_torque = jointImpedanceControl(robot_state);
                    break;

                case SysState::AWAITING:
                case SysState::ENGAGED:
                    ctrl_torque = cartesianImpedanceControl(robot_state);
                    break;

                default:
                    break;
            }

            ctrl_torque = tau_prev_ + (ctrl_torque - tau_prev_).cwiseMax(-tau_rate_max_).cwiseMin(tau_rate_max_);
            ctrl_torque = ctrl_torque.cwiseMax(-tau_max_).cwiseMin(tau_max_);
            tau_prev_ = ctrl_torque;

            if (logger_) {
                double t = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - startTime_).count();
                Vector7 q_target = Vector7::Zero();
                Matrix4 T_target = Matrix4::Identity();
                if(state_ == SysState::HOMING || state_ == SysState::RECOVERING){
                    q_target = interpolator_.getCurrentJoint();
                    //T_target = model->forwardKinematics(q_target);
                }
                else if(state_ == SysState::ENGAGED || state_ == SysState::AWAITING){
                    Eigen::Isometry3d T_ee_target = interpolator_.getCurrentCartesian();
                    T_target = T_ee_target.matrix();
                }

                ArmLogEntry entry{};
                entry.time  = t;
                entry.state = state_;
                entry.gripper_width = gripper_width_.load();
                std::copy(robot_state.q.begin(),                    robot_state.q.end(),                    entry.q.begin());
                Eigen::Map<Vector7>(entry.q_cmd.data()) = q_target;
                std::copy(robot_state.dq.begin(),                   robot_state.dq.end(),                   entry.dq.begin());
                std::copy(robot_state.tau_J.begin(),                robot_state.tau_J.end(),                entry.tau_J.begin());
                std::copy(robot_state.tau_ext_hat_filtered.begin(), robot_state.tau_ext_hat_filtered.end(), entry.tau_ext.begin());
                std::copy(robot_state.O_T_EE.begin(),               robot_state.O_T_EE.end(),               entry.O_T_EE.begin());
                Eigen::Map<Matrix4>(entry.O_T_EE_cmd.data()) = T_target;
                std::copy(robot_state.O_F_ext_hat_K.begin(),        robot_state.O_F_ext_hat_K.end(),        entry.F_ext.begin());
                logger_->write(entry);
            }
            std::array<double, 7> ctrl_array;
            Eigen::Map<Vector7>(ctrl_array.data()) = ctrl_torque;

            franka::Torques tau(ctrl_array);
            tau.motion_finished = !bRunning;
            return tau;
        };

    robot->control(control_callback);
}


Vector7 ArmControl::jointImpedanceControl(const franka::RobotState& rs) {
    Eigen::Map<const Vector7> q(rs.q.data());
    Eigen::Map<const Vector7> dq(rs.dq.data());

    Vector7 q_target = interpolator_.getCurrentJoint();

    Vector7 e = q_target - q;
    Vector7 de = -dq;

    auto coriolis_array = model->coriolis(rs);
    Vector7 tau_coriolis = Eigen::Map<Vector7>(coriolis_array.data());

    Vector7 tau = kp_joint_.cwiseProduct(e) + kd_joint_.cwiseProduct(de) + tau_coriolis;

    return tau;
}


Vector7 ArmControl::cartesianImpedanceControl(const franka::RobotState& rs) {
    Eigen::Map<const Vector7> q(rs.q.data());
    Eigen::Map<const Vector7> dq(rs.dq.data());

    Eigen::Isometry3d T_ee(Eigen::Map<const Eigen::Matrix4d>(rs.O_T_EE.data()));
    Eigen::Isometry3d T_ee_target = interpolator_.getCurrentCartesian();

    Eigen::Vector3d pos_error = T_ee_target.translation() - T_ee.translation();

    Eigen::Quaterniond q_target(T_ee_target.rotation());
    Eigen::Quaterniond q_current(T_ee.rotation());
    if (q_target.dot(q_current) < 0.0) q_target.coeffs() *= -1.0;
    Eigen::Quaterniond q_error = q_target * q_current.inverse();
    Eigen::Vector3d ori_error(q_error.x(), q_error.y(), q_error.z());

    Eigen::Matrix<double, 6, 1> error;
    error << pos_error, ori_error;

    auto J_array = model->zeroJacobian(franka::Frame::kEndEffector, rs);
    Matrix6x7 J = Eigen::Map<Matrix6x7>(J_array.data());

    Eigen::Matrix<double, 6, 1> ee_vel = J * dq;

    Eigen::Matrix<double, 6, 1> F = kp_cart_.cwiseProduct(error) - kd_cart_.cwiseProduct(ee_vel);
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
    //return tau_task + tau_null + tau_coriolis + jointLimitAvoidanceTorque(q, dq);
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
    T_target.linear() = T_origin_.rotation() * R_tool * T_cmd_world.rotation() * R_tool.transpose();

    return T_target;
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
        T_target.translation() = has_prev_valid_target_ ? prev_valid_target_pos_ : interpolator_.getCurrentCartesian().translation();
        T_target.linear() = has_prev_valid_target_ ? prev_valid_target_rot_.toRotationMatrix() : interpolator_.getCurrentCartesian().rotation();
        return;
    }

    if (has_prev_valid_target_) {
        Eigen::Vector3d dp = p_target - prev_valid_target_pos_;
        double jump_norm   = dp.norm();
        double max_step    = max_command_velocity_ * cmd_dt_;

        if (jump_norm > max_step && jump_norm > 1e-9)
            p_target = prev_valid_target_pos_ + (max_step / jump_norm) * dp;

        if (q_target.dot(prev_valid_target_rot_) < 0.0)
            q_target.coeffs() *= -1.0;

        double angle     = prev_valid_target_rot_.angularDistance(q_target);
        double max_angle = max_command_angular_velocity_ * cmd_dt_;

        if (angle > max_angle && angle > 1e-9)
            q_target = prev_valid_target_rot_.slerp(max_angle / angle, q_target);
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
    prev_valid_target_pos_ = p_target;
    prev_valid_target_rot_ = q_target;
    has_prev_valid_target_ = true;
}

void ArmControl::reOrigin() {
    std::lock_guard<std::mutex> lock(state_mtx);
    T_origin_ = Eigen::Isometry3d(Eigen::Map<const Eigen::Matrix4d>(current_state.O_T_EE.data()));
}

void ArmControl::restartLogger(const std::string& path) {
    logger_->restart(path);
}

void ArmControl::writeEpisodeConfig(double px, double py, double pz, double gx, double gy, double gz, int mode) {
    logger_->writeEpisodeConfig(px, py, pz, gx, gy, gz, mode);
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