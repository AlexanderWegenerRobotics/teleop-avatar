#pragma once

#ifdef WITH_FRANKA

#include <franka/model.h>

#else

#include <array>
#include <string>

#include <Eigen/Dense>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include "common.hpp"
#include <mutex>

namespace franka {

enum class Frame {
    kJoint1, kJoint2, kJoint3, kJoint4, kJoint5, kJoint6, kJoint7,
    kFlange, kEndEffector, kStiffness
};

struct RobotState;  // defined in robot.hpp

struct GMOInputs {
    Vector7 p;
    Vector7 tau_model;
};

class Model {
public:
    // joint_damping / joint_coulomb / rotor_inertia must mirror the MJCF's
    // damping / frictionloss / armature for this arm: anything the plant applies
    // and the model omits lands in tau_ext.
    Model(const std::string& urdf_path, const std::array<double, 4>& base_quat, const std::string& ee_frame_name,
          const Vector7& joint_damping, const Vector7& joint_coulomb, const Vector7& rotor_inertia);
    ~Model();

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&) = delete;
    Model& operator=(Model&&) = delete;

    // Signatures match real libfranka RobotState overloads
    std::array<double, 42> zeroJacobian(Frame frame, const RobotState& rs);
    std::array<double, 49> mass(const RobotState& rs);
    std::array<double, 7>  coriolis(const RobotState& rs);

    // Simulation-internal (used by robot.cpp, keep q-based)
    std::array<double, 7>  gravity(const std::array<double, 7>& q);
    std::array<double, 16> EEPose(const std::array<double, 7>& q);
    // Base-frame pose of a robot frame. kJoint1..7 map to the URDF joint frames,
    // kFlange to fr3_link8, kEndEffector/kStiffness to the configured EE frame.
    std::array<double, 16> framePose(Frame frame, const std::array<double, 7>& q);
    std::array<double, 6>  cartesianWrench(const std::array<double, 7>& q, const std::array<double, 7>& tau_ext);
    GMOInputs computeGMOInputs(const std::array<double, 7>& q, const std::array<double, 7>& dq);

private:
    pinocchio::Model pin_model_;
    pinocchio::Data  pin_data_;
    std::string ee_frame_name_;
    Vector7 joint_damping_;   // Nm.s/rad, viscous
    Vector7 joint_coulomb_;   // Nm, dry friction magnitude
    mutable std::mutex pin_mutex_;
};

}  // namespace franka

#endif  // WITH_FRANKA