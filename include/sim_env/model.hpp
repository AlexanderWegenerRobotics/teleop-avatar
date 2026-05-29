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
    Model(const std::string& urdf_path, const std::array<double, 4>& base_quat, const std::string& ee_frame_name);
    ~Model();

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&) = default;
    Model& operator=(Model&&) = default;

    // Signatures match real libfranka RobotState overloads
    std::array<double, 42> zeroJacobian(Frame frame, const RobotState& rs);
    std::array<double, 49> mass(const RobotState& rs);
    std::array<double, 7>  coriolis(const RobotState& rs);

    // Simulation-internal (used by robot.cpp, keep q-based)
    std::array<double, 7>  gravity(const std::array<double, 7>& q);
    std::array<double, 16> EEPose(const std::array<double, 7>& q);
    std::array<double, 6>  cartesianWrench(const std::array<double, 7>& q, const std::array<double, 7>& tau_ext);
    GMOInputs computeGMOInputs(const std::array<double, 7>& q, const std::array<double, 7>& dq);

private:
    pinocchio::Model pin_model_;
    pinocchio::Data  pin_data_;
    std::string ee_frame_name_;
    mutable std::mutex pin_mutex_;
};

}  // namespace franka

#endif  // WITH_FRANKA