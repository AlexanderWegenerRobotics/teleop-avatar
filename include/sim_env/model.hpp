#pragma once

#include <array>
#include <string>

#include <Eigen/Dense>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include "common.hpp"
#include <mutex>

namespace franka {

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

    std::array<double, 42> zeroJacobian(const std::array<double, 7>& q);
    std::array<double, 49> mass(const std::array<double, 7>& q);
    std::array<double, 7>  coriolis(const std::array<double, 7>& q, const std::array<double, 7>& dq);
    std::array<double, 7>  gravity(const std::array<double, 7>& q);
    std::array<double, 16> EEPose(const std::array<double, 7>& q);
    std::array<double, 6>  cartesianWrench(const std::array<double, 7>& q, const std::array<double, 7>& tau_ext);
    GMOInputs computeGMOInputs(const std::array<double, 7>& q, const std::array<double, 7>& dq);
    Matrix4 forwardKinematics(const Vector7& q);

private:
    pinocchio::Model pin_model_;
    pinocchio::Data  pin_data_;
    std::string ee_frame_name_;
    mutable std::mutex pin_mutex_;
};

}