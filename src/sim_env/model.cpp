#include "sim_env/model.hpp"

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/model.hpp>

using namespace franka;

Model::Model(const std::string& urdf_path, const std::array<double, 4>& base_quat, const std::string& ee_frame_name)
    : ee_frame_name_(ee_frame_name)
{
    pinocchio::Model full_model;
    pinocchio::urdf::buildModel(urdf_path, full_model);

    std::vector<pinocchio::JointIndex> joints_to_lock;
    for (const std::string& name : {"fr3_finger_joint1", "fr3_finger_joint2"}) {
        if (full_model.existJointName(name))
            joints_to_lock.push_back(full_model.getJointId(name));
    }

    for (int i = 0; i <= 15; ++i) {
        std::string name = "joint_" + std::to_string(i) + "_0";
        if (full_model.existJointName(name))
            joints_to_lock.push_back(full_model.getJointId(name));
    }

    Eigen::VectorXd q_ref = Eigen::VectorXd::Zero(full_model.nq);
    pin_model_ = pinocchio::buildReducedModel(full_model, joints_to_lock, q_ref);

    if (!pin_model_.existFrame(ee_frame_name_))
        std::cerr << "[Model] WARNING: EE frame '" << ee_frame_name_ << "' not found in URDF!\n";

    std::cout << "[Model] Looking for EE frame: " << ee_frame_name_ << std::endl;
    std::cout << "[Model] Frame exists: " << pin_model_.existFrame(ee_frame_name_) << std::endl;
    std::cout << "[Model] nq=" << pin_model_.nq << " nv=" << pin_model_.nv << std::endl;

    Eigen::Quaterniond q(base_quat[0], base_quat[1], base_quat[2], base_quat[3]);
    q.normalize();
    Eigen::Vector3d g_base = q.toRotationMatrix().transpose() * Eigen::Vector3d(0, 0, -9.81);
    pin_model_.gravity.linear(g_base);
    pin_data_ = pinocchio::Data(pin_model_);
}

Model::~Model() {}

std::array<double, 42> Model::zeroJacobian(const std::array<double, 7>& q) {
    Vector7 q_eig = Eigen::Map<const Vector7>(q.data());
    std::lock_guard<std::mutex> lock(pin_mutex_);
    pinocchio::computeJointJacobians(pin_model_, pin_data_, q_eig);
    pinocchio::framesForwardKinematics(pin_model_, pin_data_, q_eig);

    Matrix6x7 J = Matrix6x7::Zero();
    pinocchio::getFrameJacobian(pin_model_, pin_data_,
                                pin_model_.getFrameId(ee_frame_name_),
                                pinocchio::LOCAL_WORLD_ALIGNED, J);

    std::array<double, 42> result;
    Eigen::Map<Matrix6x7>(result.data()) = J;
    return result;
}

std::array<double, 49> Model::mass(const std::array<double, 7>& q) {
    Vector7 q_eig = Eigen::Map<const Vector7>(q.data());
    std::lock_guard<std::mutex> lock(pin_mutex_);
    pinocchio::crba(pin_model_, pin_data_, q_eig);
    pin_data_.M.triangularView<Eigen::StrictlyLower>() =
        pin_data_.M.triangularView<Eigen::StrictlyUpper>().transpose();

    std::array<double, 49> result;
    Eigen::Map<Matrix7>(result.data()) = pin_data_.M;
    return result;
}

std::array<double, 7> Model::coriolis(const std::array<double, 7>& q, const std::array<double, 7>& dq) {
    Vector7 q_eig  = Eigen::Map<const Vector7>(q.data());
    Vector7 dq_eig = Eigen::Map<const Vector7>(dq.data());

    std::lock_guard<std::mutex> lock(pin_mutex_);
    pinocchio::computeCoriolisMatrix(pin_model_, pin_data_, q_eig, dq_eig);

    std::array<double, 7> result;
    Eigen::Map<Vector7>(result.data()) = pin_data_.C * dq_eig;
    return result;
}

std::array<double, 7> Model::gravity(const std::array<double, 7>& q) {
    Vector7 q_eig = Eigen::Map<const Vector7>(q.data());

    std::lock_guard<std::mutex> lock(pin_mutex_);
    pinocchio::computeGeneralizedGravity(pin_model_, pin_data_, q_eig);

    std::array<double, 7> result;
    Eigen::Map<Vector7>(result.data()) = pin_data_.g;
    return result;
}

std::array<double, 16> Model::EEPose(const std::array<double, 7>& q) {
    Vector7 q_eig = Eigen::Map<const Vector7>(q.data());

    std::lock_guard<std::mutex> lock(pin_mutex_);
    pinocchio::forwardKinematics(pin_model_, pin_data_, q_eig);
    pinocchio::updateFramePlacements(pin_model_, pin_data_);

    const pinocchio::SE3& T = pin_data_.oMf[pin_model_.getFrameId(ee_frame_name_)];

    std::array<double, 16> result;
    Eigen::Map<Eigen::Matrix4d>(result.data()) = T.toHomogeneousMatrix();
    return result;
}

std::array<double, 6> Model::cartesianWrench(const std::array<double, 7>& q,
                                            const std::array<double, 7>& tau_ext) {
    Vector7 q_eig       = Eigen::Map<const Vector7>(q.data());
    Vector7 tau_ext_eig = Eigen::Map<const Vector7>(tau_ext.data());
    std::lock_guard<std::mutex> lock(pin_mutex_);
    pinocchio::computeJointJacobians(pin_model_, pin_data_, q_eig);
    pinocchio::framesForwardKinematics(pin_model_, pin_data_, q_eig);

    Matrix6x7 J = Matrix6x7::Zero();
    pinocchio::getFrameJacobian(pin_model_, pin_data_,
                                pin_model_.getFrameId(ee_frame_name_),
                                pinocchio::LOCAL_WORLD_ALIGNED, J);

    Eigen::Matrix<double, 6, 1> F_ext = (J * J.transpose()).ldlt().solve(J * tau_ext_eig);

    std::array<double, 6> result;
    Eigen::Map<Eigen::Matrix<double, 6, 1>>(result.data()) = F_ext;
    return result;
}

GMOInputs Model::computeGMOInputs(const std::array<double, 7>& q, const std::array<double, 7>& dq) {
    Vector7 q_eig  = Eigen::Map<const Vector7>(q.data());
    Vector7 dq_eig = Eigen::Map<const Vector7>(dq.data());

    std::lock_guard<std::mutex> lock(pin_mutex_);
    pinocchio::crba(pin_model_, pin_data_, q_eig);
    pin_data_.M.triangularView<Eigen::StrictlyLower>() = pin_data_.M.triangularView<Eigen::StrictlyUpper>().transpose();

    pinocchio::computeCoriolisMatrix(pin_model_, pin_data_, q_eig, dq_eig);
    pinocchio::computeGeneralizedGravity(pin_model_, pin_data_, q_eig);

    return { pin_data_.M * dq_eig, pin_data_.C * dq_eig + pin_data_.g };
}

Matrix4 Model::forwardKinematics(const Vector7& q) {
    std::lock_guard<std::mutex> lock(pin_mutex_);
    pinocchio::forwardKinematics(pin_model_, pin_data_, q);
    pinocchio::updateFramePlacements(pin_model_, pin_data_);

    const pinocchio::SE3& T = pin_data_.oMf[pin_model_.getFrameId(ee_frame_name_)];
    return T.toHomogeneousMatrix();
}