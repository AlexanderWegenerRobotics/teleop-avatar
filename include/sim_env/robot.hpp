#pragma once

#include <memory>
#include <array>
#include <string>
#include <functional>
#include <atomic>

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

#include "sim_env/model.hpp"

class Simulation;
struct DeviceState;

namespace franka {

struct RobotState {
    std::array<double, 7>  q;
    std::array<double, 7>  dq;
    std::array<double, 7>  tau_J;
    std::array<double, 7>  tau_J_d;
    std::array<double, 7>  tau_ext_hat_filtered;
    std::array<double, 6>  O_F_ext_hat_K;
    std::array<double, 16> O_T_EE;

    RobotState() {
        q.fill(0.0);                  dq.fill(0.0);
        tau_J.fill(0.0);              tau_J_d.fill(0.0);
        tau_ext_hat_filtered.fill(0.0);
        O_F_ext_hat_K.fill(0.0);     O_T_EE.fill(0.0);
    }
};

struct Finishable {
    bool motion_finished = false;
};

class Torques : public Finishable {
public:
    Torques(const std::array<double, 7>& torques) noexcept;
    Torques(std::initializer_list<double> torques);
    std::array<double, 7> tau_J{};
};

class Duration {
public:
    Duration() {}
    double time = 1.0 / 1000.0;
};

class Robot {
public:
    Robot();
    ~Robot();

    void set_simulation(Simulation& _sim, const YAML::Node& sim_dev, const YAML::Node& robot_dev);
    Model& loadModel();
    RobotState readOnce();
    void control(std::function<Torques(const RobotState&, Duration)> control_callback);

private:
    void populateRobotState(const DeviceState& ds, double dt);
    void updateGMO(const std::array<double, 7>& q,
                   const std::array<double, 7>& dq,
                   const std::array<double, 7>& tau_cmd,
                   double dt);
    void checkFrankaErrors(const Vector7& tau_cmd, const Vector7& dq, const Vector7& q);

private:
    Simulation*            sim    = nullptr;
    std::string            name_;
    std::string            ee_frame_name_;
    std::unique_ptr<Model> model_;
    RobotState             robot_state_;
    std::atomic<bool>      bRunning{false};
    Vector7                tau_filtered_;
    Vector7                tau_prev_;

    Vector7 r_;
    Vector7 p_prev_;
    static constexpr double K_GMO = 50.0;
};

}