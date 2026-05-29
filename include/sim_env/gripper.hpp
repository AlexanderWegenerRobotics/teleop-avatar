#pragma once

#ifndef WITH_FRANKA

#include <atomic>
#include <string>

#include <yaml-cpp/yaml.h>

class Simulation;

namespace franka {

struct GripperState {
    double width;
    bool   is_grasped;
};

class Gripper {
public:
    Gripper();
    ~Gripper();

    void set_simulation(Simulation& sim, const std::string& device_name);

    bool homing();
    bool move(double width, double speed);
    bool grasp(double width, double speed, double force, double epsilon_inner = 0.005, double epsilon_outer = 0.005);
    bool stop();

    void setWidth(double width);

    GripperState readOnce();

private:
    bool moveToWidth(double target_width, double speed, double epsilon_inner, double epsilon_outer, bool check_grasp);
    double currentWidth();
    void   commandWidth(double width);
    void activate();

    static constexpr double kMaxWidth        = 0.08;
    static constexpr double kCtrlGain        = 0.01568627451;
    static constexpr double kPollIntervalMs  = 20.0;
    static constexpr double kMoveTimeoutMs   = 10000.0;

    Simulation*  sim_  = nullptr;
    std::string  name_;
    std::atomic<bool> bStop_{false};
    std::atomic<double> commanded_width_{0.0};
};

}  // namespace franka

#else  // WITH_FRANKA — use real libfranka Gripper

#include <franka/gripper.h>

#endif  // WITH_FRANKA