#include "sim_env/gripper.hpp"
#include "sim_env/simulation.hpp"

#include <cmath>
#include <algorithm>
#include <chrono>
#include <thread>
#include <iostream>

namespace franka {

Gripper::Gripper() {}
Gripper::~Gripper() {}

void Gripper::set_simulation(Simulation& sim, const std::string& device_name) {
    sim_  = &sim;
    name_ = device_name;
    activate();
}

double Gripper::currentWidth() {
    if (!sim_) return 0.0;
    //return commanded_width_.load(); 
    return sim_->getGripperWidth(name_);
}

void Gripper::commandWidth(double width) {
    if (!sim_) return;
    double clamped = std::clamp(width, 0.0, kMaxWidth);
    sim_->setGripper(name_, clamped);
}

bool Gripper::moveToWidth(double target_width, double speed, double epsilon_inner, double epsilon_outer, bool check_grasp) {
    bStop_.store(false);

    double clamped = std::clamp(target_width, 0.0, kMaxWidth);
    commandWidth(clamped);

    auto start = std::chrono::steady_clock::now();

    while (!bStop_.load()) {
        double w = currentWidth();

        bool within_outer = std::abs(w - clamped) < epsilon_outer;
        bool within_inner = std::abs(w - clamped) < epsilon_inner;

        if (check_grasp) {
            bool grasped = (w > clamped - epsilon_inner) && (w < clamped + epsilon_outer);
            if (grasped) return true;

            bool stalled = within_outer && !within_inner;
            if (stalled) return true;
        } else {
            if (within_outer) return true;
        }

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
        if (elapsed > kMoveTimeoutMs) return false;

        std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(kPollIntervalMs)));
    }
    return false;
}

bool Gripper::homing() {
    if (!sim_) return false;
    return moveToWidth(kMaxWidth, 0.1, 0.005, 0.005, false);
}

bool Gripper::move(double width, double speed) {
    return moveToWidth(width, speed, 0.005, 0.005, false);
}

bool Gripper::grasp(double width, double speed, double force, double epsilon_inner, double epsilon_outer) {
    double squeeze_width = width - 0.005;
    return moveToWidth(squeeze_width, speed, epsilon_inner, epsilon_outer, true);
}

bool Gripper::stop() {
    bStop_.store(true);
    return true;
}

void Gripper::setWidth(double width) {
    commanded_width_.store(width);
    commandWidth(width);
}

GripperState Gripper::readOnce() {
    GripperState state{};
    state.width      = currentWidth();
    state.is_grasped = false;
    return state;
}

void Gripper::activate() {
    if (sim_) sim_->setDeviceActive(name_, true);
}

}