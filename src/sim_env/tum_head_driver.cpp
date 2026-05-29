#include "sim_env/tum_head_driver.hpp"
#ifndef WITH_FRANKA
#include "sim_env/simulation.hpp"
#endif
#include <chrono>
#include <thread>
#include <iostream>

using namespace franka_joint_driver;

Driver::Driver() {
    bRunning = true;
    for (size_t i = 0; i < 2; ++i) {
        state.push_back({ false, 0.0, 0.0 });
        command.push_back({ 0.0, false });
    }
}

Driver::~Driver() {

}

#ifndef WITH_FRANKA
void Driver::set_simulation(Simulation& _sim, const YAML::Node& device_config) {
    sim   = &_sim;
    name_ = device_config["name"].as<std::string>();
}
#endif

Driver::Error Driver::control(const Driver::CallbackFunctionTorque& driver_callback_torque) {

    constexpr std::chrono::microseconds control_period(static_cast<int>(1e6 / 1000));
    auto next_control_time = std::chrono::high_resolution_clock::now();

#ifndef WITH_FRANKA
    sim->setDeviceActive(name_, true);
    Driver::Error return_val = Driver::Error::kNoError;
    while (bRunning)
    {
        if (!sim->isRunning()) {
            bRunning = false;
            break;
        }

        DeviceState device_state = sim->getDeviceState(name_);

        for (size_t i = 0; i < 2; ++i) {
            state[i].theta = device_state.q[i];
            state[i].tau_j = device_state.tau_J[i];
        }

        driver_callback_torque(state, command);

        for (size_t i = 0; i < 2; ++i) {
            head_state.tau_j_d[i] = command[i].tau_j_d;
            if (command[i].finished) {
                std::cout << "Finished torque callback with command" << std::endl;
                bRunning = false;
                break;
            }
        }
        sim->setCtrl(name_, std::vector<double>(head_state.tau_j_d.begin(), head_state.tau_j_d.end()));

        next_control_time += control_period;
        std::this_thread::sleep_until(next_control_time);
    }
    sim->setDeviceActive(name_, false);
    return return_val;
#else
    // Placeholder for real TUM head hardware control loop.
    // Replace with actual hardware driver calls when head hardware is available.
    Driver::Error return_val = Driver::Error::kNoError;
    while (bRunning) {
        driver_callback_torque(state, command);
        for (size_t i = 0; i < 2; ++i) {
            if (command[i].finished) {
                bRunning = false;
                break;
            }
        }
        next_control_time += control_period;
        std::this_thread::sleep_until(next_control_time);
    }
    return return_val;
#endif
}
