#pragma once

#ifndef WITH_FRANKA
#include "sim_env/simulation.hpp"
#endif
#include <Eigen/Core>
#include <Eigen/Dense>
#include <functional>
#include <vector>
#include <atomic>
#include <array>

class Simulator;

namespace franka_joint_driver {

    struct HeadState{
        std::array<double, 2> q, q_cmd, dq, ddq, tau_j, tau_j_d;
        int status, fault;
        HeadState() {
            q.fill(0.0); q_cmd.fill(0.0); 
            dq.fill(0.0); ddq.fill(0.0);
            tau_j.fill(0.0);tau_j_d.fill(0.0);
        }
    };

	class Driver {
	public:
		Driver();
		~Driver();

        struct CommandTorque {
            float tau_j_d;
            bool finished = false;
        };

        struct State {
            bool error;
            float theta;
            float tau_j;
        };

        enum class Error {
            kNoError,
            kConfigSizeMismatch,
            kCommunicationError,
            kControllerNotSupported,
        };
        
        Error init() {
            return Error::kNoError;
        }

        using CallbackFunctionTorque =
            std::function<void(const std::vector<State>&, std::vector<CommandTorque>&)>;

        Error control(const CallbackFunctionTorque& driver_callback_torque);

#ifndef WITH_FRANKA
    public:
        void set_simulation(Simulation& sim, const YAML::Node& device_config);

    private:
        Simulation* sim = nullptr;
#endif
        std::string name_;
        std::atomic<bool> bRunning;
        std::vector<CommandTorque> command;
        std::vector<State> state;
        HeadState head_state;
	};
}