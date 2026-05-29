#include <iostream>
#include <stdexcept>
#include <chrono>
#include <thread>
#include <cmath>
#include <atomic>
#include <csignal>

#ifndef WITH_FRANKA
#include <GLFW/glfw3.h>
#endif
#include <yaml-cpp/yaml.h>

#include "avatar.hpp"

static std::atomic<bool> g_shutdown_requested{false};

#ifdef _WIN32
static BOOL WINAPI ctrlHandler(DWORD) {
    g_shutdown_requested.store(true);
    return TRUE;
}
#else
static void sigHandler(int) {
    g_shutdown_requested.store(true);
}
#endif

int main() {
#ifdef _WIN32
    SetConsoleCtrlHandler(ctrlHandler, TRUE);
    timeBeginPeriod(1);
#else
    signal(SIGINT,  sigHandler);
    signal(SIGTERM, sigHandler);
#endif

    try {
        YAML::Node config = YAML::LoadFile("../config/config.yaml");

        Avatar avatar(config);

#ifdef WITH_FRANKA
        std::thread avatar_thread([&]() { avatar.start(); });
        std::cout << "Avatar started (Franka mode)" << std::endl;
        while (!g_shutdown_requested.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        std::cout << "Avatar will be stopped" << std::endl;
        avatar.stop();
        if (avatar_thread.joinable()) avatar_thread.join();
#else
        if (!glfwInit())
            throw std::runtime_error("glfwInit failed");

        auto sim = avatar.getSim();

        if (sim->render_enabled_) {
            int ncam = static_cast<int>(sim->render_cams_.size());
            int cols = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(ncam))));
            int rows = (ncam + cols - 1) / cols;

            glfwWindowHint(GLFW_SAMPLES, 4);
            sim->window_ = glfwCreateWindow(cols * 640, rows * 480,
                                            "avatar — simulation", nullptr, nullptr);
            if (!sim->window_)
                throw std::runtime_error("glfwCreateWindow failed");
        }
        if (sim->shm_enabled_) {
            glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
            sim->offscreen_window_ = glfwCreateWindow(1, 1, "", nullptr, nullptr);
            if (!sim->offscreen_window_)
                throw std::runtime_error("Failed to create offscreen window for streaming");
        }

        sim->start();
        std::thread avatar_thread([&]() {
            avatar.start();
        });
        std::cout << "Avatar started" << std::endl;

        while (sim->isRunning()) {
            if (g_shutdown_requested.load()) {
                sim->stop();
                break;
            }
            if (sim->window_ && glfwWindowShouldClose(sim->window_)) {
                sim->stop();
                break;
            }
            glfwPollEvents();
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }
        std::cout << "Avatar will be stopped" << std::endl;
        avatar.stop();
        sim->stop();

        if (avatar_thread.joinable()) avatar_thread.join();

        glfwTerminate();
#endif

    } catch (const std::exception& e) {
        std::cerr << "[Fatal] " << e.what() << "\n";
        return 1;
    }
    std::cout << "Clean close" << std::endl;
    return 0;
}
