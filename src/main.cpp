#include <iostream>
#include <stdexcept>
#include <chrono>
#include <thread>
#include <cmath>
#include <atomic>
#include <csignal>
#include <ctime>
#include <cstring>   // strrchr, for resolving a crash address to module+RVA
#include <fstream>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#ifndef WITH_FRANKA
#include <GLFW/glfw3.h>
#endif
#include <yaml-cpp/yaml.h>

#include "avatar.hpp"
#include "twin/role.hpp"

static std::atomic<bool> g_shutdown_requested{false};

// ---------- crash / unhandled-exception helpers ----------

static void logCrash(const char* reason)
{
    // Print to stderr (visible in terminal) and also append to a sidecar file.
    std::time_t t = std::time(nullptr);
    char ts[32];
    std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S", std::localtime(&t));

    std::string msg = std::string("\n[CRASH ") + ts + "] " + reason + "\n";
    std::cerr << msg << std::flush;

    std::ofstream f("../avatar_crash.log", std::ios::app);
    if (f) f << msg;
}

#ifdef _WIN32
static const char* sehCodeName(DWORD code)
{
    switch (code) {
        case EXCEPTION_ACCESS_VIOLATION:      return "ACCESS_VIOLATION (bad/null pointer deref)";
        case EXCEPTION_INT_DIVIDE_BY_ZERO:    return "INT_DIVIDE_BY_ZERO";
        case EXCEPTION_FLT_DIVIDE_BY_ZERO:    return "FLT_DIVIDE_BY_ZERO";
        case EXCEPTION_STACK_OVERFLOW:        return "STACK_OVERFLOW";
        case EXCEPTION_ILLEGAL_INSTRUCTION:   return "ILLEGAL_INSTRUCTION";
        case EXCEPTION_INT_OVERFLOW:          return "INT_OVERFLOW";
        case EXCEPTION_ARRAY_BOUNDS_EXCEEDED: return "ARRAY_BOUNDS_EXCEEDED";
        case EXCEPTION_DATATYPE_MISALIGNMENT: return "DATATYPE_MISALIGNMENT";
        case EXCEPTION_PRIV_INSTRUCTION:      return "PRIV_INSTRUCTION";
        case EXCEPTION_IN_PAGE_ERROR:         return "IN_PAGE_ERROR";
        default:                              return "unknown";
    }
}

static LONG WINAPI sehHandler(EXCEPTION_POINTERS* ep)
{
    const DWORD code = ep->ExceptionRecord->ExceptionCode;
    void* addr = ep->ExceptionRecord->ExceptionAddress;

    // A bare absolute address is close to useless: ASLR rebases every module on
    // every run, so the same faulting instruction logs a different address each
    // time (which is why two crashes at the same DLL offset looked unrelated).
    // Resolve it to module + RVA, which IS stable and is what you feed to a
    // disassembler or `dumpbin`/addr2line against the matching build.
    char module[MAX_PATH] = "<unknown>";
    uintptr_t rva = 0;
    HMODULE mod = nullptr;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
                           | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           static_cast<LPCSTR>(addr), &mod) && mod) {
        char full[MAX_PATH];
        if (GetModuleFileNameA(mod, full, MAX_PATH)) {
            const char* base = strrchr(full, '\\');
            snprintf(module, sizeof(module), "%s", base ? base + 1 : full);
        }
        rva = reinterpret_cast<uintptr_t>(addr) - reinterpret_cast<uintptr_t>(mod);
    }

    char detail[256] = "";
    if (code == EXCEPTION_ACCESS_VIOLATION || code == EXCEPTION_IN_PAGE_ERROR) {
        // ExceptionInformation[0]: 0 = read, 1 = write, 8 = DEP/execute.
        // [1] is the address that was touched -- 0x0 (or a small offset from it)
        // means an actual null dereference, anything else is a wild/stale pointer.
        const ULONG_PTR op = ep->ExceptionRecord->ExceptionInformation[0];
        const ULONG_PTR at = ep->ExceptionRecord->ExceptionInformation[1];
        snprintf(detail, sizeof(detail), " | %s of 0x%llX%s",
                 op == 0 ? "read" : op == 1 ? "write" : "execute",
                 static_cast<unsigned long long>(at),
                 at < 0x10000 ? "  <-- NULL-page access, this is a null/offset-from-null deref" : "");
    }

    char buf[640];
    snprintf(buf, sizeof(buf),
             "SEH 0x%08lX %s at %s+0x%llX (abs 0x%p, thread %lu)%s",
             code, sehCodeName(code), module,
             static_cast<unsigned long long>(rva), addr,
             GetCurrentThreadId(), detail);
    logCrash(buf);
    // Let Windows produce a crash dump / default dialog.
    return EXCEPTION_CONTINUE_SEARCH;
}
static BOOL WINAPI ctrlHandler(DWORD) {
    g_shutdown_requested.store(true);
    return TRUE;
}
#else
static void sigHandler(int sig) {
    if (sig == SIGINT || sig == SIGTERM)
        g_shutdown_requested.store(true);
    else {
        logCrash(sig == SIGSEGV ? "SIGSEGV" : sig == SIGABRT ? "SIGABRT" : "fatal signal");
        std::_Exit(1);
    }
}
#endif

static void terminateHandler()
{
    if (auto ep = std::current_exception()) {
        try { std::rethrow_exception(ep); }
        catch (const std::exception& e) { logCrash((std::string("unhandled exception: ") + e.what()).c_str()); }
        catch (...) { logCrash("unhandled exception (unknown type)"); }
    } else {
        logCrash("std::terminate called (no active exception — likely pure-virtual or thread abort)");
    }
    std::_Exit(1);
}

int main(int argc, char** argv) {
    std::set_terminate(terminateHandler);

#ifdef _WIN32
    SetUnhandledExceptionFilter(sehHandler);
    SetConsoleCtrlHandler(ctrlHandler, TRUE);
    timeBeginPeriod(1);
#else
    signal(SIGINT,  sigHandler);
    signal(SIGTERM, sigHandler);
    signal(SIGSEGV, sigHandler);
    signal(SIGABRT, sigHandler);
#endif

    try {
        YAML::Node top_config = YAML::LoadFile("../config/config.yaml");
        ResolvedConfig resolved = resolveRoleConfig(top_config, parseRoleFlag(argc, argv));

        std::cout << "Role: " << roleToString(resolved.role) << std::endl;

        Avatar avatar(resolved.node, resolved.role);

#ifdef WITH_FRANKA
        std::thread avatar_thread([&]() {
            try { avatar.start(); }
            catch (const std::exception& e) { logCrash((std::string("avatar thread: ") + e.what()).c_str()); g_shutdown_requested.store(true); }
            catch (...) { logCrash("avatar thread: unknown exception"); g_shutdown_requested.store(true); }
        });
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
            try {
                avatar.start();
            } catch (const std::exception& e) {
                logCrash((std::string("avatar thread: ") + e.what()).c_str());
                sim->stop();
                g_shutdown_requested.store(true);
            } catch (...) {
                logCrash("avatar thread: unknown exception");
                sim->stop();
                g_shutdown_requested.store(true);
            }
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
