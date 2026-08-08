#pragma once

// Role selection for the avatar/twin shared binary.
//
// One executable, two roles, chosen entirely at runtime via config.yaml's
// top-level `role:` key -- see docs/twin_concept.md section 5
// ("role-based reuse"). Role governs which config sub-tree is loaded
// (avatar_config vs twin_config) and, in Avatar, whether a Reconciler is
// constructed and which direction telemetry flows.
//
// Deliberately header-only: this is pure config parsing with no
// dependency on MuJoCo/Franka, so it must compile identically in every
// build configuration (WITH_MUJOCO on/off, WITH_FRANKA on/off).

#include <optional>
#include <stdexcept>
#include <string>

#include <yaml-cpp/yaml.h>

enum class Role {
    Avatar,
    Twin
};

inline Role roleFromString(const std::string& s) {
    if (s == "avatar") return Role::Avatar;
    if (s == "twin")   return Role::Twin;
    throw std::runtime_error(
        "config.yaml: unknown role '" + s + "' (expected 'avatar' or 'twin')");
}

inline const char* roleToString(Role role) {
    return role == Role::Twin ? "twin" : "avatar";
}

// Result of resolving config.yaml against the active role: which role was
// selected, and the flat config sub-node (sim_config / robot_config /
// streamer_config / [reconciler_config]) that Avatar/Simulation already
// know how to consume -- unchanged from the pre-role single-config schema.
struct ResolvedConfig {
    Role       role;
    YAML::Node node;
};

// config.yaml:
//   role: avatar | twin
//   avatar_config: { sim_config, robot_config, streamer_config }
//   twin_config:   { sim_config, robot_config, streamer_config, reconciler_config }
//
// role_override, when set, wins over config.yaml's `role:` key entirely --
// this is how the --avatar/--twin CLI flag (see parseRoleFlag below) takes
// effect. When not set (the common case: no flag passed), behavior is
// unchanged from before the flag existed -- config.yaml's `role:` key is
// the sole source of truth, so nothing about existing launches changes.
inline ResolvedConfig resolveRoleConfig(const YAML::Node& top_config,
                                         std::optional<Role> role_override = std::nullopt) {
    Role role;
    if (role_override) {
        role = *role_override;
    } else {
        if (!top_config["role"])
            throw std::runtime_error("config.yaml: missing required 'role' key ('avatar' or 'twin')");
        role = roleFromString(top_config["role"].as<std::string>());
    }

    std::string key = std::string(roleToString(role)) + "_config";

    if (!top_config[key])
        throw std::runtime_error("config.yaml: missing '" + key + "' block for role '" +
                                  roleToString(role) + "'");

    return ResolvedConfig{role, top_config[key]};
}

// Scans argv for --avatar / --twin. Returns nullopt if neither is present,
// meaning "fall back to config.yaml's role: key" -- callers pass the result
// straight into resolveRoleConfig's role_override parameter. Shared between
// avatar.exe (main.cpp) and avatar_pipeline.exe (streamer_main.cpp) so a
// single flag, forwarded to both processes by launch.bat, picks the same
// role for both.
inline std::optional<Role> parseRoleFlag(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--twin")   return Role::Twin;
        if (arg == "--avatar") return Role::Avatar;
    }
    return std::nullopt;
}
