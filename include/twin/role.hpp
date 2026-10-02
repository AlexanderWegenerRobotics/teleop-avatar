#pragma once

// Avatar/twin role selection from config.yaml's role key or a CLI flag.

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

// selected role and its config sub-node
struct ResolvedConfig {
    Role       role;
    YAML::Node node;
};

// role_override (from the CLI flag) wins over config.yaml's role key.
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

// Scans argv for --avatar / --twin, nullopt if neither.
inline std::optional<Role> parseRoleFlag(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--twin")   return Role::Twin;
        if (arg == "--avatar") return Role::Avatar;
    }
    return std::nullopt;
}
