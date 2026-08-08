#pragma once

// Applies twin-role overlays onto already-loaded config YAML trees
// (docs/twin_concept.md). Exists so avatar and twin can run as two local
// processes against the SAME robot_config / pipeline_config files -- e.g.
// for reconciler dev/testing -- without duplicating either file (and risking
// the copies' physical/dynamics parameters, or camera/stream definitions,
// drifting apart between roles).
//
// Two overlays live here:
//
//   applyTwinTransmissionOverlay -- robot_config: avatar.transmission and
//   per-device transmission blocks (UDP command-channel ports).
//
//   applyTwinStreamerOverlay -- pipeline_config: shm names, the video
//   stream/feedback/status ports, and episode_listener_port.
//
// Both are intentionally narrow, named-key merges (never a blind whole-file
// merge): only keys actually present in the overlay file are overwritten;
// anything not mentioned (remote_ip, frequency, bitrate_kbps, ...) is left
// as-is from the base file.
//
// Deliberately header-only and yaml-cpp-only, matching twin/role.hpp's
// constraint: must compile identically in every build configuration
// (WITH_MUJOCO on/off, WITH_FRANKA on/off).

#include <stdexcept>
#include <string>

#include <yaml-cpp/yaml.h>

// Recursively overwrites, in `target`, only the keys present in `overrides`.
// A key whose value is a map in both `target` and `overrides` is merged
// key-by-key (recursing); any other key (scalar, sequence, or a key not yet
// present in target) is replaced outright. `target` must already be a real
// map node (caller ensures this, e.g. sys_config["avatar"]).
inline void mergeYamlNodeInto(YAML::Node target, const YAML::Node& overrides) {
    for (const auto& kv : overrides) {
        const std::string key = kv.first.as<std::string>();
        const YAML::Node&  val = kv.second;
        YAML::Node existing = target[key];
        if (val.IsMap() && existing && existing.IsMap())
            mergeYamlNodeInto(existing, val);
        else
            target[key] = val;
    }
}

// sys_config: the loaded robot_config tree (mutated in place).
// overlay_path: path to the overlay YAML file, as found under
//   config.yaml's twin_config.transmission_overlay.
//
// Overlay file schema (see config/robot_config_twin_overlay_local.yaml):
//   avatar:
//     <any avatar-level sub-block, e.g. transmission / pipeline_logger>: { ... }
//   devices:
//     - name: <device name, matched against the base config's devices[].name>
//       <any device-level sub-block, e.g. transmission>: { ... }
inline void applyTwinTransmissionOverlay(YAML::Node sys_config, const std::string& overlay_path) {
    YAML::Node overlay = YAML::LoadFile(overlay_path);

    if (overlay["avatar"]) {
        if (!sys_config["avatar"])
            throw std::runtime_error(
                "twin transmission_overlay: base robot_config has no 'avatar' block "
                "to override (overlay: " + overlay_path + ")");
        mergeYamlNodeInto(sys_config["avatar"], overlay["avatar"]);
    }

    if (!overlay["devices"]) return;

    for (const auto& override_dev : overlay["devices"]) {
        const std::string name = override_dev["name"].as<std::string>();

        bool matched = false;
        for (auto base_dev : sys_config["devices"]) {
            if (base_dev["name"].as<std::string>() != name) continue;
            mergeYamlNodeInto(base_dev, override_dev);
            matched = true;
            break;
        }
        if (!matched)
            throw std::runtime_error(
                "twin transmission_overlay: device '" + name +
                "' not found in base robot_config (overlay: " + overlay_path + ")");
    }
}

// cfg: the loaded pipeline_config tree (mutated in place) -- same file both
//   Simulation (avatar.exe, writer side) and streamer_main (avatar_pipeline.exe,
//   reader side) load via config["streamer_config"].
// overlay_path: path to the overlay YAML file, as found under
//   config.yaml's twin_config.streamer_overlay.
//
// Overlay file schema (see config/pipeline_config_twin_overlay_local.yaml):
//   episode_listener_port: <override>
//   stream_cameras:
//     - camera: <matched against base stream_cameras[].camera>
//       shm_name: <override>
//   cameras:
//     - name: <matched against base cameras[].name>
//       shm_name: <override>
//       stereo_partner_shm: <override, only for the stereo entry>
//       stream: { port, feedback_port, status_port, ... }
inline void applyTwinStreamerOverlay(YAML::Node cfg, const std::string& overlay_path) {
    YAML::Node overlay = YAML::LoadFile(overlay_path);

    // Top-level scalars (episode_listener_port, ...). stream_cameras/cameras
    // are sequences matched by name below, not blindly replaced here.
    for (const auto& kv : overlay) {
        const std::string key = kv.first.as<std::string>();
        if (key == "stream_cameras" || key == "cameras") continue;
        cfg[key] = kv.second;
    }

    auto mergeByField = [&](const char* list_key, const char* match_field) {
        if (!overlay[list_key]) return;
        for (const auto& override_entry : overlay[list_key]) {
            const std::string name = override_entry[match_field].as<std::string>();
            bool matched = false;
            for (auto base_entry : cfg[list_key]) {
                if (base_entry[match_field].as<std::string>() != name) continue;
                mergeYamlNodeInto(base_entry, override_entry);
                matched = true;
                break;
            }
            if (!matched)
                throw std::runtime_error(
                    "twin streamer_overlay: entry '" + name + "' not found in base '" +
                    list_key + "' (overlay: " + overlay_path + ")");
        }
    };

    mergeByField("stream_cameras", "camera");
    mergeByField("cameras", "name");
}
