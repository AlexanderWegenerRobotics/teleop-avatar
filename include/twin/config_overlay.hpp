#pragma once

// Applies twin-role overlays onto loaded robot/pipeline config trees, so avatar and twin can share one config file.
// Only keys present in the overlay are overwritten.

#include <stdexcept>
#include <string>

#include <yaml-cpp/yaml.h>

// Recursively overwrites only the keys present in overrides; maps are merged, everything else replaced.
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

// Overlays avatar.* and per-device blocks (matched by devices[].name) from the transmission overlay file.
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

// Overlays top-level keys, stream_cameras (by camera) and cameras (by name) from the streamer overlay file.
inline void applyTwinStreamerOverlay(YAML::Node cfg, const std::string& overlay_path) {
    YAML::Node overlay = YAML::LoadFile(overlay_path);

    // top-level keys; the two camera lists are merged by name below
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
