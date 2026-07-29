#pragma once

#include <string>
#include <vector>
#include <array>
#include <filesystem>

#include <yaml-cpp/yaml.h>


// ---------------------------------------------------------------------------
// Config structs
// ---------------------------------------------------------------------------

struct DeviceConfig {
    std::string            name;
    std::string            type;
    bool                   enabled;
    std::string            model_path;
    std::string            root_body;
    std::array<double, 3>  position;
    std::array<double, 4>  orientation;
    std::vector<double>    q0;
    std::string            gripper_actuator;
    std::string            attach_to;
    std::array<double, 3>  attach_offset_pos;
    std::array<double, 4>  attach_offset_quat;
};

struct ObjectConfig {
    std::string            name;
    std::string            type;
    std::string            model_path;
    std::array<double, 3>  position;
    std::array<double, 4>  orientation;
    bool                   enabled=true;
};

struct CameraConfig {
    std::string            name;
    std::string            type;
    std::array<double, 3>  pos;
    std::array<double, 4>  quat;
    double                 fovy;
};

// ---------------------------------------------------------------------------
// Result returned from SceneBuilder::build()
// ---------------------------------------------------------------------------

struct BuiltScene {
    std::filesystem::path      xml_path;
    std::vector<DeviceConfig>  devices;
    std::vector<ObjectConfig>  objects;
    std::vector<CameraConfig>  cameras;
};

// ---------------------------------------------------------------------------
// SceneBuilder
// ---------------------------------------------------------------------------

class SceneBuilder {
public:
    // Load sim_config from path and, if simulation.task_config is set, merge
    // the task file's objects into it. Call this everywhere sim_config is needed
    // so task objects are always visible regardless of entry point.
    static YAML::Node loadMergedSimConfig(const std::string& sim_config_path);

    static BuiltScene build(const YAML::Node& sim_config, const YAML::Node& robot_config);

private:
    static std::vector<DeviceConfig> parseDevices(const YAML::Node& sim_config,
                                                   const YAML::Node& robot_config);
    static std::vector<ObjectConfig> parseObjects(const YAML::Node& sim_config);
    static std::vector<CameraConfig> parseCameras(const YAML::Node& sim_config);

    static std::array<double, 4> lookAtToQuat(const std::array<double, 3>& pos,
                                               const std::array<double, 3>& look_at);

    static std::string buildSceneXML(const std::vector<DeviceConfig>& devices,
                                     const std::vector<ObjectConfig>&  objects,
                                     const std::vector<CameraConfig>&  cameras,
                                     const std::string&                baseScenePath);
};