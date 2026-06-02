#include "sim_env/scene_builder.hpp"

#include <fstream>
#include <stdexcept>
#include <filesystem>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <cstdio>
#include <iostream>

#include <tinyxml2.h>

using namespace tinyxml2;


// ===========================================================================
// Internal TinyXML-2 helpers
// ===========================================================================

static std::unique_ptr<XMLDocument> loadXMLDoc(const std::string& path) {
    auto doc = std::make_unique<XMLDocument>();
    if (doc->LoadFile(path.c_str()) != XML_SUCCESS)
        throw std::runtime_error("TinyXML2 failed to load: " + path + "  (" + doc->ErrorStr() + ")");
    return doc;
}

static void deepCopyChildren(XMLElement* dst, XMLDocument& dstDoc, const XMLElement* src) {
    for (const XMLNode* child = src->FirstChild(); child; child = child->NextSibling()) {
        XMLPrinter printer;
        child->Accept(&printer);

        XMLDocument tmp;
        if (tmp.Parse(printer.CStr()) == XML_SUCCESS && tmp.FirstChild())
            dst->InsertEndChild(tmp.FirstChild()->DeepClone(&dstDoc));
    }
}

static std::string docToString(XMLDocument& doc) {
    XMLPrinter printer;
    doc.Print(&printer);
    return printer.CStr();
}

static void mergeCompilerAttributes(XMLElement* dst, const XMLElement* src) {
    for (const XMLAttribute* a = src->FirstAttribute(); a; a = a->Next()) {
        const std::string name = a->Name();
        if (name == "meshdir" || name == "texturedir") continue;
        dst->SetAttribute(a->Name(), a->Value());
    }
}

static void prefixNamesInTree(XMLElement* el, const std::string& prefix) {
    static const std::vector<const char*> kAttrs = {
        "name", "joint", "joint1", "joint2",
        "body", "body1", "body2",
        "mesh", "material", "texture", "tendon",
        "site", "actuator", "childclass", "class"
    };
    for (const char* attr : kAttrs) {
        const char* val = el->Attribute(attr);
        if (val)
            el->SetAttribute(attr, (prefix + "_" + val).c_str());
    }
    for (XMLElement* ch = el->FirstChildElement(); ch; ch = ch->NextSiblingElement())
        prefixNamesInTree(ch, prefix);
}

static void normalizeMeshNames(XMLElement* assetEl) {
    for (XMLElement* el = assetEl->FirstChildElement("mesh");
         el; el = el->NextSiblingElement("mesh")) {
        if (!el->Attribute("name")) {
            const char* file = el->Attribute("file");
            if (file)
                el->SetAttribute("name",
                    std::filesystem::path(file).stem().string().c_str());
        }
    }
}

static XMLElement* findBodyByName(XMLElement* el, const std::string& name) {
    if (!el) return nullptr;
    if (std::string(el->Name()) == "body") {
        const char* n = el->Attribute("name");
        if (n && std::string(n) == name) return el;
    }
    for (XMLElement* ch = el->FirstChildElement(); ch; ch = ch->NextSiblingElement()) {
        XMLElement* found = findBodyByName(ch, name);
        if (found) return found;
    }
    return nullptr;
}

static void absoluteMeshPaths(XMLElement* assetEl, const std::string& meshdir) {
    for (XMLElement* el = assetEl->FirstChildElement("mesh");
         el; el = el->NextSiblingElement("mesh")) {
        const char* file = el->Attribute("file");
        if (file && file[0] != '/')
            el->SetAttribute("file", (meshdir + "/" + file).c_str());
    }
}

static void absoluteTexturePaths(XMLElement* assetEl, const std::string& texturedir) {
    for (XMLElement* el = assetEl->FirstChildElement("texture");
         el; el = el->NextSiblingElement("texture")) {
        const char* file = el->Attribute("file");
        if (file && file[0] != '/')
            el->SetAttribute("file", (texturedir + "/" + file).c_str());
    }
}

static std::string resolveMeshdir(const XMLElement* compilerEl,
                                   const std::string& modelPath) {
    if (!compilerEl) return {};
    const char* md = compilerEl->Attribute("meshdir");
    if (!md) return {};
    return std::filesystem::absolute(
        std::filesystem::path(modelPath).parent_path() / md).string();
}

static std::string resolveTexturedir(const XMLElement* compilerEl,
                                      const std::string& modelPath) {
    if (!compilerEl) return {};
    const char* td = compilerEl->Attribute("texturedir");
    if (!td) return {};
    return std::filesystem::absolute(
        std::filesystem::path(modelPath).parent_path() / td).string();
}

static void injectModel(const std::string& modelPath,
                        const std::string& namePrefix,
                        const std::array<double, 3>& pos,
                        const std::array<double, 4>& quat,
                        const std::string& attach_to,
                        const std::array<double, 3>& attach_offset_pos,
                        const std::array<double, 4>& attach_offset_quat,
                        XMLDocument& scene,
                        XMLElement*  compilerEl,
                        XMLElement*  assetEl,
                        XMLElement*  worldbody,
                        XMLElement*  root,
                        bool is_mocap = false,
                        bool is_dynamic = false) {
    auto doc = loadXMLDoc(modelPath);
    XMLElement* docRoot = doc->RootElement();

    if (XMLElement* comp = docRoot->FirstChildElement("compiler"))
        mergeCompilerAttributes(compilerEl, comp);

    std::string meshdir = resolveMeshdir(
        docRoot->FirstChildElement("compiler"), modelPath);

    std::string texturedir = resolveTexturedir(
        docRoot->FirstChildElement("compiler"), modelPath);

    if (XMLElement* ra = docRoot->FirstChildElement("asset"))
        normalizeMeshNames(ra);

    // Resolve absolute paths BEFORE prefixing, so file= attributes are
    // rewritten to absolute paths while names are still in their original form.
    if (XMLElement* ra = docRoot->FirstChildElement("asset")) {
        if (!meshdir.empty())
            absoluteMeshPaths(ra, meshdir);
        if (!texturedir.empty())
            absoluteTexturePaths(ra, texturedir);
    }

    prefixNamesInTree(docRoot, namePrefix);

    for (XMLElement* def = docRoot->FirstChildElement("default");
         def; def = def->NextSiblingElement("default"))
        root->InsertEndChild(def->DeepClone(&scene));

    if (XMLElement* ra = docRoot->FirstChildElement("asset")) {
        deepCopyChildren(assetEl, scene, ra);
    }

    if (XMLElement* rw = docRoot->FirstChildElement("worldbody")) {
    char posBuf[64], quatBuf[80];

    if (!attach_to.empty()) {
        std::snprintf(posBuf,  sizeof(posBuf),  "%.6g %.6g %.6g", attach_offset_pos[0], attach_offset_pos[1], attach_offset_pos[2]);
        std::snprintf(quatBuf, sizeof(quatBuf), "%.6g %.6g %.6g %.6g", attach_offset_quat[0], attach_offset_quat[1], attach_offset_quat[2], attach_offset_quat[3]);
    } else {
        std::snprintf(posBuf,  sizeof(posBuf),  "%.6g %.6g %.6g", pos[0], pos[1], pos[2]);
        std::snprintf(quatBuf, sizeof(quatBuf), "%.6g %.6g %.6g %.6g", quat[0], quat[1], quat[2], quat[3]);
    }

    if (is_dynamic) {
            // Inject top-level bodies directly into worldbody so freejoint is at the right depth
            for (XMLElement* child = rw->FirstChildElement(); child; child = child->NextSiblingElement()) {
                XMLNode* clone = child->DeepClone(&scene);
                XMLElement* cloneEl = clone->ToElement();
                if (cloneEl && std::string(cloneEl->Name()) == "body") {
                    cloneEl->SetAttribute("pos",  posBuf);
                    cloneEl->SetAttribute("quat", quatBuf);
                }
                worldbody->InsertEndChild(clone);
            }
        } else {
            // All other types (static, visual, mocap): wrap in a frame body as before
            XMLElement* frame = scene.NewElement("body");
            frame->SetAttribute("name", (namePrefix + "_frame").c_str());
            if (is_mocap)
                frame->SetAttribute("mocap", "true");
            frame->SetAttribute("pos",  posBuf);
            frame->SetAttribute("quat", quatBuf);
            deepCopyChildren(frame, scene, rw);

            if (!attach_to.empty()) {
                XMLElement* parent = findBodyByName(worldbody, attach_to);
                if (!parent)
                    throw std::runtime_error("injectModel: attach_to body '" + attach_to + "' not found in scene");
                parent->InsertEndChild(frame);
            } else {
                worldbody->InsertEndChild(frame);
            }
        }
    }

    for (const char* tag : {"actuator", "tendon", "equality", "contact"})
        for (XMLElement* el = docRoot->FirstChildElement(tag);
             el; el = el->NextSiblingElement(tag))
            root->InsertEndChild(el->DeepClone(&scene));
}


// ===========================================================================
// SceneBuilder — public interface
// ===========================================================================

BuiltScene SceneBuilder::build(const YAML::Node& sim_config, const YAML::Node& robot_config) {
    BuiltScene result;
    result.devices = parseDevices(sim_config, robot_config);
    result.objects = parseObjects(sim_config);
    result.cameras = parseCameras(sim_config);

    std::string baseScenePath;
    if (sim_config["simulation"] && sim_config["simulation"]["model_path"])
        baseScenePath = sim_config["simulation"]["model_path"].as<std::string>();

    std::string xml = buildSceneXML(result.devices, result.objects, result.cameras, baseScenePath);

    result.xml_path = std::filesystem::path(baseScenePath).parent_path() / "_generated_scene.xml";

    {
        std::ofstream out(result.xml_path);
        if (!out.is_open())
            throw std::runtime_error("Cannot write generated scene: " + result.xml_path.string());
        out << xml;
    }
    return result;
}


// ===========================================================================
// YAML parsing
// ===========================================================================

std::vector<DeviceConfig> SceneBuilder::parseDevices(const YAML::Node& sim_config, const YAML::Node& robot_config) {
    std::unordered_map<std::string, YAML::Node> sim_devs;
    if (sim_config["devices"])
        for (const auto& sd : sim_config["devices"])
            sim_devs[sd["name"].as<std::string>()] = YAML::Node(sd);

    std::vector<DeviceConfig> devices;
    if (!robot_config["devices"]) return devices;

    // Collect names of disabled devices so dependents (e.g. grippers
    // attached to a disabled arm) can be skipped too.
    std::unordered_set<std::string> disabled_devices;
    for (const auto& rd : robot_config["devices"])
        if (rd["enabled"] && !rd["enabled"].as<bool>())
            disabled_devices.insert(rd["name"].as<std::string>());

    for (const auto& rd : robot_config["devices"]) {
        if (rd["enabled"] && !rd["enabled"].as<bool>()) continue;

        std::string name = rd["name"].as<std::string>();

        auto it = sim_devs.find(name);
        if (it == sim_devs.end()) {
            std::cerr << "[SceneBuilder] WARNING: device '" << name << "' in robot_config has no sim_config entry — skipping.\n";
            continue;
        }
        const YAML::Node& sd = it->second;

        DeviceConfig dev;
        dev.name             = name;
        dev.type             = rd["type"].as<std::string>();
        dev.enabled          = true;
        dev.model_path       = sd["model_path"].as<std::string>();
        dev.root_body        = sd["root_body"].as<std::string>();
        dev.gripper_actuator = sd["gripper_actuator"] ? sd["gripper_actuator"].as<std::string>() : "";
        dev.attach_to = sd["attach_to"] ? sd["attach_to"].as<std::string>() : "";

        // Skip devices attached to a disabled device (body name is prefixed
        // with the parent device name, e.g. "arm_left_fr3_link7").
        if (!dev.attach_to.empty()) {
            bool parent_disabled = false;
            for (const auto& disabled : disabled_devices) {
                if (dev.attach_to.rfind(disabled + "_", 0) == 0) {
                    parent_disabled = true;
                    break;
                }
            }
            if (parent_disabled) {
                std::cerr << "[SceneBuilder] Skipping '" << name
                          << "' because its attach_to parent is disabled.\n";
                continue;
            }
        }

        if (sd["attach_offset"]) {
            auto p = sd["attach_offset"]["position"];
            dev.attach_offset_pos = {p[0].as<double>(), p[1].as<double>(), p[2].as<double>()};
            auto o = sd["attach_offset"]["orientation"];
            dev.attach_offset_quat = {o[0].as<double>(), o[1].as<double>(),o[2].as<double>(), o[3].as<double>()};
        } else {
            dev.attach_offset_pos  = {0, 0, 0};
            dev.attach_offset_quat = {1, 0, 0, 0};
        }

        auto pos = rd["base_pose"]["position"];
        dev.position = {pos[0].as<double>(), pos[1].as<double>(), pos[2].as<double>()};

        auto ori = rd["base_pose"]["orientation"];
        dev.orientation = {ori[0].as<double>(), ori[1].as<double>(),
                           ori[2].as<double>(), ori[3].as<double>()};

        if (rd["q0"])
            for (const auto& q : rd["q0"])
                dev.q0.push_back(q.as<double>());

        devices.push_back(dev);
    }
    return devices;
}

std::vector<ObjectConfig> SceneBuilder::parseObjects(const YAML::Node& sim_config) {
    std::vector<ObjectConfig> objects;
    if (!sim_config["objects"]) return objects;

    for (const auto& node : sim_config["objects"]) {
        ObjectConfig obj;
        obj.name       = node["name"].as<std::string>();
        obj.type       = node["type"].as<std::string>();
        obj.model_path = node["model_path"].as<std::string>();
        obj.enabled = node["enabled"] ? node["enabled"].as<bool>() : true;

        if (node["pose"]) {
            auto pos = node["pose"]["position"];
            obj.position = {pos[0].as<double>(), pos[1].as<double>(), pos[2].as<double>()};
            auto ori = node["pose"]["orientation"];
            obj.orientation = {ori[0].as<double>(), ori[1].as<double>(),
                               ori[2].as<double>(), ori[3].as<double>()};
        } else {
            obj.position    = {0, 0, 0};
            obj.orientation = {1, 0, 0, 0};
        }

        objects.push_back(obj);
    }
    return objects;
}

std::vector<CameraConfig> SceneBuilder::parseCameras(const YAML::Node& sim_config) {
    std::vector<CameraConfig> cameras;
    if (!sim_config["cameras"]) return cameras;

    for (const auto& node : sim_config["cameras"]) {
        if (!node["enabled"].as<bool>()) continue;

        CameraConfig cam;
        cam.name = node["name"].as<std::string>();
        cam.type = node["type"].as<std::string>();
        cam.fovy = node["fovy"].as<double>();

        auto p = node["pos"];
        cam.pos = {p[0].as<double>(), p[1].as<double>(), p[2].as<double>()};

        auto la = node["look_at"];
        std::array<double, 3> look_at = {
            la[0].as<double>(), la[1].as<double>(), la[2].as<double>()};

        cam.quat = lookAtToQuat(cam.pos, look_at);
        cameras.push_back(cam);
    }
    return cameras;
}

std::array<double, 4> SceneBuilder::lookAtToQuat(const std::array<double, 3>& pos,
                                                   const std::array<double, 3>& look_at) {
    double fx = look_at[0] - pos[0];
    double fy = look_at[1] - pos[1];
    double fz = look_at[2] - pos[2];
    double flen = std::sqrt(fx*fx + fy*fy + fz*fz);
    if (flen < 1e-9) return {1.0, 0.0, 0.0, 0.0};
    fx /= flen; fy /= flen; fz /= flen;

    double wx = 0.0, wy = 0.0, wz = 1.0;
    if (std::abs(fz) > 0.999) { wx = 1.0; wy = 0.0; wz = 0.0; }

    double rx = fy*wz - fz*wy;
    double ry = fz*wx - fx*wz;
    double rz = fx*wy - fy*wx;
    double rlen = std::sqrt(rx*rx + ry*ry + rz*rz);
    rx /= rlen; ry /= rlen; rz /= rlen;

    double ux = ry*fz - rz*fy;
    double uy = rz*fx - rx*fz;
    double uz = rx*fy - ry*fx;

    double m00 = rx,  m01 = ux,  m02 = -fx;
    double m10 = ry,  m11 = uy,  m12 = -fy;
    double m20 = rz,  m21 = uz,  m22 = -fz;

    double trace = m00 + m11 + m22;
    double qw, qx, qy, qz;
    if (trace > 0.0) {
        double s = 0.5 / std::sqrt(trace + 1.0);
        qw = 0.25 / s;
        qx = (m21 - m12) * s;
        qy = (m02 - m20) * s;
        qz = (m10 - m01) * s;
    } else if (m00 > m11 && m00 > m22) {
        double s = 2.0 * std::sqrt(1.0 + m00 - m11 - m22);
        qw = (m21 - m12) / s;
        qx = 0.25 * s;
        qy = (m01 + m10) / s;
        qz = (m02 + m20) / s;
    } else if (m11 > m22) {
        double s = 2.0 * std::sqrt(1.0 + m11 - m00 - m22);
        qw = (m02 - m20) / s;
        qx = (m01 + m10) / s;
        qy = 0.25 * s;
        qz = (m12 + m21) / s;
    } else {
        double s = 2.0 * std::sqrt(1.0 + m22 - m00 - m11);
        qw = (m10 - m01) / s;
        qx = (m02 + m20) / s;
        qy = (m12 + m21) / s;
        qz = 0.25 * s;
    }

    return {qw, qx, qy, qz};
}


// ===========================================================================
// XML assembly
// ===========================================================================

std::string SceneBuilder::buildSceneXML(const std::vector<DeviceConfig>& devices,
                                         const std::vector<ObjectConfig>&  objects,
                                         const std::vector<CameraConfig>&  cameras,
                                         const std::string&                baseScenePath) {
    XMLDocument scene;
    XMLElement* root = scene.NewElement("mujoco");
    root->SetAttribute("model", "scene");
    scene.InsertFirstChild(root);

    XMLElement* compilerEl = scene.NewElement("compiler");
    root->InsertEndChild(compilerEl);

    XMLElement* optionEl = scene.NewElement("option");
    optionEl->SetAttribute("gravity", "0 0 -9.81");
    optionEl->SetAttribute("integrator", "implicitfast");
    root->InsertEndChild(optionEl);

    XMLElement* assetEl   = scene.NewElement("asset");
    XMLElement* worldbody = scene.NewElement("worldbody");
    root->InsertEndChild(assetEl);
    root->InsertEndChild(worldbody);

    {
        auto baseDoc  = loadXMLDoc(baseScenePath);
        XMLElement* baseRoot = baseDoc->RootElement();
        if (XMLElement* ba = baseRoot->FirstChildElement("asset"))
            deepCopyChildren(assetEl, scene, ba);
        if (XMLElement* bw = baseRoot->FirstChildElement("worldbody"))
            deepCopyChildren(worldbody, scene, bw);
    }

    for (const auto& dev : devices)
        if(dev.enabled){
            injectModel(dev.model_path, dev.name,
                        dev.position, dev.orientation,
                        dev.attach_to, dev.attach_offset_pos, dev.attach_offset_quat,
                        scene, compilerEl, assetEl, worldbody, root);
        }

    for (const auto& obj : objects)
        if(obj.enabled){
        injectModel(obj.model_path, obj.name,
                    obj.position, obj.orientation,
                    "", {0,0,0}, {1,0,0,0},
                    scene, compilerEl, assetEl, worldbody, root, obj.type == "mocap", obj.type == "dynamic");
        }

    for (const auto& cam : cameras) {
        XMLElement* camEl = scene.NewElement("camera");
        camEl->SetAttribute("name", cam.name.c_str());

        char posBuf[64], quatBuf[80], fovBuf[32];
        std::snprintf(posBuf,  sizeof(posBuf),  "%.6g %.6g %.6g",
                      cam.pos[0], cam.pos[1], cam.pos[2]);
        std::snprintf(quatBuf, sizeof(quatBuf), "%.6g %.6g %.6g %.6g",
                      cam.quat[0], cam.quat[1], cam.quat[2], cam.quat[3]);
        std::snprintf(fovBuf,  sizeof(fovBuf),  "%.6g", cam.fovy);

        camEl->SetAttribute("pos",  posBuf);
        camEl->SetAttribute("quat", quatBuf);
        camEl->SetAttribute("fovy", fovBuf);

        worldbody->InsertEndChild(camEl);
    }

    return docToString(scene);
}