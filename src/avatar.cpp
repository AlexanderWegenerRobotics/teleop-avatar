#include "avatar.hpp"
#include "common.hpp"
#include "network/udp_reliable.hpp"
#include "data_logger.hpp"
#include "intention/annotation_msg.hpp"

#include <algorithm>
#include <iostream>
#include <filesystem>
#include "network/platform_socket.hpp"

Avatar::Avatar(const YAML::Node& config) {
    YAML::Node sys_config = YAML::LoadFile(config["robot_config"].as<std::string>());

    session_id_ = std::to_string(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count());

    for (const auto& dev : sys_config["devices"]) {
        if (!dev["enabled"].as<bool>(true)) continue;
        std::string type = dev["type"].as<std::string>();
        std::string name = dev["name"].as<std::string>();

        if (type == "arm") {
            arm_instances.push_back(new ArmControl(dev, session_id_));
            device_records_[name] = DeviceRecord{};
        } else if (type == "head") {
            head_instances.push_back(new HeadControl(dev, session_id_));
            device_records_[name] = DeviceRecord{};
        }
    }

    device_registry_ = std::make_shared<DeviceRegistry>();

    SelfCollisionConfig scp_config;
    if (sys_config["avatar"]["self_collision"]) {
        auto sc = sys_config["avatar"]["self_collision"];
        scp_config.d_min   = sc["d_min"].as<double>(0.04);
        scp_config.alpha   = sc["alpha"].as<double>(1.0);
        scp_config.beta    = sc["beta"].as<double>(0.10);
        scp_config.enabled = sc["enabled"].as<bool>(true);
    }

    for (auto& arm : arm_instances) {
        arm->initSelfCollisionProtection(device_registry_, scp_config);
    }

    if (auto* a = getArm("arm_right")) a->setCollisionImportanceWeight(1.0);
    if (auto* a = getArm("arm_left"))  a->setCollisionImportanceWeight(1.0);

    log_base_dir_ = "logs";
    if (sys_config["avatar"]["log_dir"]){
        std::cout << "Found log dir" << std::endl;
        log_base_dir_ = sys_config["avatar"]["log_dir"].as<std::string>();
    }
    std::filesystem::create_directories(log_base_dir_);

    int max_index = -1;
    for (const auto& entry : std::filesystem::directory_iterator(log_base_dir_)) {
        if (!entry.is_directory()) continue;
        const std::string name = entry.path().filename().string();
        bool all_digits = !name.empty() && std::all_of(name.begin(), name.end(), ::isdigit);
        if (!all_digits) continue;
        try {
            int idx = std::stoi(name);
            if (idx > max_index) max_index = idx;
        } catch (...) {}
    }
    if (max_index >= 0) {
        episode_index_ = max_index + 1;
        std::cout << "[AVATAR-INFO]: Resuming from episode " << episode_index_ << std::endl;
    }

    episode_sock_ = socket(AF_INET, SOCK_DGRAM, 0);
#ifdef _WIN32
    DWORD timeout_ms = 1000;
    setsockopt(episode_sock_, SOL_SOCKET, SO_RCVTIMEO,
               reinterpret_cast<const char*>(&timeout_ms), sizeof(timeout_ms));
#else
    struct timeval tv{};
    tv.tv_sec  = 1;
    tv.tv_usec = 0;
    setsockopt(episode_sock_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif

    // ── Pipeline logger episode signaling ─────────────────────────────────
    if (sys_config["avatar"]["pipeline_logger"]) {
        const auto& pl = sys_config["avatar"]["pipeline_logger"];
        if (pl["enabled"].as<bool>(false)) {
            logger_host_ = pl["host"].as<std::string>("127.0.0.1");
            logger_port_ = pl["port"].as<int>(7100);
            logger_sock_ = socket(AF_INET, SOCK_DGRAM, 0);
            if (logger_sock_ == kInvalidSocket) {
                std::cerr << "[AVATAR-WARN] Failed to create pipeline logger socket\n";
                logger_port_ = 0;
            } else {
                std::cout << "[AVATAR-INFO] Pipeline logger signals -> " << logger_host_ << ":" << logger_port_ << std::endl;
            }
        }
    }

    if (sys_config["avatar"]["transmission"]) {
        UdpReliableConfig cmd_cfg;
        cmd_cfg.transport.remote_ip   = sys_config["avatar"]["transmission"]["remote_ip"].as<std::string>();
        cmd_cfg.transport.remote_port = sys_config["avatar"]["transmission"]["send_port"].as<int>();
        cmd_cfg.transport.bind_port   = sys_config["avatar"]["transmission"]["receive_port"].as<int>();
        cmd_cfg.poll_rate_hz          = sys_config["avatar"]["transmission"]["frequency"].as<int>();
        cmd_channel_ = std::make_unique<UdpReliable>(cmd_cfg);

        cmd_channel_->registerHandler("state_change",[this](const ReliableEnvelope& env, const msgpack::object& payload) {
            std::map<std::string, msgpack::object> fields;
            payload.convert(fields);
            auto it = fields.find("requested_state");
            if (it != fields.end()) {
                cmd_requested_.store(static_cast<SysState>(it->second.as<uint8_t>()));
            }
        });

        cmd_channel_->registerHandler("arm_reset", [this](const ReliableEnvelope& env, const msgpack::object& payload) {
            std::map<std::string, msgpack::object> fields;
            payload.convert(fields);
            auto it = fields.find("device");
            if (it == fields.end()) return;
            std::string dev_name = it->second.as<std::string>();

            ArmControl* arm = getArm(dev_name);
            if (!arm) return;

            arm->recovery().requestRecovery(RecoveryTrigger::OPERATOR_RESET, arm->getQ0());

            auto rec_it = device_records_.find(dev_name);
            if (rec_it != device_records_.end())
                rec_it->second.active = false;

            std::cout << "[AVATAR-INFO]: Reset requested for " << dev_name << std::endl;
        });

        cmd_channel_->registerHandler("arm_resume", [this](const ReliableEnvelope& env, const msgpack::object& payload) {
            if (reset_all_pending_.load()) return;

            std::map<std::string, msgpack::object> fields;
            payload.convert(fields);
            auto it = fields.find("device");
            if (it == fields.end()) return;
            std::string dev_name = it->second.as<std::string>();

            ArmControl* arm = getArm(dev_name);
            if (!arm || !arm->recovery().isWaitingAck()) return;

            arm->reOrigin();
            arm->recovery().confirmResume();

            auto rec_it = device_records_.find(dev_name);
            if (rec_it != device_records_.end())
                rec_it->second.active = true;

            if (state_ == SysState::ENGAGED) {
                arm->requestState(SysState::ENGAGED);
            }

            std::cout << "[AVATAR-INFO]: Resume confirmed for " << dev_name << std::endl;
        });

        cmd_channel_->registerHandler("reset_all", [this](const ReliableEnvelope& env, const msgpack::object& payload) {
            std::string reason = "reset_all";
            if (payload.type == msgpack::type::MAP) {
                std::map<std::string, msgpack::object> fields;
                payload.convert(fields);
                auto it = fields.find("reason");
                if (it != fields.end())
                    reason = it->second.as<std::string>();
            }

            markEpisodeEnd(reason);

            for (auto& arm : arm_instances) {
                arm->recovery().requestRecovery(RecoveryTrigger::OPERATOR_RESET, arm->getQ0());
                auto rec_it = device_records_.find(arm->getDeviceName());
                if (rec_it != device_records_.end())
                    rec_it->second.active = false;
            }

            reset_all_pending_.store(true);

            std::cout << "[AVATAR-INFO]: Global reset requested (" << reason << ")" << std::endl;
        });

        cmd_channel_->registerHandler("annotation", [this](const ReliableEnvelope& env, const msgpack::object& payload) {
            AnnotationMsg ann;
            payload.convert(ann);
            if (scene_logger_)
                scene_logger_->writeAnnotation(ann.label, ann.atype, ann.confidence, ann.score, ann.frame_id);
            std::cout << "[AVATAR-INFO]: annotation frame=" << ann.frame_id
                      << " label=" << ann.label
                      << " type=" << static_cast<int>(ann.atype)
                      << " conf=" << ann.confidence << "\n";
        });

        cmd_channel_->registerHandler("gaze_sample", [this](const ReliableEnvelope& env, const msgpack::object& payload) {
            if (!intention_buffer_) return;
            GazeSampleMsg gaze;
            payload.convert(gaze);
            intention_buffer_->fuseGaze(gaze);
        });
    }

#ifndef WITH_FRANKA
    sim_ = std::make_shared<Simulation>(config);

    YAML::Node sim_config = YAML::LoadFile(config["sim_config"].as<std::string>());

    for (const auto& obj : sim_config["objects"]) {
        if (!obj["role"]) continue;
        const std::string role      = obj["role"].as<std::string>();
        const std::string name      = obj["name"].as<std::string>();
        const std::string body_name = obj["body_name"].as<std::string>("");
        const std::string color     = obj["color"].as<std::string>("");
        const std::string mp        = obj["model_path"].as<std::string>("");
        ObjectDef def;
        def.name        = name;
        def.mujoco_body = name + "_" + body_name;
        def.color       = color;
        def.model_path  = mp;
        if (obj["pose"] && obj["pose"]["position"]) {
            auto p = obj["pose"]["position"].as<std::vector<double>>();
            def.fixed_x = p[0]; def.fixed_y = p[1]; def.fixed_z = p[2];
        }
        if (role == "object") object_defs_.push_back(def);
        else if (role == "bin") bin_defs_.push_back(def);
    }

    std::unordered_map<std::string, YAML::Node> sim_devs;
    for (const auto& sd : sim_config["devices"])
        sim_devs[sd["name"].as<std::string>()] = YAML::Node(sd);

    std::unordered_map<std::string, YAML::Node> robot_devs;
    for (const auto& rd : sys_config["devices"])
        robot_devs[rd["name"].as<std::string>()] = YAML::Node(rd);

    for (auto& arm : arm_instances) {
        arm->robot->set_simulation(*sim_, sim_devs[arm->getDeviceName()], robot_devs[arm->getDeviceName()]);
        std::string gripper_name = "hand_" + arm->getDeviceName().substr(arm->getDeviceName().find('_') + 1);
        if (sim_devs.count(gripper_name)){
            arm->gripper->set_simulation(*sim_, gripper_name);
        }
    }
    for (auto& head : head_instances)
        head->module->set_simulation(*sim_, sim_devs[head->getDeviceName()]);

    current_episode_cfg_ = requestEpisodeConfig();
    applyEpisodeConfig(current_episode_cfg_);

    for (const auto& dev : sys_config["devices"]) {
        if (!dev["enabled"].as<bool>(true)) continue;
        if (dev["type"].as<std::string>() != "head") continue;
        if (!dev["camera"]) continue;

        auto cam = dev["camera"];
        auto pos = cam["position"].as<std::vector<double>>();
        auto eul = cam["euler_xyz"].as<std::vector<double>>();

        CameraExtrinsics extrinsics;
        extrinsics.position    = Eigen::Vector3d(pos[0], pos[1], pos[2]);
        extrinsics.orientation =
            Eigen::AngleAxisd(eul[2], Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(eul[1], Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(eul[0], Eigen::Vector3d::UnitX());

        CameraIntrinsics intrinsics = sim_->getCameraIntrinsics(cam["name"].as<std::string>());

        IntentionBufferConfig buf_cfg;
        buf_cfg.intrinsics = intrinsics;
        buf_cfg.extrinsics = extrinsics;

        intention_buffer_ = std::make_unique<IntentionBuffer>(buf_cfg);

        IntentionRecognizerConfig rec_cfg;
        rec_cfg.log_path   = log_base_dir_ + "/intention_log.csv";
        rec_cfg.session_id = session_id_;
        intention_recognizer_ = std::make_unique<IntentionRecognizer>(rec_cfg);

        intention_buffer_->setCallback([this](const IntentionSample& s) {
            if (intention_recognizer_) intention_recognizer_->push(s);
        });
        break;
    }
#endif
}

Avatar::~Avatar() {
    if (episode_sock_ != kInvalidSocket) close_socket(episode_sock_);
    if (logger_sock_  != kInvalidSocket) close_socket(logger_sock_);
}

void Avatar::start(){
    if (cmd_channel_) cmd_channel_->start();
    for(const auto& head : head_instances){
        head->start();
    }
    for(const auto& arm : arm_instances){
        arm->start();
    }
    std::cout << "[AVATAR-INFO]: All devices started" << std::endl;

    constexpr std::chrono::microseconds control_period(static_cast<int>(1e6 / 100));
	auto next_control_time = std::chrono::high_resolution_clock::now();

    auto loop_start_time = std::chrono::high_resolution_clock::now();

    SysState prev_state = SysState::IDLE;
    state_ = SysState::IDLE;
    if (cmd_channel_) cmd_channel_->setState(SysState::IDLE);
    auto last_heartbeat = std::chrono::steady_clock::now();
    auto start_time = std::chrono::steady_clock::now();
    if (cmd_channel_) cmd_channel_->resetAliveTimer();
    bRunning = true;
   
    while(bRunning){
        SysState cmd_state = cmd_requested_.load();

        if (cmd_channel_ && !cmd_channel_->isAlive()) {
            if (state_ != SysState::IDLE && state_ != SysState::OFFLINE) {
                std::cout << "[AVATAR-WARN]: Operator connection lost, reverting to IDLE." << std::endl;
                requestAllDevices(SysState::IDLE);
                state_ = SysState::IDLE;
                cmd_requested_.store(SysState::IDLE);
                cmd_channel_->setState(SysState::IDLE);
            }
        } else {
            updateStateMachine(cmd_state);
        }

        processRecoveryNotifications();
        processResetAllCompletion();

        if (state_ != prev_state && cmd_channel_) {
            cmd_channel_->setState(state_);
        }
        prev_state = state_;

        auto now = std::chrono::steady_clock::now();
        if (cmd_channel_ && std::chrono::duration_cast<std::chrono::milliseconds>(now - last_heartbeat).count() >= 500) {
            msgpack::sbuffer buf;
            msgpack::pack(buf, std::map<std::string, int64_t>{{"uptime_ms", std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count()}});
            cmd_channel_->send("heartbeat", buf, false);
            last_heartbeat = now;
        }
        #ifndef WITH_FRANKA
            for (auto& arm : arm_instances) {
                Eigen::Isometry3d T = arm->getTargetPose();
                std::string dev = arm->getDeviceName();
                std::string side = dev.substr(dev.find('_') + 1);
                sim_->setFramePose("target_" + side + "_frame", T.translation(), Eigen::Quaterniond(T.rotation()), 0.107);

                T = arm->getRawTargetPose();
                sim_->setFramePose("target_raw_" + side + "_frame", T.translation(), Eigen::Quaterniond(T.rotation()), 0.107);
            }

            if (intention_buffer_) {
                StateSnapshot snap;
                snap.frame_id    = sim_->getFrameId();
                snap.timestamp_ns = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count());

                for (auto& arm : arm_instances) {
                    std::string side = arm->getDeviceName().substr(arm->getDeviceName().find('_') + 1);
                    if (side == "left")  snap.T_ee_left  = arm->getTargetPose();
                    if (side == "right") snap.T_ee_right = arm->getTargetPose();
                }

                snap.gripper_left  = static_cast<float>(sim_->getGripperWidth("hand_left"));
                snap.gripper_right = static_cast<float>(sim_->getGripperWidth("hand_right"));

                DeviceState head_state = sim_->getDeviceState("head");
                if (head_state.q.size() >= 2) {
                    snap.head_pan  = static_cast<float>(head_state.q[0]);
                    snap.head_tilt = static_cast<float>(head_state.q[1]);
                }

                for (const auto& so : current_episode_cfg_.objects) {
                    auto it = std::find_if(object_defs_.begin(), object_defs_.end(),
                        [&](const ObjectDef& d){ return d.name == so.name; });
                    if (it == object_defs_.end()) continue;
                    Eigen::Vector3d p; Eigen::Quaterniond q;
                    if (!sim_->getFreeBodyPose(it->mujoco_body, p, q)) continue;
                    ObjectSlot s;
                    s.name    = so.name;
                    s.type    = SlotType::PICK_OBJ;
                    s.T_world = Eigen::Isometry3d::Identity();
                    s.T_world.translation() = p;
                    s.T_world.linear()      = q.toRotationMatrix();
                    snap.slots.push_back(std::move(s));
                }

                for (const auto& bin : bin_defs_) {
                    Eigen::Vector3d p; Eigen::Quaterniond q;
                    if (!sim_->getFreeBodyPose(bin.mujoco_body, p, q)) {
                        p = Eigen::Vector3d(bin.fixed_x, bin.fixed_y, bin.fixed_z);
                        q = Eigen::Quaterniond::Identity();
                    }
                    ObjectSlot s;
                    s.name         = bin.name;
                    s.type         = SlotType::PLACE_POSE;
                    s.T_world      = Eigen::Isometry3d::Identity();
                    s.T_world.translation() = p;
                    s.T_world.linear()      = q.toRotationMatrix();
                    s.half_extents = Eigen::Vector3d(0.120, 0.100, 0.145);
                    snap.slots.push_back(std::move(s));
                }

                intention_buffer_->snapshot(snap);
            }

            if (scene_logger_) {
                double t = std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - loop_start_time).count();
                SceneLogEntry entry{};
                entry.time = t;
                entry.mode = current_episode_cfg_.mode;
                entry.seed = current_episode_cfg_.seed;

                for (size_t i = 0; i < object_defs_.size() && i < SceneLogEntry::MAX_OBJECTS; ++i) {
                    Eigen::Vector3d p; Eigen::Quaterniond q;
                    if (sim_->getFreeBodyPose(object_defs_[i].mujoco_body, p, q)) {
                        entry.object_pos [i] = {p.x(), p.y(), p.z()};
                        entry.object_quat[i] = {q.w(), q.x(), q.y(), q.z()};
                    }
                    entry.object_names[i] = object_defs_[i].name;
                }
                entry.n_objects = static_cast<int>(std::min(object_defs_.size(),
                    static_cast<size_t>(SceneLogEntry::MAX_OBJECTS)));

                for (size_t i = 0; i < bin_defs_.size() && i < SceneLogEntry::MAX_BINS; ++i) {
                    Eigen::Vector3d p; Eigen::Quaterniond q;
                    if (!sim_->getFreeBodyPose(bin_defs_[i].mujoco_body, p, q)) {
                        p = Eigen::Vector3d(bin_defs_[i].fixed_x, bin_defs_[i].fixed_y, bin_defs_[i].fixed_z);
                        q = Eigen::Quaterniond::Identity();
                    }
                    entry.bin_pos [i] = {p.x(), p.y(), p.z()};
                    entry.bin_quat[i] = {q.w(), q.x(), q.y(), q.z()};
                    entry.bin_names[i] = bin_defs_[i].name;
                }
                entry.n_bins = static_cast<int>(std::min(bin_defs_.size(),
                    static_cast<size_t>(SceneLogEntry::MAX_BINS)));

                scene_logger_->write(entry);
            }
        #endif

        next_control_time += control_period;
        std::this_thread::sleep_until(next_control_time);
    }
}

void Avatar::stop(){
    bRunning = false;
    if (intention_recognizer_) intention_recognizer_->stop();
    if (scene_logger_) {
        scene_logger_->enable(false);
        scene_logger_->stop();
    }
    for(const auto& head : head_instances){
        head->stop();
    }
    for(const auto& arm : arm_instances){
        arm->stop();
    }
    if (cmd_channel_) cmd_channel_->stop();
}

void Avatar::processRecoveryNotifications() {
    if (!cmd_channel_) return;
    for (auto& arm : arm_instances) {
        if (arm->recovery().needsNotification()) {
            sendDeviceEvent(arm->getDeviceName(), "reset_complete");
            arm->recovery().clearNotification();
        }
    }
}

Avatar::EpisodeConfig Avatar::requestEpisodeConfig() {
    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port   = htons(9100);
    inet_pton(AF_INET, "127.0.0.1", &server_addr.sin_addr);

    msgpack::sbuffer req_buf;
    msgpack::pack(req_buf, std::map<std::string, std::string>{{"type", "request_episode_config"}});
    sendto(episode_sock_, req_buf.data(), static_cast<int>(req_buf.size()), 0,
           reinterpret_cast<sockaddr*>(&server_addr), static_cast<int>(sizeof(server_addr)));

    char recv_buf[8192];
    socklen_t addr_len = static_cast<socklen_t>(sizeof(server_addr));
    ssize_t n = recvfrom(episode_sock_, recv_buf, static_cast<int>(sizeof(recv_buf)), 0,
                         reinterpret_cast<sockaddr*>(&server_addr), &addr_len);

    EpisodeConfig cfg{};
    cfg.mode = 0;
    for (const auto& def : object_defs_) {
        SpawnedObject so;
        so.name       = def.name;
        so.color      = def.color;
        so.model_path = def.model_path;
        so.x = def.fixed_x; so.y = def.fixed_y; so.z = def.fixed_z;
        cfg.objects.push_back(so);
    }

    if (n <= 0) {
        std::cerr << "[AVATAR-WARN]: Episode config server unreachable, using defaults" << std::endl;
        return cfg;
    }

    try {
        auto oh = msgpack::unpack(recv_buf, n);
        auto root = oh.get();
        std::map<std::string, msgpack::object> fields;
        root.convert(fields);

        cfg.seed = fields.at("seed").as<int>();
        cfg.mode = fields.at("mode").as<int>();
        cfg.color_bin_mapping = fields.at("color_bin_mapping").as<std::string>();

        cfg.objects.clear();
        auto obj_list = fields.at("objects").as<std::vector<msgpack::object>>();
        for (const auto& item : obj_list) {
            std::map<std::string, msgpack::object> o;
            item.convert(o);
            SpawnedObject so;
            so.name       = o.at("name").as<std::string>();
            so.color      = o.at("color").as<std::string>();
            so.model_path = o.at("model_path").as<std::string>();
            so.x          = o.at("x").as<double>();
            so.y          = o.at("y").as<double>();
            so.z          = o.at("z").as<double>();
            cfg.objects.push_back(so);
        }
    } catch (const std::exception& e) {
        std::cerr << "[AVATAR-WARN]: Failed to parse episode config: " << e.what() << std::endl;
    }

    return cfg;
}

void Avatar::startNewEpisodeFolder() {
    char idx_buf[8];
    std::snprintf(idx_buf, sizeof(idx_buf), "%03d", episode_index_++);
    std::string folder = log_base_dir_ + "/" + std::string(idx_buf);
    std::filesystem::create_directories(folder);

    for (auto& arm : arm_instances) {
        std::string path = folder + "/" + arm->getDeviceName() + ".csv";
        arm->restartLogger(path);
    }
    for (auto& head : head_instances) {
        std::string path = folder + "/" + head->getDeviceName() + ".csv";
        head->restartLogger(path);
    }

    if (scene_logger_) {
        scene_logger_->stop();
        scene_logger_.reset();
    }
    scene_logger_ = std::make_unique<DataLogger<SceneLogEntry>>(
        folder + "/scene.csv", sceneLogHeader, sceneLogRow, session_id_);
    scene_logger_->start();

    if (intention_recognizer_)
        intention_recognizer_->restartLogger(folder + "/intention_log.csv");

    std::cout << "[AVATAR-INFO]: New episode folder: " << folder << std::endl;
}

void Avatar::applyEpisodeConfig(const EpisodeConfig& cfg) {
    std::cout << "[AVATAR-INFO]: Episode config seed=" << cfg.seed
              << " mode=" << (cfg.mode == 0 ? "unimanual" : "bimanual")
              << " n_objects=" << cfg.objects.size() << std::endl;

#ifndef WITH_FRANKA
    for (const auto& so : cfg.objects) {
        auto it = std::find_if(object_defs_.begin(), object_defs_.end(),
            [&](const ObjectDef& d){ return d.name == so.name; });
        if (it == object_defs_.end()) continue;
        sim_->setFreeBodyPose(it->mujoco_body,
            Eigen::Vector3d(so.x, so.y, so.z),
            Eigen::Quaterniond::Identity());
        std::cout << "[AVATAR-INFO]:   " << so.name << " (" << so.color
                  << ") -> (" << so.x << "," << so.y << "," << so.z << ")" << std::endl;
    }
#endif
}

void Avatar::processResetAllCompletion() {
    if (!reset_all_pending_.load()) return;

    for (auto& arm : arm_instances) {
        if (!arm->recovery().isWaitingAck()) return;
    }

    current_episode_cfg_ = requestEpisodeConfig();
    applyEpisodeConfig(current_episode_cfg_);

    startNewEpisodeFolder();

    for (auto& arm : arm_instances) {
        arm->reOrigin();
        arm->recovery().confirmResume();
        auto rec_it = device_records_.find(arm->getDeviceName());
        if (rec_it != device_records_.end())
            rec_it->second.active = true;
    }

    state_ = SysState::ENGAGED;
    cmd_requested_.store(SysState::ENGAGED);
    if (cmd_channel_) cmd_channel_->setState(SysState::ENGAGED);
    requestAllDevices(SysState::ENGAGED);
    if (scene_logger_) scene_logger_->enable(true);

    markEpisodeStart();

    for (auto& arm : arm_instances)
        arm->writeEpisodeConfig(current_episode_cfg_.seed, current_episode_cfg_.mode,
                                current_episode_cfg_.color_bin_mapping);

    reset_all_pending_.store(false);

    std::cout << "[AVATAR-INFO]: Global reset complete, awaiting engagement." << std::endl;
}

void Avatar::sendDeviceEvent(const std::string& device, const std::string& event) {
    msgpack::sbuffer buf;
    msgpack::pack(buf, std::map<std::string, std::string>{{"device", device}, {"event", event}});
    cmd_channel_->send("device_event", buf, true);
}

void Avatar::updateStateMachine(SysState cmd_state){
    if (reset_all_pending_.load()) return;

    if(cmd_state == SysState::STOP){
        state_ = SysState::STOP;
    }
    switch (state_) {
        case SysState::IDLE:
            if(cmd_state == SysState::HOMING){
                requestAllDevices(SysState::HOMING);
                state_ = SysState::HOMING;
                std::cout << "[AVATAR-INFO]: Homing." << std::endl;
            }
            break;

        case SysState::HOMING:
            if(cmd_state == SysState::IDLE){
                requestAllDevices(SysState::IDLE);
                state_ = SysState::IDLE;
            }
            else if(allInState(SysState::AWAITING)){
                state_ = SysState::AWAITING;
                std::cout << "[AVATAR-INFO]: Awaiting engagement." << std::endl;
            }
            break;

        case SysState::AWAITING:
            if(cmd_state == SysState::IDLE){
                requestAllDevices(SysState::IDLE);
                state_ = SysState::IDLE;
            }
            else if(cmd_state == SysState::ENGAGED && allInState(SysState::AWAITING)){
                requestAllDevices(SysState::ENGAGED);
                state_ = SysState::ENGAGED;
                startNewEpisodeFolder();
                markEpisodeStart();
                if (scene_logger_) scene_logger_->enable(true);
                std::cout << "[AVATAR-INFO]: Engage system." << std::endl;
            }
            break;

        case SysState::ENGAGED:
            if(cmd_state == SysState::IDLE){
                requestAllDevices(SysState::IDLE);
                state_ = SysState::IDLE;
                if (scene_logger_) scene_logger_->enable(false);
                markEpisodeEnd("operator_idle");
                std::cout << "[AVATAR-INFO]: Switch engage -> idle." << std::endl;
            }
            else if(cmd_state == SysState::PAUSED){
                requestAllDevices(SysState::PAUSED);
                state_ = SysState::PAUSED;
                if (scene_logger_) scene_logger_->enable(false);
                markEpisodeEnd("operator_pause");
                std::cout << "[AVATAR-INFO]: Switch engage -> pause." << std::endl;
            }
            break;
            
        case SysState::PAUSED:
            if(cmd_state == SysState::IDLE){
                requestAllDevices(SysState::IDLE);
                state_ = SysState::IDLE;
                std::cout << "[AVATAR-INFO]: Switch pause -> idle." << std::endl;
            }
            else if(cmd_state == SysState::ENGAGED && allInState(SysState::PAUSED)){
                requestAllDevices(SysState::ENGAGED);
                state_ = SysState::ENGAGED;
                std::cout << "[AVATAR-INFO]: Switch pause -> engage." << std::endl;
            }
            break;

        default:
            break;
    }
}

bool Avatar::allInState(SysState state) {
    for (auto& arm  : arm_instances) {
        auto it = device_records_.find(arm->getDeviceName());
        if (it != device_records_.end() && !it->second.active) continue;
        if (arm->getState() != state) return false;
    }
    for (auto& head : head_instances) {
        auto it = device_records_.find(head->getDeviceName());
        if (it != device_records_.end() && !it->second.active) continue;
        if (head->getState() != state) return false;
    }
    return true;
}

bool Avatar::anyoneInState(SysState state) {
    for (auto& arm  : arm_instances) {
        auto it = device_records_.find(arm->getDeviceName());
        if (it != device_records_.end() && !it->second.active) continue;
        if (arm->getState() == state) return true;
    }
    for (auto& head : head_instances) {
        auto it = device_records_.find(head->getDeviceName());
        if (it != device_records_.end() && !it->second.active) continue;
        if (head->getState() == state) return true;
    }
    return false;
}

void Avatar::requestAllDevices(SysState state) {
    for (auto& arm  : arm_instances) {
        auto it = device_records_.find(arm->getDeviceName());
        if (it != device_records_.end() && !it->second.active) continue;
        arm->requestState(state);
    }
    for (auto& head : head_instances) {
        auto it = device_records_.find(head->getDeviceName());
        if (it != device_records_.end() && !it->second.active) continue;
        head->requestState(state);
    }
}

ArmControl* Avatar::getArm(const std::string& name) {
    for (auto& arm : arm_instances)
        if (arm->getDeviceName() == name) return arm;
    return nullptr;
}

void Avatar::markEpisodeStart() {
    for (auto& arm  : arm_instances) arm->markEpisodeStart();
    for (auto& head : head_instances) head->markEpisodeStart();
    current_episode_idx_ = episode_index_ - 1;
    sendEpisodeEvent("episode_start", "");
}

void Avatar::markEpisodeEnd(const std::string& reason) {
    for (auto& arm  : arm_instances) arm->markEpisodeEnd(reason);
    for (auto& head : head_instances) head->markEpisodeEnd(reason);

    if (current_episode_idx_ >= 0) {
        sendEpisodeEvent("episode_end", reason);
        current_episode_idx_ = -1;
    }
}

void Avatar::sendEpisodeEvent(const std::string& type, const std::string& reason) {
    if (logger_sock_ == kInvalidSocket || logger_port_ == 0) return;

    EpisodeEventMsg msg;
    msg.type           = type;
    msg.session_id     = session_id_;
    msg.episode_index  = current_episode_idx_;
    msg.reason         = reason;

    msgpack::sbuffer buf;
    msgpack::pack(buf, msg);

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(static_cast<uint16_t>(logger_port_));
    inet_pton(AF_INET, logger_host_.c_str(), &addr.sin_addr);

    sendto(logger_sock_,
           buf.data(), static_cast<int>(buf.size()),
           0,
           reinterpret_cast<sockaddr*>(&addr),
           static_cast<int>(sizeof(addr)));
}