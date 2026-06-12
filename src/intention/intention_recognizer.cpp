#include "intention/intention_recognizer.hpp"

#include <iostream>

IntentionRecognizer::IntentionRecognizer(const IntentionRecognizerConfig& config)
    : config_(config)
    , start_time_(std::chrono::high_resolution_clock::now())
{
    logger_ = std::make_unique<DataLogger<IntentionLogEntry>>(
        config_.log_path, intentionLogHeader, intentionLogRow, config_.session_id);
    logger_->start();
    logger_->enable(true);
}

void IntentionRecognizer::push(const IntentionSample& sample) {
    std::lock_guard<std::mutex> lock(mtx_);

    window_.push_back(sample);
    if (static_cast<int>(window_.size()) > config_.window_frames)
        window_.pop_front();

    double t = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - start_time_).count();

    IntentionLogEntry entry{};
    entry.time                 = t;
    entry.frame_id             = sample.frame_id;
    entry.timestamp_ns         = sample.timestamp_ns;
    entry.timestamp_arrival_ns = sample.timestamp_arrival_ns;
    entry.gaze_valid           = static_cast<uint8_t>(sample.gaze_valid);
    entry.gaze_px_x     = sample.gaze_px_x;
    entry.gaze_px_y     = sample.gaze_px_y;
    entry.gripper_left  = sample.gripper_left;
    entry.gripper_right = sample.gripper_right;
    entry.n_slots       = static_cast<uint8_t>(std::min(sample.slot_types.size(), size_t(10)));

    entry.slot_belief.fill(0.0f);
    for (size_t i = 0; i < std::min(sample.slot_belief.size(), size_t(11)); ++i)
        entry.slot_belief[i] = sample.slot_belief[i];

    entry.slot_types.fill(0);
    for (size_t i = 0; i < std::min(sample.slot_types.size(), size_t(10)); ++i)
        entry.slot_types[i] = sample.slot_types[i];

    entry.slot_px_u.fill(-1.0f);
    entry.slot_px_v.fill(-1.0f);
    for (size_t i = 0; i < std::min(sample.slot_px_u.size(), size_t(10)); ++i) {
        entry.slot_px_u[i] = sample.slot_px_u[i];
        entry.slot_px_v[i] = sample.slot_px_v[i];
    }

    auto lp = sample.T_ee_left.translation();
    entry.ee_left_pos  = {lp.x(), lp.y(), lp.z()};
    auto rp = sample.T_ee_right.translation();
    entry.ee_right_pos = {rp.x(), rp.y(), rp.z()};

    entry.slot_distances.fill(0.0f);
    for (size_t i = 0; i < std::min(sample.slot_distances.size(), size_t(20)); ++i)
        entry.slot_distances[i] = sample.slot_distances[i];

    logger_->write(entry);
}

void IntentionRecognizer::stop() {
    logger_->stop();
}

void IntentionRecognizer::restartLogger(const std::string& path) {
    start_time_ = std::chrono::high_resolution_clock::now();
    logger_->restart(path);
}

void IntentionRecognizer::markEpisodeStart() {
    if (logger_) logger_->markEpisodeStart();
}

void IntentionRecognizer::markEpisodeEnd(const std::string& reason) {
    if (logger_) logger_->markEpisodeEnd(reason);
}

void IntentionRecognizer::writeEpisodeConfig(int seed, int mode, const std::string& color_bin_mapping) {
    if (logger_) logger_->writeEpisodeConfig(seed, mode, color_bin_mapping);
}
