#pragma once

#include <chrono>
#include <deque>
#include <memory>
#include <mutex>

#include "data_logger.hpp"
#include "intention/intention_sample.hpp"

struct IntentionRecognizerConfig {
    int         window_frames = 60;
    std::string log_path      = "../log/intention_log.csv";
    std::string session_id    = "";
};

class IntentionRecognizer {
public:
    explicit IntentionRecognizer(const IntentionRecognizerConfig& config);
    void push(const IntentionSample& sample);
    void stop();
    void restartLogger(const std::string& path);
    void markEpisodeStart();
    void markEpisodeEnd(const std::string& reason);
    void writeEpisodeConfig(int seed, int mode, const std::string& color_bin_mapping);

private:
    IntentionRecognizerConfig                      config_;
    std::mutex                                     mtx_;
    std::deque<IntentionSample>                    window_;
    std::unique_ptr<DataLogger<IntentionLogEntry>> logger_;
    std::chrono::high_resolution_clock::time_point start_time_;
};
