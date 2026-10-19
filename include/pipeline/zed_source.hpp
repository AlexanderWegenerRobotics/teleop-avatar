#pragma once

#include "pipeline/camera_source.hpp"

#include <sl/Camera.hpp>

#include <atomic>
#include <string>
#include <thread>
#include <vector>

class ZedSource : public CameraSource {
public:
    ZedSource(int serial, const std::string& resolution, int fps,
              bool auto_exposure, int exposure_100us, int gain);
    ~ZedSource();

    void start(FrameCallback cb) override;
    void stop()                  override;
    uint32_t width()  const      override { return width_; }
    uint32_t height() const      override { return height_; }
    bool     bgra()   const      override { return true; }

private:
    void run();

    sl::Camera           zed_;
    sl::Mat              sbs_;
    std::vector<uint8_t> packed_;
    uint32_t             width_  = 0;
    uint32_t             height_ = 0;
    int                  fps_;

    FrameCallback        cb_;
    std::thread          thread_;
    std::atomic<bool>    bRunning_{false};
};
