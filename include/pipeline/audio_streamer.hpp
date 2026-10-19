#pragma once

#include "network/platform_socket.hpp"

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

#include <gst/gst.h>

// Wire format, little-endian, one Opus packet per datagram:
//   u32 magic 'TAUD' | u8 version=1 | u8 channels | u8 frame_ms | u8 reserved
//   u32 seq | u64 capture_ns (system_clock, ns since Unix epoch) | Opus payload
struct AudioStreamerConfig {
    bool        enabled  = false;
    std::string device   = "Maono";
    std::string host;
    int         port     = 5020;
    int         bitrate  = 64000;
    int         frame_ms = 10;
    int         channels = 1;
    int         retry_s  = 5;
};

class AudioStreamer {
public:
    explicit AudioStreamer(const AudioStreamerConfig& config);
    ~AudioStreamer();

    void start();
    void stop();

private:
    void run();
    bool open();
    void close();
    bool pump();
    GstElement* createSource(std::string& name_out);
    void send(const uint8_t* data, size_t size, uint64_t capture_ns);

    AudioStreamerConfig  config_;
    GstElement*          pipeline_ = nullptr;
    GstElement*          appsink_  = nullptr;
    GstBus*              bus_      = nullptr;
    socket_t             sock_     = kInvalidSocket;
    uint32_t             seq_      = 0;
    std::vector<uint8_t> packet_;
    std::thread          thread_;
    std::atomic<bool>    running_{false};
};
