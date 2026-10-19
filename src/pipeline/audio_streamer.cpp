#include "pipeline/audio_streamer.hpp"

#include <gst/app/gstappsink.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>

namespace {

constexpr uint32_t kMagic      = 0x44554154;
constexpr uint8_t  kVersion    = 1;
constexpr size_t   kHeaderSize = 20;

uint64_t wallNs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

const char* factoryName(GstElement* e) {
    GstElementFactory* f = gst_element_get_factory(e);
    return f ? gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(f)) : "";
}

}

AudioStreamer::AudioStreamer(const AudioStreamerConfig& config) : config_(config) {}

AudioStreamer::~AudioStreamer() { stop(); }

void AudioStreamer::start() {
    if (!config_.enabled || running_) return;

    sock_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock_ == kInvalidSocket) {
        std::cerr << "[Audio] socket creation failed, audio disabled" << std::endl;
        return;
    }
    sockaddr_in dest{};
    dest.sin_family = AF_INET;
    dest.sin_port   = htons(static_cast<uint16_t>(config_.port));
    if (inet_pton(AF_INET, config_.host.c_str(), &dest.sin_addr) != 1 ||
        connect(sock_, reinterpret_cast<sockaddr*>(&dest), sizeof(dest)) < 0) {
        std::cerr << "[Audio] invalid destination " << config_.host << ":" << config_.port
                  << ", audio disabled" << std::endl;
        close_socket(sock_);
        sock_ = kInvalidSocket;
        return;
    }

    running_ = true;
    thread_  = std::thread(&AudioStreamer::run, this);
}

void AudioStreamer::stop() {
    running_ = false;
    if (thread_.joinable()) thread_.join();
    close();
    if (sock_ != kInvalidSocket) {
        close_socket(sock_);
        sock_ = kInvalidSocket;
    }
}

GstElement* AudioStreamer::createSource(std::string& name_out) {
    GstDeviceMonitor* mon = gst_device_monitor_new();
    gst_device_monitor_add_filter(mon, "Audio/Source", nullptr);
    GList* devices = gst_device_monitor_get_devices(mon);

    const std::string want = lower(config_.device);
    GstElement* best = nullptr;
    for (GList* l = devices; l; l = l->next) {
        GstDevice* dev = GST_DEVICE(l->data);
        gchar* dn = gst_device_get_display_name(dev);
        const std::string name = dn ? dn : "";
        g_free(dn);
        if (!want.empty() && lower(name).find(want) == std::string::npos) continue;

        GstElement* e = gst_device_create_element(dev, "audio_src");
        if (!e) continue;
        gboolean loopback = FALSE;
        if (g_object_class_find_property(G_OBJECT_GET_CLASS(e), "loopback"))
            g_object_get(e, "loopback", &loopback, nullptr);
        if (loopback) {
            gst_object_unref(e);
            continue;
        }
        const bool preferred = std::strcmp(factoryName(e), "wasapi2src") == 0;
        if (!best || preferred) {
            if (best) gst_object_unref(best);
            best = e;
            name_out = name;
            if (preferred) break;
        } else {
            gst_object_unref(e);
        }
    }
    g_list_free_full(devices, gst_object_unref);
    gst_object_unref(mon);

    if (best && g_object_class_find_property(G_OBJECT_GET_CLASS(best), "low-latency"))
        g_object_set(best, "low-latency", TRUE, nullptr);
    return best;
}

bool AudioStreamer::open() {
    std::string device_name;
    GstElement* src = createSource(device_name);
    if (!src) return false;

    const std::string desc =
        "queue max-size-time=100000000 max-size-buffers=0 max-size-bytes=0 leaky=downstream"
        " ! audioconvert ! audioresample"
        " ! audio/x-raw,rate=48000,channels=" + std::to_string(config_.channels) +
        " ! level name=mic_level interval=5000000000 post-messages=true"
        " ! opusenc bitrate=" + std::to_string(config_.bitrate) +
        " frame-size=" + std::to_string(config_.frame_ms) +
        " audio-type=restricted-lowdelay inband-fec=true"
        " ! appsink name=asink sync=false emit-signals=false max-buffers=100 drop=true";

    GError* err = nullptr;
    GstElement* tail = gst_parse_bin_from_description(desc.c_str(), TRUE, &err);
    if (!tail) {
        std::cerr << "[Audio] pipeline error: " << (err ? err->message : "?") << std::endl;
        if (err) g_error_free(err);
        gst_object_unref(src);
        return false;
    }

    pipeline_ = gst_pipeline_new("audio_pipeline");
    gst_bin_add_many(GST_BIN(pipeline_), src, tail, nullptr);
    if (!gst_element_link(src, tail)) {
        std::cerr << "[Audio] cannot link microphone '" << device_name << "'" << std::endl;
        close();
        return false;
    }
    appsink_ = gst_bin_get_by_name(GST_BIN(tail), "asink");
    bus_     = gst_element_get_bus(pipeline_);

    if (gst_element_set_state(pipeline_, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE ||
        gst_element_get_state(pipeline_, nullptr, nullptr, 3 * GST_SECOND) == GST_STATE_CHANGE_FAILURE) {
        std::cerr << "[Audio] microphone '" << device_name << "' failed to start" << std::endl;
        close();
        return false;
    }

    std::cout << "[Audio] streaming '" << device_name << "' (" << factoryName(src) << ") -> "
              << config_.host << ":" << config_.port << "  opus " << config_.bitrate / 1000
              << " kbps, " << config_.frame_ms << " ms frames, " << config_.channels << " ch" << std::endl;
    return true;
}

void AudioStreamer::close() {
    if (pipeline_) gst_element_set_state(pipeline_, GST_STATE_NULL);
    if (appsink_)  { gst_object_unref(appsink_);  appsink_  = nullptr; }
    if (bus_)      { gst_object_unref(bus_);      bus_      = nullptr; }
    if (pipeline_) { gst_object_unref(pipeline_); pipeline_ = nullptr; }
}

bool AudioStreamer::pump() {
    GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(appsink_), 50 * GST_MSECOND);
    if (sample) {
        GstBuffer* buf = gst_sample_get_buffer(sample);
        GstClock*  clock = gst_element_get_clock(pipeline_);
        if (buf && clock && GST_BUFFER_PTS_IS_VALID(buf)) {
            const uint64_t     now_wall = wallNs();
            const GstClockTime running  = gst_clock_get_time(clock) - gst_element_get_base_time(pipeline_);
            const int64_t      age      = static_cast<int64_t>(running) - static_cast<int64_t>(GST_BUFFER_PTS(buf));
            const uint64_t     capture  = now_wall - static_cast<uint64_t>(std::max<int64_t>(age, 0));
            GstMapInfo map;
            if (gst_buffer_map(buf, &map, GST_MAP_READ)) {
                send(map.data, map.size, capture);
                gst_buffer_unmap(buf, &map);
            }
        }
        if (clock) gst_object_unref(clock);
        gst_sample_unref(sample);
    }

    while (GstMessage* msg = gst_bus_pop_filtered(bus_,
               static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS | GST_MESSAGE_ELEMENT))) {
        bool fatal = false;
        if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
            GError* err = nullptr; gchar* dbg = nullptr;
            gst_message_parse_error(msg, &err, &dbg);
            std::cerr << "[Audio] capture error: " << (err ? err->message : "?") << std::endl;
            if (err) g_error_free(err);
            g_free(dbg);
            fatal = true;
        } else if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_EOS) {
            fatal = true;
        } else {
            const GstStructure* s = gst_message_get_structure(msg);
            if (s && gst_structure_has_name(s, "level")) {
                const GValue* peaks = gst_structure_get_value(s, "peak");
                GValueArray*  arr   = peaks ? static_cast<GValueArray*>(g_value_get_boxed(peaks)) : nullptr;
                if (arr && arr->n_values > 0)
                    std::cout << "[Audio] mic peak " << g_value_get_double(g_value_array_get_nth(arr, 0))
                              << " dB, " << seq_ << " packets sent" << std::endl;
            }
        }
        gst_message_unref(msg);
        if (fatal) return false;
    }
    return true;
}

void AudioStreamer::send(const uint8_t* data, size_t size, uint64_t capture_ns) {
    packet_.resize(kHeaderSize + size);
    uint8_t* p = packet_.data();
    std::memcpy(p, &kMagic, 4);
    p[4] = kVersion;
    p[5] = static_cast<uint8_t>(config_.channels);
    p[6] = static_cast<uint8_t>(config_.frame_ms);
    p[7] = 0;
    std::memcpy(p + 8,  &seq_, 4);
    std::memcpy(p + 12, &capture_ns, 8);
    std::memcpy(p + kHeaderSize, data, size);
    ::send(sock_, reinterpret_cast<const char*>(p), static_cast<int>(packet_.size()), 0);
    ++seq_;
}

void AudioStreamer::run() {
    using clock = std::chrono::steady_clock;
    auto next_try = clock::now();
    bool warned   = false;

    while (running_) {
        if (!pipeline_) {
            if (clock::now() < next_try) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            if (!open()) {
                if (!warned)
                    std::cerr << "[Audio] no usable microphone matching '" << config_.device
                              << "', video continues without audio, retrying every "
                              << config_.retry_s << " s" << std::endl;
                warned   = true;
                next_try = clock::now() + std::chrono::seconds(config_.retry_s);
                continue;
            }
            warned = false;
        }
        if (!pump()) {
            close();
            std::cerr << "[Audio] microphone lost, retrying every " << config_.retry_s << " s" << std::endl;
            warned   = true;
            next_try = clock::now() + std::chrono::seconds(config_.retry_s);
        }
    }
}
