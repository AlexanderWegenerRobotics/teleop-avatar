#include "pipeline/video_streamer.hpp"

#include <stdexcept>
#include <iostream>
#include <cstring>
#include <chrono>

#include <gst/app/gstappsrc.h>

VideoStreamer::VideoStreamer(const StreamerConfig& config)
    : config_(config)
{
    target_fps_.store(config_.fps);
}

VideoStreamer::~VideoStreamer() {
    stop();
}

void VideoStreamer::start() {
    buildPipeline();

    loop_ = g_main_loop_new(nullptr, FALSE);
    loop_thread_ = std::thread([this]() { g_main_loop_run(loop_); });

    StreamQualityConfig qc_config{};
    qc_config.listen_port    = config_.feedback_port;
    qc_config.bitrate_normal = config_.bitrate_kbps;
    qc_config.fps_normal     = config_.fps;
    qc_config.fec_normal     = config_.fec_percentage;
    qc_config.status_host        = config_.host;
    qc_config.status_port        = config_.status_port;
    qc_config.status_interval_ms = config_.status_interval_ms;

    quality_ = std::make_unique<StreamQualityController>(qc_config);
    quality_->setOnQualityChange([this](const StreamQualityParams& params) {
        if (encoder_)
            g_object_set(G_OBJECT(encoder_), "bitrate", params.bitrate_kbps, nullptr);
        if (fec_)
            g_object_set(G_OBJECT(fec_), "percentage", params.fec_percentage, nullptr);
        target_fps_.store(params.fps);
    });
    quality_->start();

    GstBus* bus = gst_element_get_bus(pipeline_);
    gst_bus_add_watch(bus, [](GstBus*, GstMessage* msg, gpointer) -> gboolean {
        if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
            GError* err; gchar* dbg;
            gst_message_parse_error(msg, &err, &dbg);
            std::cerr << "[GST ERROR] " << err->message << std::endl;
            std::cerr << "[GST DEBUG] " << (dbg ? dbg : "none") << std::endl;
            g_error_free(err);
            g_free(dbg);
        }
        return TRUE;
    }, nullptr);
    gst_object_unref(bus);

    gst_element_set_state(pipeline_, GST_STATE_PLAYING);

    bRunning_ = true;

    std::cout << "[INFO] Streamer running on "
              << config_.host << ":" << config_.port
              << " " << config_.stream_width << "x" << config_.stream_height
              << "+2 (timestamp rows) @ " << config_.fps << "fps" << std::endl;
}

void VideoStreamer::stop() {
    bRunning_ = false;
    if (quality_) quality_->stop();
    stopEncodedLog();

    if (pipeline_) {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
        appsrc_   = nullptr;
        encoder_  = nullptr;
        fec_      = nullptr;
        logsink_  = nullptr;
    }

    if (loop_) {
        g_main_loop_quit(loop_);
        if (loop_thread_.joinable()) loop_thread_.join();
        g_main_loop_unref(loop_);
        loop_ = nullptr;
    }
}

static bool nvencAvailable() {
    GstElementFactory* f = gst_element_factory_find("nvh264enc");
    if (f) { gst_object_unref(f); return true; }
    return false;
}

// True when the CUDA colour-convert elements exist, so RGB→NV12 can run on the GPU.
static bool cudaConvertAvailable() {
    GstElementFactory* up = gst_element_factory_find("cudaupload");
    GstElementFactory* cv = gst_element_factory_find("cudaconvert");
    bool ok = (up != nullptr) && (cv != nullptr);
    if (up) gst_object_unref(up);
    if (cv) gst_object_unref(cv);
    return ok;
}

void VideoStreamer::buildPipeline() {
    // Height + 2: the extra two rows carry the embedded wall-clock timestamp and frame ID.
    // The receiver reads and crops them before display.
    int padded_height = config_.stream_height + 2;

    // GPU encoder (nvh264enc) when present, else CPU (x264enc); with CUDA convert elements
    // the RGB→NV12 colour convert also runs on the GPU. config-interval=1 repeats SPS/PPS
    // in-band; byte-stream caps keep the encoder output compatible with rtph264pay and make
    // the optional logging branch a decodable Annex-B .h264 elementary stream.
    auto build = [&](bool gpu, bool cuda_convert) -> std::string {
        std::string convert_seg = !gpu
            ? std::string(" ! videoconvert ! video/x-raw,format=I420")
            : (cuda_convert
                ? std::string(" ! cudaupload ! cudaconvert ! video/x-raw(memory:CUDAMemory),format=NV12")
                : std::string(" ! videoconvert ! video/x-raw,format=NV12"));

        std::string encoder_seg = gpu
            ? std::string(
                " ! nvh264enc name=encoder preset=low-latency-hq rc-mode=cbr gop-size=30"
                " bitrate=" + std::to_string(config_.bitrate_kbps))
            : std::string(
                " ! x264enc name=encoder tune=zerolatency speed-preset=ultrafast"
                " bitrate=" + std::to_string(config_.bitrate_kbps) +
                " key-int-max=30");
        encoder_seg += " ! video/x-h264,stream-format=byte-stream,alignment=au";

        std::string sink_seg =
            " ! rtph264pay pt=96 config-interval=1"
            " ! rtpulpfecenc name=fec percentage=" + std::to_string(config_.fec_percentage) +
            " ! udpsink host=" + config_.host +
            " port="           + std::to_string(config_.port);

        std::string tail = config_.log_enabled
            ? std::string(
                " ! tee name=enc_tee"
                " enc_tee. ! queue") + sink_seg +
                " enc_tee. ! queue ! appsink name=logsink emit-signals=true sync=false max-buffers=4 drop=true"
            : sink_seg;

        return
            "appsrc name=src stream-type=0 format=3 is-live=true block=false"
            " caps=video/x-raw,format=RGB"
            ",width="      + std::to_string(config_.stream_width)  +
            ",height="     + std::to_string(padded_height)          +
            ",framerate="  + std::to_string(config_.fps) + "/1"
            " ! queue max-size-buffers=2 leaky=downstream"
            + convert_seg + encoder_seg + tail;
    };

    GError* err = nullptr;
    auto try_build = [&](bool gpu, bool cuda_convert, const char* label) -> bool {
        std::string s = build(gpu, cuda_convert);
        std::cout << "[INFO] Encoder: " << label << std::endl;
        std::cout << "[INFO] Pipeline: " << s << std::endl;
        pipeline_ = gst_parse_launch(s.c_str(), &err);
        if (err) {
            std::cerr << "[WARN] pipeline build failed (" << err->message << ")." << std::endl;
            g_error_free(err);
            err = nullptr;
            if (pipeline_) { gst_object_unref(pipeline_); pipeline_ = nullptr; }
            return false;
        }
        return true;
    };

    bool built = false;
    if (nvencAvailable()) {
        if (cudaConvertAvailable()) built = try_build(true, true,  "nvh264enc (GPU) + cudaconvert");
        if (!built)                 built = try_build(true, false, "nvh264enc (GPU) + videoconvert");
    }
    if (!built && !try_build(false, false, "x264enc (CPU)"))
        throw std::runtime_error("GStreamer pipeline error: CPU fallback failed to build");

    appsrc_ = gst_bin_get_by_name(GST_BIN(pipeline_), "src");
    if (!appsrc_)
        throw std::runtime_error("Failed to get appsrc element from pipeline");

    encoder_ = gst_bin_get_by_name(GST_BIN(pipeline_), "encoder");
    if (!encoder_)
        throw std::runtime_error("Failed to get encoder element from pipeline");

    fec_ = gst_bin_get_by_name(GST_BIN(pipeline_), "fec");
    if (!fec_)
        throw std::runtime_error("Failed to get fec element from pipeline");

    if (config_.log_enabled) {
        logsink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "logsink");
        if (logsink_)
            g_signal_connect(logsink_, "new-sample", G_CALLBACK(&VideoStreamer::onNewSample), this);
    }

    gst_app_src_set_stream_type(GST_APP_SRC(appsrc_), GST_APP_STREAM_TYPE_STREAM);
    gst_app_src_set_latency(GST_APP_SRC(appsrc_), 0, -1);
    gst_app_src_set_max_bytes(GST_APP_SRC(appsrc_), 0);
}

// Appsink callback (streaming thread): appends each encoded access unit to the open episode file.
GstFlowReturn VideoStreamer::onNewSample(GstAppSink* sink, gpointer user) {
    VideoStreamer* self = static_cast<VideoStreamer*>(user);
    GstSample* sample = gst_app_sink_pull_sample(sink);
    if (!sample) return GST_FLOW_OK;

    GstBuffer* buf = gst_sample_get_buffer(sample);
    GstMapInfo map;
    if (buf && gst_buffer_map(buf, &map, GST_MAP_READ)) {
        std::lock_guard<std::mutex> lk(self->enc_mutex_);
        if (self->enc_file_) std::fwrite(map.data, 1, map.size, self->enc_file_);
        gst_buffer_unmap(buf, &map);
    }
    gst_sample_unref(sample);
    return GST_FLOW_OK;
}

// Opens a per-episode file that the appsink callback appends the encoded H.264 stream to.
void VideoStreamer::startEncodedLog(const std::string& path) {
    std::lock_guard<std::mutex> lk(enc_mutex_);
    if (enc_file_) { std::fclose(enc_file_); enc_file_ = nullptr; }
    enc_file_ = std::fopen(path.c_str(), "wb");
    if (!enc_file_)
        std::cerr << "[VideoStreamer] failed to open encoded log: " << path << std::endl;
    else
        std::cout << "[VideoStreamer] Encoded log started -> " << path << std::endl;
}

// Closes the current episode's encoded file.
void VideoStreamer::stopEncodedLog() {
    std::lock_guard<std::mutex> lk(enc_mutex_);
    if (enc_file_) { std::fclose(enc_file_); enc_file_ = nullptr; }
}

void VideoStreamer::pushFrame(const uint8_t* rgb, uint32_t width, uint32_t height) {
    if (!appsrc_ || !bRunning_) return;

    int tfps = target_fps_.load();
    if (tfps > 0 && tfps < config_.fps) {
        int skip = config_.fps / tfps;
        if (frame_count_ % skip != 0) {
            frame_count_++;
            return;
        }
    }

    size_t row_bytes    = width * 3;
    size_t image_bytes  = row_bytes * height;
    size_t padded_bytes = image_bytes + row_bytes * 2;

    GstBuffer* buffer = gst_buffer_new_allocate(nullptr, padded_bytes, nullptr);
    if (!buffer) return;

    GST_BUFFER_PTS(buffer)      = frame_count_ * GST_SECOND / config_.fps;
    GST_BUFFER_DURATION(buffer) = GST_SECOND / config_.fps;

    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_WRITE)) {
        gst_buffer_unref(buffer);
        return;
    }

    std::memcpy(map.data, rgb, image_bytes);

    uint8_t* ts_row = map.data + image_bytes;
    std::memset(ts_row, 0, row_bytes * 2);

    uint64_t now_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());

    for (int bit = 0; bit < 64; ++bit) {
        uint8_t val = ((now_ns >> bit) & 1) ? 255 : 0;
        ts_row[bit * 3 + 0] = val;
        ts_row[bit * 3 + 1] = val;
        ts_row[bit * 3 + 2] = val;
    }

    uint8_t* id_row = ts_row + row_bytes;
    for (int bit = 0; bit < 64; ++bit) {
        uint8_t val = ((frame_count_ >> bit) & 1) ? 255 : 0;
        id_row[bit * 3 + 0] = val;
        id_row[bit * 3 + 1] = val;
        id_row[bit * 3 + 2] = val;
    }

    gst_buffer_unmap(buffer, &map);

    gst_app_src_push_buffer(GST_APP_SRC(appsrc_), buffer);
    frame_count_++;
}
