#ifdef WITH_V4L2

#include "pipeline/v4l2_source.hpp"

#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>

#include <cstring>
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <vector>

#ifdef WITH_TURBOJPEG
#include <turbojpeg.h>
#endif

static int xioctl(int fd, unsigned long req, void* arg) {
    // ioctl wrapper that retries on EINTR.
    int r;
    do { r = ioctl(fd, req, arg); } while (r == -1 && errno == EINTR);
    return r;
}

static void yuyvToRgb(const uint8_t* yuyv, uint8_t* rgb, int width, int height) {
    // Convert YUYV packed to interleaved RGB888.
    int n = width * height / 2;
    for (int i = 0; i < n; ++i) {
        int y0 = yuyv[0], u = yuyv[1], y1 = yuyv[2], v = yuyv[3];
        yuyv += 4;

        auto clamp = [](int x) -> uint8_t {
            return x < 0 ? 0 : x > 255 ? 255 : static_cast<uint8_t>(x);
        };

        int c0 = y0 - 16, c1 = y1 - 16;
        int d  = u  - 128, e  = v   - 128;

        rgb[0] = clamp((298 * c0 + 409 * e + 128) >> 8);
        rgb[1] = clamp((298 * c0 - 100 * d - 208 * e + 128) >> 8);
        rgb[2] = clamp((298 * c0 + 516 * d + 128) >> 8);
        rgb[3] = clamp((298 * c1 + 409 * e + 128) >> 8);
        rgb[4] = clamp((298 * c1 - 100 * d - 208 * e + 128) >> 8);
        rgb[5] = clamp((298 * c1 + 516 * d + 128) >> 8);
        rgb += 6;
    }
}

V4L2Source::V4L2Source(const std::string& device, int width, int height, int fps,
                       const std::string& format, int exposure_100us)
    : device_(device), width_(width), height_(height), fps_(fps), mjpeg_(format == "mjpeg"),
      exposure_100us_(exposure_100us)
{
#ifndef WITH_TURBOJPEG
    if (mjpeg_)
        throw std::runtime_error("[V4L2Source] v4l2_format mjpeg needs libturbojpeg (apt install libturbojpeg0-dev, rebuild)");
#endif
    std::cout << "[V4L2Source] configured " << device_
              << " " << width_ << "x" << height_ << " @ " << fps_ << "fps "
              << (mjpeg_ ? "MJPEG" : "YUYV") << std::endl;
}

V4L2Source::~V4L2Source() {
    // Stop thread and release device resources.
    stop();
}

void V4L2Source::initDevice() {
    // Open device, set format, map buffers, start streaming.
    fd_ = open(device_.c_str(), O_RDWR);
    if (fd_ < 0)
        throw std::runtime_error("[V4L2Source] cannot open " + device_ + ": " + strerror(errno));

    v4l2_format fmt{};
    fmt.type                = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width       = static_cast<uint32_t>(width_);
    fmt.fmt.pix.height      = static_cast<uint32_t>(height_);
    fmt.fmt.pix.pixelformat = mjpeg_ ? V4L2_PIX_FMT_MJPEG : V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field       = V4L2_FIELD_NONE;
    if (xioctl(fd_, VIDIOC_S_FMT, &fmt) < 0)
        throw std::runtime_error("[V4L2Source] VIDIOC_S_FMT failed: " + std::string(strerror(errno)));

    width_  = static_cast<int>(fmt.fmt.pix.width);
    height_ = static_cast<int>(fmt.fmt.pix.height);
    if (fmt.fmt.pix.pixelformat != (mjpeg_ ? V4L2_PIX_FMT_MJPEG : V4L2_PIX_FMT_YUYV))
        throw std::runtime_error("[V4L2Source] requested pixel format not supported by " + device_);

    v4l2_streamparm parm{};
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator   = 1;
    parm.parm.capture.timeperframe.denominator = static_cast<uint32_t>(fps_);
    xioctl(fd_, VIDIOC_S_PARM, &parm);
    if (xioctl(fd_, VIDIOC_G_PARM, &parm) == 0 && parm.parm.capture.timeperframe.numerator > 0) {
        const double actual = static_cast<double>(parm.parm.capture.timeperframe.denominator) /
                              parm.parm.capture.timeperframe.numerator;
        if (static_cast<int>(actual + 0.5) != fps_)
            std::cout << "[V4L2Source] WARNING: device runs at " << actual << " fps, requested " << fps_ << std::endl;
    }

    v4l2_control prio{};
    prio.id    = V4L2_CID_EXPOSURE_AUTO_PRIORITY;
    prio.value = 0;
    xioctl(fd_, VIDIOC_S_CTRL, &prio);

    // Optional manual exposure. V4L2_CID_EXPOSURE_ABSOLUTE is defined in 100 us units.
    // Long auto-exposure in dim light shifts the image ~exposure/2 into the past.
    if (exposure_100us_ > 0) {
        v4l2_control ae{};
        ae.id    = V4L2_CID_EXPOSURE_AUTO;
        ae.value = V4L2_EXPOSURE_MANUAL;
        if (xioctl(fd_, VIDIOC_S_CTRL, &ae) < 0)
            std::cout << "[V4L2Source] WARNING: cannot switch to manual exposure: " << strerror(errno) << std::endl;
        v4l2_control ex{};
        ex.id    = V4L2_CID_EXPOSURE_ABSOLUTE;
        ex.value = exposure_100us_;
        if (xioctl(fd_, VIDIOC_S_CTRL, &ex) < 0)
            std::cout << "[V4L2Source] WARNING: cannot set exposure " << exposure_100us_ << ": " << strerror(errno) << std::endl;
        else
            std::cout << "[V4L2Source] manual exposure " << exposure_100us_ * 100 << " us" << std::endl;
    }

    v4l2_requestbuffers req{};
    req.count  = 4;
    req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (xioctl(fd_, VIDIOC_REQBUFS, &req) < 0)
        throw std::runtime_error("[V4L2Source] VIDIOC_REQBUFS failed");

    n_buffers_ = static_cast<int>(req.count);
    for (int i = 0; i < n_buffers_; ++i) {
        v4l2_buffer buf{};
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index  = static_cast<uint32_t>(i);
        if (xioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0)
            throw std::runtime_error("[V4L2Source] VIDIOC_QUERYBUF failed");

        buf_lengths_[i] = buf.length;
        buffers_[i] = mmap(nullptr, buf.length,
                           PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buf.m.offset);
        if (buffers_[i] == MAP_FAILED)
            throw std::runtime_error("[V4L2Source] mmap failed");
    }

    for (int i = 0; i < n_buffers_; ++i) {
        v4l2_buffer buf{};
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index  = static_cast<uint32_t>(i);
        if (xioctl(fd_, VIDIOC_QBUF, &buf) < 0)
            throw std::runtime_error("[V4L2Source] VIDIOC_QBUF failed");
    }

    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd_, VIDIOC_STREAMON, &type) < 0)
        throw std::runtime_error("[V4L2Source] VIDIOC_STREAMON failed");

    rgb_buf_.resize(static_cast<size_t>(width_) * height_ * 3);
#ifdef WITH_TURBOJPEG
    if (mjpeg_ && !tj_) tj_ = tjInitDecompress();
#endif

    // Sample the CLOCK_MONOTONIC -> CLOCK_REALTIME offset once. VIDIOC_DQBUF gives us
    // a monotonic-clock timestamp per buffer; this offset lets us convert it into
    // the wall-clock (system_clock) domain the rest of the pipeline expects.
    {
        timespec mono{}, real{};
        clock_gettime(CLOCK_MONOTONIC, &mono);
        clock_gettime(CLOCK_REALTIME, &real);
        int64_t mono_ns = static_cast<int64_t>(mono.tv_sec) * 1000000000LL + mono.tv_nsec;
        int64_t real_ns = static_cast<int64_t>(real.tv_sec) * 1000000000LL + real.tv_nsec;
        clock_offset_ns_ = real_ns - mono_ns;
    }

    std::cout << "[V4L2Source] opened " << device_
              << " negotiated " << width_ << "x" << height_ << " @ " << fps_ << "fps" << std::endl;
}

void V4L2Source::uninitDevice() {
    // Stop streaming, unmap buffers, close fd.
    if (fd_ < 0) return;
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    xioctl(fd_, VIDIOC_STREAMOFF, &type);
    for (int i = 0; i < n_buffers_; ++i)
        if (buffers_[i]) munmap(buffers_[i], buf_lengths_[i]);
    close(fd_);
    fd_        = -1;
    n_buffers_ = 0;
#ifdef WITH_TURBOJPEG
    if (tj_) { tjDestroy(static_cast<tjhandle>(tj_)); tj_ = nullptr; }
#endif
}

void V4L2Source::start(FrameCallback cb) {
    // Initialise device and launch capture thread.
    cb_ = cb;
    initDevice();
    bRunning_ = true;
    thread_   = std::thread(&V4L2Source::run, this);
}

void V4L2Source::stop() {
    // Signal thread to exit, join, then release device.
    bRunning_ = false;
    if (thread_.joinable()) thread_.join();
    uninitDevice();
}

uint32_t V4L2Source::width()  const { return static_cast<uint32_t>(width_);  }
uint32_t V4L2Source::height() const { return static_cast<uint32_t>(height_); }

void V4L2Source::run() {
    // Dequeue frames, convert YUYV→RGB, forward via callback.

    // Periodic report: kernel buffer timestamp -> dequeued here, plus decode/convert
    // time. For uvcvideo the buffer timestamp is the driver's estimate of frame start
    // (or first-packet arrival), so exposure before it is still not included.
    constexpr int kReportEvery = 150;              // ~5 s at 30 fps
    int      n = 0;
    double   sum_dq_ms = 0.0, max_dq_ms = 0.0, sum_cv_ms = 0.0;
    int64_t  first_ts_ns = 0;

    while (bRunning_) {
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(fd_, &fds);
        timeval tv{};
        tv.tv_sec  = 1;
        tv.tv_usec = 0;

        int r = select(fd_ + 1, &fds, nullptr, nullptr, &tv);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) continue;

        v4l2_buffer buf{};
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        if (xioctl(fd_, VIDIOC_DQBUF, &buf) < 0) {
            if (errno == EAGAIN) continue;
            std::cerr << "[V4L2Source] VIDIOC_DQBUF error: " << strerror(errno) << std::endl;
            continue;
        }

        for (;;) {
            fd_set more;
            FD_ZERO(&more);
            FD_SET(fd_, &more);
            timeval zero{};
            if (select(fd_ + 1, &more, nullptr, nullptr, &zero) <= 0) break;
            v4l2_buffer newer{};
            newer.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            newer.memory = V4L2_MEMORY_MMAP;
            if (xioctl(fd_, VIDIOC_DQBUF, &newer) < 0) break;
            xioctl(fd_, VIDIOC_QBUF, &buf);
            buf = newer;
        }

        // buf.timestamp is CLOCK_MONOTONIC-based; translate to system_clock domain.
        int64_t buf_ts_ns = static_cast<int64_t>(buf.timestamp.tv_sec) * 1000000000LL +
                            static_cast<int64_t>(buf.timestamp.tv_usec) * 1000LL;
        uint64_t capture_time_ns = static_cast<uint64_t>(buf_ts_ns + clock_offset_ns_);

        timespec dq_mono{};
        clock_gettime(CLOCK_MONOTONIC, &dq_mono);
        const int64_t dq_ns = static_cast<int64_t>(dq_mono.tv_sec) * 1000000000LL + dq_mono.tv_nsec;

        bool ok = true;
        if (mjpeg_) {
#ifdef WITH_TURBOJPEG
            ok = tjDecompress2(static_cast<tjhandle>(tj_),
                               static_cast<const unsigned char*>(buffers_[buf.index]), buf.bytesused,
                               rgb_buf_.data(), width_, 0, height_, TJPF_RGB, TJFLAG_FASTDCT) == 0;
            if (!ok && !decode_warned_) {
                decode_warned_ = true;
                std::cerr << "[V4L2Source] MJPEG decode failed: " << tjGetErrorStr2(static_cast<tjhandle>(tj_)) << std::endl;
            }
#endif
        } else {
            yuyvToRgb(static_cast<const uint8_t*>(buffers_[buf.index]), rgb_buf_.data(), width_, height_);
        }
        {
            timespec cv_mono{};
            clock_gettime(CLOCK_MONOTONIC, &cv_mono);
            const int64_t cv_ns = static_cast<int64_t>(cv_mono.tv_sec) * 1000000000LL + cv_mono.tv_nsec;
            const double dq_ms = (dq_ns - buf_ts_ns) / 1e6;
            sum_dq_ms += dq_ms;
            if (dq_ms > max_dq_ms) max_dq_ms = dq_ms;
            sum_cv_ms += (cv_ns - dq_ns) / 1e6;
            if (n == 0) first_ts_ns = buf_ts_ns;
            if (++n >= kReportEvery) {
                const double span_s = (buf_ts_ns - first_ts_ns) / 1e9;
                std::cout << "[V4L2Source] buffer ts -> dequeue mean " << sum_dq_ms / n
                          << " ms  max " << max_dq_ms
                          << " | " << (mjpeg_ ? "mjpeg decode" : "yuyv->rgb") << " " << sum_cv_ms / n << " ms"
                          << " | fps " << (span_s > 0 ? (n - 1) / span_s : 0.0) << std::endl;
                n = 0; sum_dq_ms = 0.0; max_dq_ms = 0.0; sum_cv_ms = 0.0;
            }
        }

        if (ok)
            cb_(rgb_buf_.data(), static_cast<uint32_t>(width_), static_cast<uint32_t>(height_), capture_time_ns);
        xioctl(fd_, VIDIOC_QBUF, &buf);
    }
}

#endif // WITH_V4L2