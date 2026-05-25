#include "pipeline/v4l2_source.hpp"

#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

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

V4L2Source::V4L2Source(const std::string& device, int width, int height, int fps)
    : device_(device), width_(width), height_(height), fps_(fps)
{
    std::cout << "[V4L2Source] configured " << device_
              << " " << width_ << "x" << height_ << " @ " << fps_ << "fps" << std::endl;
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
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field       = V4L2_FIELD_NONE;
    if (xioctl(fd_, VIDIOC_S_FMT, &fmt) < 0)
        throw std::runtime_error("[V4L2Source] VIDIOC_S_FMT failed: " + std::string(strerror(errno)));

    width_  = static_cast<int>(fmt.fmt.pix.width);
    height_ = static_cast<int>(fmt.fmt.pix.height);

    v4l2_streamparm parm{};
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator   = 1;
    parm.parm.capture.timeperframe.denominator = static_cast<uint32_t>(fps_);
    xioctl(fd_, VIDIOC_S_PARM, &parm);

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

        yuyvToRgb(static_cast<const uint8_t*>(buffers_[buf.index]), rgb_buf_.data(), width_, height_);
        cb_(rgb_buf_.data(), static_cast<uint32_t>(width_), static_cast<uint32_t>(height_));
        xioctl(fd_, VIDIOC_QBUF, &buf);
    }
}