#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#ifdef _WIN32
// ── Windows implementation using named file mapping ───────────────────────────
// Equivalent to POSIX shm_open + mmap(MAP_SHARED).
// POSIX names like "/avatar_cam" are mapped to "Local\avatar_cam".

constexpr int    SHM_N_SLOTS  = 3;
constexpr size_t SHM_MAX_W    = 1280;
constexpr size_t SHM_MAX_H    = 960;
constexpr size_t SHM_CHANNELS = 3;

struct SharedFrameBuffer {
    std::atomic<uint32_t> write_idx{0};
    std::atomic<uint32_t> frame_count{0};
    uint32_t width  = 0;
    uint32_t height = 0;
    uint8_t  slots[SHM_N_SLOTS][SHM_MAX_W * SHM_MAX_H * SHM_CHANNELS]{};
};

// Convert a POSIX shm name ("/avatar_cam") to a Win32 named-object name.
inline std::string toWin32Name(const std::string& posix_name) {
    std::string n = posix_name;
    if (!n.empty() && n[0] == '/') n = n.substr(1);
    return "Local\\" + n;
}

class SharedMemoryWriter {
public:
    SharedMemoryWriter(const std::string& name, uint32_t width, uint32_t height) {
        std::string wname = toWin32Name(name);
        DWORD sz_hi = static_cast<DWORD>(sizeof(SharedFrameBuffer) >> 32);
        DWORD sz_lo = static_cast<DWORD>(sizeof(SharedFrameBuffer) & 0xFFFFFFFF);
        handle_ = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr,
                                     PAGE_READWRITE, sz_hi, sz_lo,
                                     wname.c_str());
        if (!handle_)
            throw std::runtime_error("CreateFileMapping failed for: " + wname);

        buf_ = static_cast<SharedFrameBuffer*>(
            MapViewOfFile(handle_, FILE_MAP_ALL_ACCESS, 0, 0,
                          sizeof(SharedFrameBuffer)));
        if (!buf_) {
            CloseHandle(handle_);
            throw std::runtime_error("MapViewOfFile (write) failed for: " + wname);
        }

        buf_->width       = width;
        buf_->height      = height;
        buf_->write_idx   = 0;
        buf_->frame_count = 0;
    }

    ~SharedMemoryWriter() {
        if (buf_)    UnmapViewOfFile(buf_);
        if (handle_) CloseHandle(handle_);
    }

    void write(const uint8_t* rgb, size_t size) {
        constexpr size_t max_size = SHM_MAX_W * SHM_MAX_H * SHM_CHANNELS;
        if (size > max_size)
            throw std::runtime_error("SharedMemoryWriter::write: frame exceeds slot capacity");
        uint32_t slot = buf_->write_idx.load() % SHM_N_SLOTS;
        std::memcpy(buf_->slots[slot], rgb, size);
        buf_->write_idx.fetch_add(1);
        buf_->frame_count.fetch_add(1);
    }

    uint32_t width()  const { return buf_->width; }
    uint32_t height() const { return buf_->height; }

private:
    HANDLE             handle_ = nullptr;
    SharedFrameBuffer* buf_    = nullptr;
};

class SharedMemoryReader {
public:
    SharedMemoryReader(const std::string& name) {
        std::string wname = toWin32Name(name);
        handle_ = OpenFileMappingA(FILE_MAP_READ, FALSE, wname.c_str());
        if (!handle_)
            throw std::runtime_error("OpenFileMapping failed for: " + wname);

        buf_ = static_cast<const SharedFrameBuffer*>(
            MapViewOfFile(handle_, FILE_MAP_READ, 0, 0,
                          sizeof(SharedFrameBuffer)));
        if (!buf_) {
            CloseHandle(handle_);
            throw std::runtime_error("MapViewOfFile (read) failed for: " + wname);
        }

        last_frame_ = buf_->frame_count.load();
    }

    ~SharedMemoryReader() {
        if (buf_)    UnmapViewOfFile(const_cast<SharedFrameBuffer*>(buf_));
        if (handle_) CloseHandle(handle_);
    }

    bool hasNewFrame() const {
        return buf_->frame_count.load() != last_frame_;
    }

    const uint8_t* read() {
        uint32_t slot = (buf_->write_idx.load(std::memory_order_acquire)
                         + SHM_N_SLOTS - 1) % SHM_N_SLOTS;
        last_frame_   = buf_->frame_count.load();
        return buf_->slots[slot];
    }

    uint32_t width()  const { return buf_->width; }
    uint32_t height() const { return buf_->height; }

private:
    HANDLE                   handle_     = nullptr;
    const SharedFrameBuffer* buf_        = nullptr;
    uint32_t                 last_frame_ = 0;
};

#else
// ── POSIX implementation ──────────────────────────────────────────────────────
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

constexpr int    SHM_N_SLOTS  = 3;
constexpr size_t SHM_MAX_W = 1280;
constexpr size_t SHM_MAX_H = 960;
constexpr size_t SHM_CHANNELS = 3;

struct SharedFrameBuffer {
    std::atomic<uint32_t> write_idx;
    std::atomic<uint32_t> frame_count;
    uint32_t width;
    uint32_t height;
    uint8_t  slots[SHM_N_SLOTS][SHM_MAX_W * SHM_MAX_H * SHM_CHANNELS];
};

class SharedMemoryWriter {
public:
    SharedMemoryWriter(const std::string& name, uint32_t width, uint32_t height)
        : name_(name)
    {
        fd_ = shm_open(name.c_str(), O_CREAT | O_RDWR, 0666);
        if (fd_ < 0) throw std::runtime_error("shm_open failed: " + name);

        ftruncate(fd_, sizeof(SharedFrameBuffer));
        buf_ = static_cast<SharedFrameBuffer*>(
            mmap(nullptr, sizeof(SharedFrameBuffer),
                 PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));

        if (buf_ == MAP_FAILED) throw std::runtime_error("mmap failed");

        buf_->width       = width;
        buf_->height      = height;
        buf_->write_idx   = 0;
        buf_->frame_count = 0;
    }

    ~SharedMemoryWriter() {
        if (buf_) munmap(buf_, sizeof(SharedFrameBuffer));
        if (fd_ >= 0) close(fd_);
        shm_unlink(name_.c_str());
    }

    void write(const uint8_t* rgb, size_t size) {
        size_t max_size = SHM_MAX_W * SHM_MAX_H * SHM_CHANNELS;
        if (size > max_size)
            throw std::runtime_error("SharedMemoryWriter::write: frame size " +
                std::to_string(size) + " exceeds slot capacity " +
                std::to_string(max_size));
        uint32_t slot = buf_->write_idx.load() % SHM_N_SLOTS;
        std::memcpy(buf_->slots[slot], rgb, size);
        buf_->write_idx.fetch_add(1);
        buf_->frame_count.fetch_add(1);
    }

    uint32_t width()  const { return buf_->width; }
    uint32_t height() const { return buf_->height; }

private:
    std::string          name_;
    int                  fd_  = -1;
    SharedFrameBuffer*   buf_ = nullptr;
};

class SharedMemoryReader {
public:
    SharedMemoryReader(const std::string& name)
        : name_(name)
    {
        fd_ = shm_open(name.c_str(), O_RDONLY, 0666);
        if (fd_ < 0) throw std::runtime_error("shm_open failed: " + name);

        buf_ = static_cast<SharedFrameBuffer*>(mmap(nullptr, sizeof(SharedFrameBuffer),PROT_READ, MAP_SHARED, fd_, 0));

        if (buf_ == MAP_FAILED) throw std::runtime_error("mmap failed");
        last_frame_ = buf_->frame_count.load();
    }

    ~SharedMemoryReader() {
        if (buf_) munmap(buf_, sizeof(SharedFrameBuffer));
        if (fd_ >= 0) close(fd_);
    }

    bool hasNewFrame() const {
        return buf_->frame_count.load() != last_frame_;
    }

    const uint8_t* read() {
        // take a snapshot of write_idx before reading
        uint32_t slot = (buf_->write_idx.load(std::memory_order_acquire) + SHM_N_SLOTS - 1) % SHM_N_SLOTS;
        last_frame_   = buf_->frame_count.load();
        return buf_->slots[slot];
    }

    uint32_t width()  const { return buf_->width; }
    uint32_t height() const { return buf_->height; }

private:
    std::string         name_;
    int                 fd_       = -1;
    SharedFrameBuffer*  buf_      = nullptr;
    uint32_t            last_frame_ = 0;
};

#endif // _WIN32