#pragma once

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

// Pinhole camera with OpenCV radial-tangential distortion (k1 k2 p1 p2 k3).
struct LensModel {
    int    width  = 0;
    int    height = 0;
    double fx = 0.0, fy = 0.0, cx = 0.0, cy = 0.0;
    std::array<double, 5> dist{};

    static LensModel pinholeFromFovy(int w, int h, double fovy_deg) {
        LensModel m;
        m.width = w;
        m.height = h;
        m.fy = (h / 2.0) / std::tan(fovy_deg * 3.14159265358979323846 / 360.0);
        m.fx = m.fy;
        m.cx = (w - 1) / 2.0;
        m.cy = (h - 1) / 2.0;
        return m;
    }

    void distortNormalized(double x, double y, double& xd, double& yd) const {
        const double r2 = x * x + y * y;
        const double rad = 1.0 + r2 * (dist[0] + r2 * (dist[1] + r2 * dist[4]));
        xd = x * rad + 2.0 * dist[2] * x * y + dist[3] * (r2 + 2.0 * x * x);
        yd = y * rad + dist[2] * (r2 + 2.0 * y * y) + 2.0 * dist[3] * x * y;
    }

    void undistortNormalized(double xd, double yd, double& x, double& y) const {
        x = xd;
        y = yd;
        for (int i = 0; i < 30; ++i) {
            const double r2 = x * x + y * y;
            const double rad = 1.0 + r2 * (dist[0] + r2 * (dist[1] + r2 * dist[4]));
            x = (xd - (2.0 * dist[2] * x * y + dist[3] * (r2 + 2.0 * x * x))) / rad;
            y = (yd - (dist[2] * (r2 + 2.0 * y * y) + 2.0 * dist[3] * x * y)) / rad;
        }
    }
};

// Per-pixel lookup table: output pixel -> source pixel, applied with fixed-point bilinear sampling on RGB8.
class RemapTable {
public:
    RemapTable() = default;

    // Renders what a lens with distortion `lens` sees, from an ideal pinhole image with the same K.
    static RemapTable distort(const LensModel& lens) {
        RemapTable t(lens.width, lens.height, lens.width, lens.height);
        for (int v = 0; v < lens.height; ++v)
            for (int u = 0; u < lens.width; ++u) {
                double x, y;
                lens.undistortNormalized((u - lens.cx) / lens.fx, (v - lens.cy) / lens.fy, x, y);
                t.set(u, v, x * lens.fx + lens.cx, y * lens.fy + lens.cy);
            }
        return t;
    }

    // Resamples an image of the distorted camera `src` into the ideal pinhole `out`.
    static RemapTable undistort(const LensModel& src, const LensModel& out) {
        RemapTable t(out.width, out.height, src.width, src.height);
        for (int v = 0; v < out.height; ++v)
            for (int u = 0; u < out.width; ++u) {
                double xd, yd;
                src.distortNormalized((u - out.cx) / out.fx, (v - out.cy) / out.fy, xd, yd);
                t.set(u, v, xd * src.fx + src.cx, yd * src.fy + src.cy);
            }
        return t;
    }

    bool empty() const { return offset_.empty(); }
    int width()  const { return w_; }
    int height() const { return h_; }
    int srcWidth()  const { return src_w_; }
    int srcHeight() const { return src_h_; }

    // src is srcWidth() x srcHeight() RGB8, dst holds width() x height() RGB8.
    void apply(const uint8_t* src, uint8_t* dst, int n_threads = 4) const {
        auto rows = [&](int r0, int r1) {
            const size_t stride = static_cast<size_t>(src_w_) * 3;
            for (size_t i = static_cast<size_t>(r0) * w_; i < static_cast<size_t>(r1) * w_; ++i) {
                uint8_t* o = dst + 3 * i;
                const int32_t off = offset_[i];
                if (off < 0) {
                    o[0] = o[1] = o[2] = 0;
                    continue;
                }
                const uint8_t* p = src + off;
                const uint16_t* w = &weight_[4 * i];
                for (int c = 0; c < 3; ++c)
                    o[c] = static_cast<uint8_t>((w[0] * p[c] + w[1] * p[3 + c] + w[2] * p[stride + c]
                                                 + w[3] * p[stride + 3 + c] + 32768) >> 16);
            }
        };
        std::vector<std::thread> pool;
        const int step = (h_ + n_threads - 1) / n_threads;
        for (int r = 0; r < h_; r += step) pool.emplace_back(rows, r, std::min(h_, r + step));
        for (auto& t : pool) t.join();
    }

private:
    RemapTable(int w, int h, int src_w, int src_h)
        : w_(w), h_(h), src_w_(src_w), src_h_(src_h),
          offset_(static_cast<size_t>(w) * h, -1), weight_(static_cast<size_t>(w) * h * 4, 0) {}

    void set(int u, int v, double sx, double sy) {
        const size_t i = static_cast<size_t>(v) * w_ + u;
        const int x0 = static_cast<int>(std::floor(sx)), y0 = static_cast<int>(std::floor(sy));
        if (x0 < 0 || y0 < 0 || x0 + 1 >= src_w_ || y0 + 1 >= src_h_) return;
        const double ax = sx - x0, ay = sy - y0;
        offset_[i] = (y0 * src_w_ + x0) * 3;
        const double wf[4] = {(1 - ax) * (1 - ay), ax * (1 - ay), (1 - ax) * ay, ax * ay};
        int sum = 0;
        for (int k = 0; k < 3; ++k) {
            weight_[4 * i + k] = static_cast<uint16_t>(std::lround(wf[k] * 65535.0));
            sum += weight_[4 * i + k];
        }
        weight_[4 * i + 3] = static_cast<uint16_t>(std::max(0, 65535 - sum));
    }

    int w_ = 0, h_ = 0, src_w_ = 0, src_h_ = 0;
    std::vector<int32_t>  offset_;
    std::vector<uint16_t> weight_;
};

// Undistortion file written by teleop-perception: calibrated camera plus the pinhole to resample into.
inline bool loadUndistortFile(const std::string& path, LensModel& src, LensModel& out) {
    YAML::Node n;
    try {
        n = YAML::LoadFile(path);
    } catch (...) {
        return false;
    }
    auto read = [](const YAML::Node& c, LensModel& m) {
        m.width = c["width"].as<int>();
        m.height = c["height"].as<int>();
        m.fx = c["fx"].as<double>();
        m.fy = c["fy"].as<double>();
        m.cx = c["cx"].as<double>();
        m.cy = c["cy"].as<double>();
        m.dist.fill(0.0);
        if (c["dist"]) {
            auto d = c["dist"].as<std::vector<double>>();
            for (size_t i = 0; i < std::min<size_t>(5, d.size()); ++i) m.dist[i] = d[i];
        }
    };
    read(n["camera"], src);
    read(n["output"], out);
    return true;
}
