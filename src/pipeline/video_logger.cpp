#include "pipeline/video_logger.hpp"

#include <hdf5.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// PIMPL: HDF5 handles
// ---------------------------------------------------------------------------

struct VideoLogger::Impl {
    hid_t file_id  = H5I_INVALID_HID;
    hid_t dset_img = H5I_INVALID_HID;
    hid_t dset_ts  = H5I_INVALID_HID;
    hid_t dset_fid = H5I_INVALID_HID;

    hsize_t frame_count = 0;

    void close() {
        if (dset_img != H5I_INVALID_HID) { H5Dclose(dset_img); dset_img = H5I_INVALID_HID; }
        if (dset_ts  != H5I_INVALID_HID) { H5Dclose(dset_ts);  dset_ts  = H5I_INVALID_HID; }
        if (dset_fid != H5I_INVALID_HID) { H5Dclose(dset_fid); dset_fid = H5I_INVALID_HID; }
        if (file_id  != H5I_INVALID_HID) { H5Fclose(file_id);  file_id  = H5I_INVALID_HID; }
        frame_count = 0;
    }
};

// ---------------------------------------------------------------------------
// HDF5 attribute helpers
// ---------------------------------------------------------------------------

static void writeStrAttr(hid_t loc, const char* name, const std::string& value) {
    // Use variable-length strings so h5py reads them back without size-zero issues.
    hid_t atype = H5Tcopy(H5T_C_S1);
    H5Tset_size(atype, H5T_VARIABLE);
    H5Tset_strpad(atype, H5T_STR_NULLTERM);
    H5Tset_cset(atype, H5T_CSET_UTF8);
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr  = H5Acreate2(loc, name, atype, space, H5P_DEFAULT, H5P_DEFAULT);
    const char* ptr = value.c_str();
    H5Awrite(attr, atype, &ptr);   // VL strings: write pointer-to-pointer
    H5Aclose(attr); H5Sclose(space); H5Tclose(atype);
}

static void writeI32Attr(hid_t loc, const char* name, int32_t value) {
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr  = H5Acreate2(loc, name, H5T_NATIVE_INT32, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_INT32, &value);
    H5Aclose(attr); H5Sclose(space);
}

// ---------------------------------------------------------------------------
// 1-D extendable uint64 dataset (timestamps / frame IDs)
// ---------------------------------------------------------------------------

static hid_t make1DDataset(hid_t file, const char* path) {
    hsize_t init[1]  = {0};
    hsize_t max[1]   = {H5S_UNLIMITED};
    hsize_t chunk[1] = {256};

    hid_t space = H5Screate_simple(1, init, max);
    hid_t props = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(props, 1, chunk);

    hid_t dset = H5Dcreate2(file, path,
                             H5T_STD_U64LE, space,
                             H5P_DEFAULT, props, H5P_DEFAULT);
    H5Pclose(props);
    H5Sclose(space);
    return dset;
}

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

VideoLogger::VideoLogger(const LoggerConfig& config)
    : config_(config), impl_(std::make_unique<Impl>())
{}

VideoLogger::~VideoLogger() {
    stopEpisode("destructor");   // no-op if already stopped; acquires mutex internally
}

// ---------------------------------------------------------------------------
// Episode lifecycle
// ---------------------------------------------------------------------------

// Internal: must be called with mutex_ already held.
void VideoLogger::stopEpisodeLocked(const std::string& reason) {
    if (!episode_active_) return;

    writeStrAttr(impl_->file_id, "end_reason",   reason);
    writeI32Attr(impl_->file_id, "frame_count",  static_cast<int32_t>(impl_->frame_count));

    std::cout << "[VideoLogger:" << config_.camera_name << "] Episode stopped."
              << " frames=" << impl_->frame_count
              << " reason=" << reason << std::endl;

    impl_->close();
    episode_active_ = false;
}

void VideoLogger::startEpisode(const std::string& session_id, int episode_index) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (episode_active_) stopEpisodeLocked("interrupted");

    char idx_buf[8];
    std::snprintf(idx_buf, sizeof(idx_buf), "%03d", episode_index);

    namespace fs = std::filesystem;
    fs::path dir = fs::path(config_.output_dir) / idx_buf;
    fs::create_directories(dir);
    std::string path = (dir / "images.hdf5").string();

    // Write with the earliest compatible format so any HDF5 1.8+ reader (h5py, etc.) can open it.
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_libver_bounds(fapl, H5F_LIBVER_EARLIEST, H5F_LIBVER_LATEST);
    impl_->file_id = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);
    if (impl_->file_id == H5I_INVALID_HID)
        throw std::runtime_error("[VideoLogger] H5Fcreate failed: " + path);

    // Groups: /observations/images/
    hid_t obs_grp = H5Gcreate2(impl_->file_id, "/observations",
                                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t img_grp = H5Gcreate2(obs_grp, "images",
                                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Gclose(img_grp);
    H5Gclose(obs_grp);

    // Image dataset: (T, H, W, 3) uint8, chunked+compressed
    {
        hsize_t H = static_cast<hsize_t>(config_.height);
        hsize_t W = static_cast<hsize_t>(config_.width);

        hsize_t init[4]  = {0,              H, W, 3};
        hsize_t max[4]   = {H5S_UNLIMITED,  H, W, 3};
        hsize_t chunk[4] = {8,              H, W, 3};  // 8-frame chunks

        hid_t space = H5Screate_simple(4, init, max);
        hid_t props = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(props, 4, chunk);
        H5Pset_deflate(props, 4);  // gzip level 4

        std::string dset_path = "/observations/images/" + config_.camera_name;
        impl_->dset_img = H5Dcreate2(impl_->file_id, dset_path.c_str(),
                                      H5T_STD_U8LE, space,
                                      H5P_DEFAULT, props, H5P_DEFAULT);
        H5Pclose(props);
        H5Sclose(space);

        if (impl_->dset_img == H5I_INVALID_HID) {
            impl_->close();
            throw std::runtime_error("[VideoLogger] failed to create image dataset");
        }
    }

    // 1-D datasets for metadata
    impl_->dset_ts  = make1DDataset(impl_->file_id, "/observations/timestamp_ns");
    impl_->dset_fid = make1DDataset(impl_->file_id, "/observations/frame_id");

    // File-level attributes
    writeStrAttr(impl_->file_id, "session_id",     session_id);
    writeI32Attr(impl_->file_id, "episode_index",  episode_index);

    impl_->frame_count = 0;
    episode_active_    = true;

    std::cout << "[VideoLogger:" << config_.camera_name << "] Episode started → " << path << std::endl;
}

void VideoLogger::stopEpisode(const std::string& reason) {
    std::lock_guard<std::mutex> lk(mutex_);
    stopEpisodeLocked(reason);
}

// ---------------------------------------------------------------------------
// Frame writing
// ---------------------------------------------------------------------------

void VideoLogger::writeFrame(const uint8_t* rgb, uint32_t src_w, uint32_t src_h,
                              uint64_t timestamp_ns, uint64_t frame_id) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (!episode_active_) return;

    const uint8_t* frame = rgb;
    uint32_t effective_src_w = src_w;
    uint32_t effective_src_h = src_h;

    uint32_t dst_w = static_cast<uint32_t>(config_.width);
    uint32_t dst_h = static_cast<uint32_t>(config_.height);

    // ── Determine crop window ──────────────────────────────────────────────
    int cx = config_.crop_x, cy = config_.crop_y;
    int cw = config_.crop_w, ch = config_.crop_h;

    bool do_crop = (cw > 0 && ch > 0);

    if (!do_crop && config_.center_crop) {
        // Auto center-crop: largest rectangle matching dst aspect ratio.
        float dst_aspect = static_cast<float>(dst_w) / static_cast<float>(dst_h);
        float src_aspect = static_cast<float>(src_w) / static_cast<float>(src_h);
        if (src_aspect > dst_aspect) {
            // Source is wider → crop left/right
            ch = static_cast<int>(src_h);
            cw = static_cast<int>(std::round(src_h * dst_aspect));
        } else {
            // Source is taller → crop top/bottom
            cw = static_cast<int>(src_w);
            ch = static_cast<int>(std::round(src_w / dst_aspect));
        }
        cx = (static_cast<int>(src_w) - cw) / 2;
        cy = (static_cast<int>(src_h) - ch) / 2;
        do_crop = true;
    }

    // ── Apply crop (build a row-contiguous crop buffer) ────────────────────
    if (do_crop) {
        // Clamp to source bounds
        cx = std::max(0, std::min(cx, static_cast<int>(src_w) - 1));
        cy = std::max(0, std::min(cy, static_cast<int>(src_h) - 1));
        cw = std::min(cw, static_cast<int>(src_w) - cx);
        ch = std::min(ch, static_cast<int>(src_h) - cy);

        crop_buf_.resize(static_cast<size_t>(cw) * ch * 3);
        for (int row = 0; row < ch; ++row) {
            const uint8_t* src_row = rgb + (static_cast<size_t>(cy + row) * src_w + cx) * 3;
            uint8_t*       dst_row = crop_buf_.data() + static_cast<size_t>(row) * cw * 3;
            std::memcpy(dst_row, src_row, static_cast<size_t>(cw) * 3);
        }
        frame             = crop_buf_.data();
        effective_src_w   = static_cast<uint32_t>(cw);
        effective_src_h   = static_cast<uint32_t>(ch);
    }

    // ── Resize if needed ───────────────────────────────────────────────────
    if (effective_src_w != dst_w || effective_src_h != dst_h) {
        resize_buf_.resize(dst_w * dst_h * 3);
        resizeFrame(frame, effective_src_w, effective_src_h, resize_buf_.data(), dst_w, dst_h);
        frame = resize_buf_.data();
    }

    hsize_t T = impl_->frame_count;
    hsize_t H = static_cast<hsize_t>(dst_h);
    hsize_t W = static_cast<hsize_t>(dst_w);

    // ── Extend and write image ─────────────────────────────────────────────
    {
        hsize_t new_dims[4] = {T + 1, H, W, 3};
        H5Dset_extent(impl_->dset_img, new_dims);

        hid_t fspace = H5Dget_space(impl_->dset_img);
        hsize_t offset[4] = {T, 0, 0, 0};
        hsize_t count[4]  = {1, H, W, 3};
        H5Sselect_hyperslab(fspace, H5S_SELECT_SET, offset, nullptr, count, nullptr);

        hsize_t mdims[4] = {1, H, W, 3};
        hid_t mspace = H5Screate_simple(4, mdims, nullptr);

        H5Dwrite(impl_->dset_img, H5T_STD_U8LE, mspace, fspace, H5P_DEFAULT, frame);
        H5Sclose(mspace);
        H5Sclose(fspace);
    }

    // ── Extend and write timestamp ─────────────────────────────────────────
    {
        hsize_t new_dims[1] = {T + 1};
        H5Dset_extent(impl_->dset_ts, new_dims);

        hid_t fspace = H5Dget_space(impl_->dset_ts);
        hsize_t offset[1] = {T};
        hsize_t count[1]  = {1};
        H5Sselect_hyperslab(fspace, H5S_SELECT_SET, offset, nullptr, count, nullptr);

        hsize_t mdims[1] = {1};
        hid_t mspace = H5Screate_simple(1, mdims, nullptr);

        H5Dwrite(impl_->dset_ts, H5T_STD_U64LE, mspace, fspace, H5P_DEFAULT, &timestamp_ns);
        H5Sclose(mspace);
        H5Sclose(fspace);
    }

    // ── Extend and write frame_id ──────────────────────────────────────────
    {
        hsize_t new_dims[1] = {T + 1};
        H5Dset_extent(impl_->dset_fid, new_dims);

        hid_t fspace = H5Dget_space(impl_->dset_fid);
        hsize_t offset[1] = {T};
        hsize_t count[1]  = {1};
        H5Sselect_hyperslab(fspace, H5S_SELECT_SET, offset, nullptr, count, nullptr);

        hsize_t mdims[1] = {1};
        hid_t mspace = H5Screate_simple(1, mdims, nullptr);

        H5Dwrite(impl_->dset_fid, H5T_STD_U64LE, mspace, fspace, H5P_DEFAULT, &frame_id);
        H5Sclose(mspace);
        H5Sclose(fspace);
    }

    ++impl_->frame_count;
}

// ---------------------------------------------------------------------------
// Resize: area-weighted box filter (correct anti-aliased downscale)
// ---------------------------------------------------------------------------

void VideoLogger::resizeFrame(const uint8_t* src, uint32_t src_w, uint32_t src_h,
                               uint8_t* dst,       uint32_t dst_w, uint32_t dst_h) {
    const float sx = static_cast<float>(src_w) / static_cast<float>(dst_w);
    const float sy = static_cast<float>(src_h) / static_cast<float>(dst_h);

    for (uint32_t dy = 0; dy < dst_h; ++dy) {
        const float y0 = dy * sy;
        const float y1 = y0 + sy;
        const int iy0  = static_cast<int>(y0);
        const int iy1  = std::min(static_cast<int>(std::ceil(y1)), static_cast<int>(src_h));

        for (uint32_t dx = 0; dx < dst_w; ++dx) {
            const float x0 = dx * sx;
            const float x1 = x0 + sx;
            const int ix0  = static_cast<int>(x0);
            const int ix1  = std::min(static_cast<int>(std::ceil(x1)), static_cast<int>(src_w));

            float r = 0.f, g = 0.f, b = 0.f, total_w = 0.f;

            for (int iy = iy0; iy < iy1; ++iy) {
                float wy = 1.f;
                if (iy   == iy0)     wy  = 1.f - (y0 - iy0);
                if (iy   == iy1 - 1) wy *= (y1 - std::floor(y1) > 0.f) ? (y1 - std::floor(y1)) : 1.f;

                for (int ix = ix0; ix < ix1; ++ix) {
                    float wx = 1.f;
                    if (ix == ix0)     wx  = 1.f - (x0 - ix0);
                    if (ix == ix1 - 1) wx *= (x1 - std::floor(x1) > 0.f) ? (x1 - std::floor(x1)) : 1.f;

                    const float w = wx * wy;
                    const uint8_t* p = src + (static_cast<size_t>(iy) * src_w + ix) * 3;
                    r += w * p[0];
                    g += w * p[1];
                    b += w * p[2];
                    total_w += w;
                }
            }

            uint8_t* out = dst + (static_cast<size_t>(dy) * dst_w + dx) * 3;
            if (total_w > 0.f) {
                out[0] = static_cast<uint8_t>(r / total_w + 0.5f);
                out[1] = static_cast<uint8_t>(g / total_w + 0.5f);
                out[2] = static_cast<uint8_t>(b / total_w + 0.5f);
            } else {
                out[0] = out[1] = out[2] = 0;
            }
        }
    }
}
