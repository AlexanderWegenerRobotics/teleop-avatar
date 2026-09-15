#pragma once

#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <cstring>
#include <functional>
#include <type_traits>
#include <utility>

#include "udp_transport.hpp"
#include "common.hpp"

struct UdpStreamConfig {
    TransportConfig transport;
    int             send_rate_hz  = 100;
    int             recv_enabled  = true;
};

template<typename TRecv, typename TSend>
class UdpStream {
    static_assert(std::is_trivially_copyable_v<TRecv>, "TRecv must be trivially copyable");
    static_assert(std::is_trivially_copyable_v<TSend>, "TSend must be trivially copyable");
    static_assert(offsetof(TRecv, header) == 0, "TRecv must start with MsgHeader");
    static_assert(offsetof(TSend, header) == 0, "TSend must start with MsgHeader");

public:
    explicit UdpStream(const UdpStreamConfig& config)
        : config_(config)
        , transport_(config.transport)
    {
        std::memset(&recv_msg_, 0, sizeof(TRecv));
        std::memset(&send_msg_, 0, sizeof(TSend));
        last_recv_time_ = std::chrono::steady_clock::now();
    }

    ~UdpStream() { stop(); }

    UdpStream(const UdpStream&) = delete;
    UdpStream& operator=(const UdpStream&) = delete;

    // Invoked on the receive thread the instant a packet is accepted, before
    // any consumer polls hasNew(). Set it before start(); it is read without a
    // lock and is not meant to change while running. Keep the callback short --
    // it runs on the receive thread and delays the next drain.
    void setOnReceive(std::function<void()> cb) { on_receive_ = std::move(cb); }

    void start() {
        running_ = true;
        send_thread_ = std::thread(&UdpStream::runSend, this);
        if (config_.recv_enabled) recv_thread_ = std::thread(&UdpStream::runRecv, this);
    }

    void stop() {
        running_ = false;
        if (send_thread_.joinable()) send_thread_.join();
        if (recv_thread_.joinable()) recv_thread_.join();
    }

    void setSendData(const TSend& msg) {
        std::lock_guard<std::mutex> lock(send_mtx_);
        send_msg_ = msg;
    }

    TRecv getRecvData() {
        std::lock_guard<std::mutex> lock(recv_mtx_);
        has_new_ = false;
        return recv_msg_;
    }

    void setState(SysState state, FaultCode fault = FaultCode::NONE) {
        std::lock_guard<std::mutex> lock(send_mtx_);
        sticky_state_ = state;
        sticky_fault_ = fault;
    }

    bool hasNew() const { return has_new_; }
    bool isAlive() const {
        auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - last_recv_time_).count();
        return delta < ALIVE_TIMEOUT_MS;
    }

    uint32_t lastRecvSequence() const { return last_recv_seq_; }
    uint32_t droppedPackets() const { return dropped_count_; }

private:
    // Send and receive used to share one thread paced at send_rate_hz, which put
    // a 0-5 ms polling delay (at 200 Hz) in front of every inbound command on a
    // socket that was already non-blocking. They are now independent: send stays
    // periodic, receive blocks on the socket and fires on_receive_ as soon as a
    // packet is accepted. The poll timeout below only bounds how quickly the
    // thread notices a stop() request -- a packet wakes it immediately.
    void runSend() {
        auto period = std::chrono::microseconds(1000000 / config_.send_rate_hz);
        auto next = std::chrono::steady_clock::now();

        while (running_) {
            doSend();
            next += period;
            std::this_thread::sleep_until(next);
        }
    }

    void runRecv() {
        while (running_) {
            if (!transport_.waitReadable(kRecvPollTimeoutUs)) continue;
            if (receive() && on_receive_) on_receive_();
        }
    }

    void doSend() {
        std::lock_guard<std::mutex> lock(send_mtx_);
        send_msg_.header.sequence = ++send_seq_;
        send_msg_.header.timestamp_ns = timestamp_ns();
        send_msg_.header.state = sticky_state_;
        send_msg_.header.fault_code = sticky_fault_;
        transport_.sendTo(&send_msg_, sizeof(TSend));
    }

    // Returns true if at least one packet was accepted into recv_msg_.
    bool receive() {
        Poco::Net::SocketAddress sender;
        uint8_t buffer[sizeof(TRecv) + 64];
        bool accepted = false;

        while (true) {
            int n = transport_.receiveFrom(buffer, sizeof(buffer), sender);
            if (n <= 0) break;

            if (n == sizeof(TRecv)) {
                TRecv msg;
                std::memcpy(&msg, buffer, sizeof(TRecv));

                uint32_t seq = msg.header.sequence;
                if (seq > last_recv_seq_ + 1 && last_recv_seq_ > 0) {
                    dropped_count_ += (seq - last_recv_seq_ - 1);
                }

                if (seq > last_recv_seq_ || last_recv_seq_ == 0) {
                    std::lock_guard<std::mutex> lock(recv_mtx_);
                    recv_msg_ = msg;
                    has_new_ = true;
                    last_recv_seq_ = seq;
                    last_recv_time_ = std::chrono::steady_clock::now();
                    accepted = true;
                }
            }
        }
        return accepted;
    }

    UdpStreamConfig config_;
    UdpTransport      transport_;

    std::thread       send_thread_;
    std::thread       recv_thread_;
    std::atomic<bool> running_{false};

    std::function<void()> on_receive_;

    // Shutdown responsiveness only; packet arrival wakes the poll immediately.
    static constexpr int kRecvPollTimeoutUs = 2000;

    std::mutex        send_mtx_;
    TSend             send_msg_;
    uint32_t          send_seq_ = 0;
    SysState          sticky_state_ = SysState::OFFLINE;
    FaultCode         sticky_fault_ = FaultCode::NONE;

    std::mutex        recv_mtx_;
    TRecv             recv_msg_;
    std::atomic<bool> has_new_{false};

    std::chrono::steady_clock::time_point last_recv_time_;
    std::atomic<uint32_t> last_recv_seq_{0};
    std::atomic<uint32_t> dropped_count_{0};

    static constexpr int ALIVE_TIMEOUT_MS = 500;
};