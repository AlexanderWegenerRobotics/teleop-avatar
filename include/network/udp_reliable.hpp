#pragma once

#include <string>
#include <functional>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <cstdint>

#include <msgpack.hpp>

#include "udp_transport.hpp"
#include "common.hpp"

struct ReliableEnvelope {
    uint32_t    sequence      = 0;
    uint64_t    timestamp_ns  = 0;
    uint8_t     state         = static_cast<uint8_t>(SysState::OFFLINE);
    uint8_t     fault_code    = static_cast<uint8_t>(FaultCode::NONE);
    std::string msg_type;
    bool        ack_requested = false;
    msgpack::object payload;

    MSGPACK_DEFINE_MAP(sequence, timestamp_ns, state, fault_code, msg_type, ack_requested, payload)
};

struct AckMsg {
    std::string msg_type = "ack";
    uint32_t    ack_sequence = 0;

    MSGPACK_DEFINE_MAP(msg_type, ack_sequence)
};

struct UdpReliableConfig {
    TransportConfig transport;
    int             poll_rate_hz       = 100;
    int             default_retry_ms   = 200;
    int             max_retries        = 10;
    int             aggressive_retry_ms = 10;
    int             aggressive_max     = 50;
};

using MessageHandler = std::function<void(const ReliableEnvelope&, const msgpack::object&)>;

class UdpReliable {
public:
    explicit UdpReliable(const UdpReliableConfig& config);
    ~UdpReliable();

    UdpReliable(const UdpReliable&) = delete;
    UdpReliable& operator=(const UdpReliable&) = delete;

    void start();
    void stop();

    void registerHandler(const std::string& msg_type, MessageHandler handler);
    void send(const std::string& msg_type, const msgpack::sbuffer& payload_buf, bool ack_requested = false);
    void sendImmediate(const std::string& msg_type, const msgpack::sbuffer& payload_buf, bool ack_requested = false, int retry_ms = -1, int max_retries = -1);
    void setState(SysState state, FaultCode fault = FaultCode::NONE);

    bool isAlive() const;
    uint64_t lastRecvTimeMs() const;
    void resetAliveTimer();

private:
    struct PendingMessage {
        msgpack::sbuffer packed_data;
        uint32_t         sequence;
        int              retry_interval_ms;
        int              max_retries;
        int              attempts;
        std::chrono::steady_clock::time_point last_send_time;
        std::chrono::steady_clock::time_point first_send_time;
    };

    void run();
    void processIncoming();
    void processRetries();
    void handleAck(uint32_t ack_seq);
    void dispatchMessage(const ReliableEnvelope& env);
    msgpack::sbuffer packEnvelope(const std::string& msg_type, const msgpack::sbuffer& payload_buf, bool ack_requested);

    UdpReliableConfig config_;
    UdpTransport          transport_;

    std::thread           thread_;
    std::atomic<bool>     running_{false};

    std::mutex            send_mtx_;
    uint32_t              send_seq_ = 0;
    SysState              current_state_ = SysState::OFFLINE;
    FaultCode             current_fault_ = FaultCode::NONE;

    std::mutex            pending_mtx_;
    std::vector<PendingMessage> pending_acks_;

    std::mutex            handler_mtx_;
    std::unordered_map<std::string, MessageHandler> handlers_;

    std::atomic<uint64_t> last_recv_time_ms_{0};
    std::atomic<uint32_t> recv_seq_{0};

    static constexpr int  RECV_BUFFER_SIZE = 65536;
    static constexpr int  ALIVE_TIMEOUT_MS = 10000;
};