#include "network/udp_reliable.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>

UdpReliable::UdpReliable(const UdpReliableConfig& config)
    : config_(config)
    , transport_(config.transport)
{
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    last_recv_time_ms_.store(static_cast<uint64_t>(now_ms));
}

UdpReliable::~UdpReliable() {
    stop();
}

void UdpReliable::start() {
    running_ = true;
    thread_ = std::thread(&UdpReliable::run, this);
}

void UdpReliable::stop() {
    running_ = false;
    if (thread_.joinable()) thread_.join();
}

void UdpReliable::registerHandler(const std::string& msg_type, MessageHandler handler) {
    std::lock_guard<std::mutex> lock(handler_mtx_);
    handlers_[msg_type] = std::move(handler);
}

void UdpReliable::setState(SysState state, FaultCode fault) {
    std::lock_guard<std::mutex> lock(send_mtx_);
    current_state_ = state;
    current_fault_ = fault;
}

void UdpReliable::send(const std::string& msg_type, const msgpack::sbuffer& payload_buf, bool ack_requested) {
    sendImmediate(msg_type, payload_buf, ack_requested, config_.default_retry_ms, config_.max_retries);
}

void UdpReliable::sendImmediate(const std::string& msg_type, const msgpack::sbuffer& payload_buf, bool ack_requested, int retry_ms, int max_retries) {
    auto packed = packEnvelope(msg_type, payload_buf, ack_requested);

    transport_.sendTo(packed.data(), static_cast<int>(packed.size()));

    if (ack_requested) {
        int r_ms = (retry_ms > 0) ? retry_ms : config_.default_retry_ms;
        int r_max = (max_retries > 0) ? max_retries : config_.max_retries;

        uint32_t seq;
        {
            std::lock_guard<std::mutex> lock(send_mtx_);
            seq = send_seq_;
        }

        PendingMessage pm;
        pm.packed_data.write(packed.data(), packed.size());
        pm.sequence = seq;
        pm.retry_interval_ms = r_ms;
        pm.max_retries = r_max;
        pm.attempts = 1;
        pm.last_send_time = std::chrono::steady_clock::now();
        pm.first_send_time = pm.last_send_time;

        std::lock_guard<std::mutex> lock(pending_mtx_);
        pending_acks_.push_back(std::move(pm));
    }
}

msgpack::sbuffer UdpReliable::packEnvelope(const std::string& msg_type, const msgpack::sbuffer& payload_buf, bool ack_requested) {
    std::lock_guard<std::mutex> lock(send_mtx_);
    ++send_seq_;

    msgpack::object_handle oh = msgpack::unpack(payload_buf.data(), payload_buf.size());

    ReliableEnvelope env;
    env.sequence      = send_seq_;
    env.timestamp_ns  = timestamp_ns();
    env.state         = static_cast<uint8_t>(current_state_);
    env.fault_code    = static_cast<uint8_t>(current_fault_);
    env.msg_type      = msg_type;
    env.ack_requested = ack_requested;
    env.payload       = oh.get();

    msgpack::sbuffer buf;
    msgpack::pack(buf, env);
    return buf;
}

void UdpReliable::run() {
    auto period = std::chrono::microseconds(1000000 / config_.poll_rate_hz);
    auto next = std::chrono::steady_clock::now();

    while (running_) {
        processIncoming();
        processRetries();
        next += period;
        std::this_thread::sleep_until(next);
    }
}

void UdpReliable::processIncoming() {
    uint8_t buffer[RECV_BUFFER_SIZE];
    Poco::Net::SocketAddress sender;

    while (true) {
        int n = transport_.receiveFrom(buffer, RECV_BUFFER_SIZE, sender);
        if (n <= 0) break;

        try {
            msgpack::object_handle oh = msgpack::unpack(
                reinterpret_cast<const char*>(buffer), static_cast<size_t>(n));
            msgpack::object obj = oh.get();

            ReliableEnvelope env;
            obj.convert(env);

            auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
            last_recv_time_ms_.store(static_cast<uint64_t>(now_ms));
            recv_seq_.store(env.sequence);

            if (env.msg_type == "ack") {
                AckMsg ack;
                env.payload.convert(ack);
                handleAck(ack.ack_sequence);
            } else {
                if (env.ack_requested) {
                    AckMsg ack;
                    ack.ack_sequence = env.sequence;

                    msgpack::sbuffer ack_payload;
                    msgpack::pack(ack_payload, ack);

                    auto packed = packEnvelope("ack", ack_payload, false);
                    transport_.sendTo(packed.data(), static_cast<int>(packed.size()), sender);
                }
                dispatchMessage(env);
            }
        } catch (const msgpack::type_error&) {
        } catch (const msgpack::unpack_error&) {
        }
    }
}

void UdpReliable::dispatchMessage(const ReliableEnvelope& env) {
    std::lock_guard<std::mutex> lock(handler_mtx_);
    auto it = handlers_.find(env.msg_type);
    if (it != handlers_.end()) {
        it->second(env, env.payload);
    }
}

void UdpReliable::processRetries() {
    std::lock_guard<std::mutex> lock(pending_mtx_);
    auto now = std::chrono::steady_clock::now();

    auto it = pending_acks_.begin();
    while (it != pending_acks_.end()) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - it->last_send_time).count();

        if (elapsed >= it->retry_interval_ms) {
            if (it->attempts >= it->max_retries) {
                it = pending_acks_.erase(it);
                continue;
            }

            transport_.sendTo(it->packed_data.data(), static_cast<int>(it->packed_data.size()));
            it->attempts++;
            it->last_send_time = now;
        }
        ++it;
    }
}

void UdpReliable::handleAck(uint32_t ack_seq) {
    std::lock_guard<std::mutex> lock(pending_mtx_);
    auto it = std::find_if(pending_acks_.begin(), pending_acks_.end(), [ack_seq](const PendingMessage& pm) { return pm.sequence == ack_seq; });
    if (it != pending_acks_.end()) {
        pending_acks_.erase(it);
    }
}

bool UdpReliable::isAlive() const {
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::steady_clock::now().time_since_epoch()).count();
    bool alive = (static_cast<uint64_t>(now_ms) - last_recv_time_ms_.load()) < ALIVE_TIMEOUT_MS;
    return alive;
}

uint64_t UdpReliable::lastRecvTimeMs() const {
    return last_recv_time_ms_.load();
}

void UdpReliable::resetAliveTimer() {
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    last_recv_time_ms_.store(static_cast<uint64_t>(now_ms));
}