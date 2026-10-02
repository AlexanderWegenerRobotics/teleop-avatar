#pragma once

#include <string>

#include <Poco/Net/DatagramSocket.h>
#include <Poco/Net/SocketAddress.h>

struct TransportConfig {
    std::string bind_address = "0.0.0.0";
    int         bind_port    = 0;
    std::string remote_ip    = "";
    int         remote_port  = 0;
    int         recv_timeout_us = 100;
};

class UdpTransport {
public:
    explicit UdpTransport(const TransportConfig& config);
    ~UdpTransport();

    UdpTransport(const UdpTransport&) = delete;
    UdpTransport& operator=(const UdpTransport&) = delete;

    int sendTo(const void* data, int size);
    int sendTo(const void* data, int size, const Poco::Net::SocketAddress& dest);
    int receiveFrom(void* buffer, int max_size, Poco::Net::SocketAddress& sender);

    // Wait until a datagram is readable or timeout_us passes. Socket stays non-blocking.
    bool waitReadable(int timeout_us);

    bool hasRemote() const { return has_remote_; }
    const Poco::Net::SocketAddress& remoteAddress() const { return remote_addr_; }

private:
    Poco::Net::DatagramSocket socket_;
    Poco::Net::SocketAddress remote_addr_;
    bool has_remote_ = false;
};