#include "network/udp_transport.hpp"
#include <Poco/Exception.h>
#include <Poco/Timespan.h>

UdpTransport::UdpTransport(const TransportConfig& config) {
    Poco::Net::SocketAddress bind_addr(config.bind_address, config.bind_port);
    socket_.bind(bind_addr, true);
    socket_.setReceiveTimeout(Poco::Timespan(0, config.recv_timeout_us));
    socket_.setSendTimeout(Poco::Timespan(0, 1000));
    socket_.setBlocking(false);

    if (!config.remote_ip.empty() && config.remote_port > 0) {
        remote_addr_ = Poco::Net::SocketAddress(config.remote_ip, config.remote_port);
        has_remote_ = true;
    }
}

UdpTransport::~UdpTransport() {
    socket_.close();
}

int UdpTransport::sendTo(const void* data, int size) {
    if (!has_remote_) return -1;
    try {
        return socket_.sendTo(data, size, remote_addr_);
    } catch (const Poco::Exception&) {
        return -1;
    }
}

int UdpTransport::sendTo(const void* data, int size, const Poco::Net::SocketAddress& dest) {
    try {
        return socket_.sendTo(data, size, dest);
    } catch (const Poco::Exception&) {
        return -1;
    }
}

bool UdpTransport::waitReadable(int timeout_us) {
    try {
        return socket_.poll(Poco::Timespan(0, timeout_us),
                            Poco::Net::Socket::SELECT_READ);
    } catch (const Poco::Exception&) {
        return false;
    }
}

int UdpTransport::receiveFrom(void* buffer, int max_size, Poco::Net::SocketAddress& sender) {
    try {
        return socket_.receiveFrom(buffer, max_size, sender);
    } catch (const Poco::TimeoutException&) {
        return 0;
    } catch (const Poco::Exception&) {
        return -1;
    }
}