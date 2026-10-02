#pragma once

// Cross-platform socket header. On Windows include before <windows.h> (winsock2 conflict).

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <winsock2.h>
    #include <ws2tcpip.h>  // inet_pton, sockaddr_in, etc.
    using socket_t  = SOCKET;
    using ssize_t   = SSIZE_T;
    using socklen_t = int;
    inline void close_socket(socket_t s) { closesocket(s); }
    static constexpr socket_t kInvalidSocket = INVALID_SOCKET;
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    using socket_t = int;
    inline void close_socket(socket_t s) { ::close(s); }
    static constexpr socket_t kInvalidSocket = -1;
#endif
