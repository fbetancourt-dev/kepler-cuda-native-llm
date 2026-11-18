#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/un.h>
#include "gpu_turbo_governor.h"
#include "../../hardened_governor/include/governor_protocol.h"

static int g_client_fd = -1;
static bool g_has_lease = false;
static uint32_t g_request_counter = 0;
static pthread_mutex_t g_gov_mutex = PTHREAD_MUTEX_INITIALIZER;
static bool g_fork_handler_registered = false;
static bool g_daemon_unavailable_warned = false;

static int client_read_exact(int fd, void *buf, size_t count) {
    size_t total = 0;
    char *p = (char *)buf;
    while (total < count) {
        ssize_t n = recv(fd, p + total, count - total, 0);
        if (n <= 0) return (n == 0) ? 0 : -1;
        total += n;
    }
    return (int)total;
}

static int client_send_exact(int fd, const void *buf, size_t count) {
    size_t total = 0;
    const char *p = (const char *)buf;
    while (total < count) {
        ssize_t n = send(fd, p + total, count - total, MSG_NOSIGNAL);
        if (n <= 0) return -1;
        total += n;
    }
    return (int)total;
}

static void atfork_child_handler(void) {
    // Si el proceso hace fork, el hijo invalida el socket heredado
    if (g_client_fd >= 0) {
        close(g_client_fd);
        g_client_fd = -1;
    }
    g_has_lease = false;
}

int gpu_governor_client_init(void) {
    pthread_mutex_lock(&g_gov_mutex);

    if (!g_fork_handler_registered) {
        pthread_atfork(NULL, NULL, atfork_child_handler);
        g_fork_handler_registered = true;
    }

    if (g_client_fd >= 0) {
        pthread_mutex_unlock(&g_gov_mutex);
        return 0; // Ya conectado
    }

    int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) {
        pthread_mutex_unlock(&g_gov_mutex);
        return -1;
    }

    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 50000; // 50 ms timeout
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, GOVERNOR_SOCKET_PATH, sizeof(addr.sun_path) - 1);

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(fd);
        if (!g_daemon_unavailable_warned) {
            fprintf(stderr, "[libcuda_kepler] Notice: GPU Turbo Governor daemon not running at %s. Continuing in safe driver default clock.\n",
                    GOVERNOR_SOCKET_PATH);
            g_daemon_unavailable_warned = true;
        }
        pthread_mutex_unlock(&g_gov_mutex);
        return -1; // Fallback graceful
    }

    g_client_fd = fd;
    pthread_mutex_unlock(&g_gov_mutex);
    return 0;
}

int gpu_governor_request_turbo(uint32_t ttl_ms) {
    if (gpu_governor_client_init() != 0) {
        return 0; // Graceful degradation
    }

    pthread_mutex_lock(&g_gov_mutex);
    gov_request_t req;
    req.proto_version = GOV_PROTOCOL_VERSION;
    req.cmd = GOV_CMD_REQUEST_LEASE;
    req.request_id = ++g_request_counter;
    req.lease_ms = (ttl_ms > 0) ? ttl_ms : 3000;

    if (client_send_exact(g_client_fd, &req, sizeof(req)) != sizeof(req)) {
        close(g_client_fd);
        g_client_fd = -1;
        pthread_mutex_unlock(&g_gov_mutex);
        return 0;
    }

    gov_response_t resp;
    if (client_read_exact(g_client_fd, &resp, sizeof(resp)) == sizeof(resp)) {
        if (resp.status == GOV_STATUS_SUCCESS || resp.status == GOV_STATUS_CLAMPED) {
            g_has_lease = true;
        }
    }
    pthread_mutex_unlock(&g_gov_mutex);
    return 0;
}

int gpu_governor_renew_turbo(uint32_t ttl_ms) {
    pthread_mutex_lock(&g_gov_mutex);
    if (g_client_fd < 0 || !g_has_lease) {
        pthread_mutex_unlock(&g_gov_mutex);
        return 0;
    }

    gov_request_t req;
    req.proto_version = GOV_PROTOCOL_VERSION;
    req.cmd = GOV_CMD_RENEW_LEASE;
    req.request_id = ++g_request_counter;
    req.lease_ms = (ttl_ms > 0) ? ttl_ms : 3000;

    if (client_send_exact(g_client_fd, &req, sizeof(req)) == sizeof(req)) {
        gov_response_t resp;
        client_read_exact(g_client_fd, &resp, sizeof(resp));
    }
    pthread_mutex_unlock(&g_gov_mutex);
    return 0;
}

int gpu_governor_release_turbo(void) {
    pthread_mutex_lock(&g_gov_mutex);
    if (g_client_fd < 0 || !g_has_lease) {
        pthread_mutex_unlock(&g_gov_mutex);
        return 0;
    }

    gov_request_t req;
    req.proto_version = GOV_PROTOCOL_VERSION;
    req.cmd = GOV_CMD_RELEASE_LEASE;
    req.request_id = ++g_request_counter;
    req.lease_ms = 0;

    if (client_send_exact(g_client_fd, &req, sizeof(req)) == sizeof(req)) {
        gov_response_t resp;
        client_read_exact(g_client_fd, &resp, sizeof(resp));
    }

    g_has_lease = false;
    pthread_mutex_unlock(&g_gov_mutex);
    return 0;
}

void gpu_governor_client_close(void) {
    pthread_mutex_lock(&g_gov_mutex);
    if (g_client_fd >= 0) {
        if (g_has_lease) {
            gov_request_t req;
            req.proto_version = GOV_PROTOCOL_VERSION;
            req.cmd = GOV_CMD_RELEASE_LEASE;
            req.request_id = ++g_request_counter;
            req.lease_ms = 0;
            client_send_exact(g_client_fd, &req, sizeof(req));
            g_has_lease = false;
        }
        close(g_client_fd);
        g_client_fd = -1;
    }
    pthread_mutex_unlock(&g_gov_mutex);
}
