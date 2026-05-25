/*
 * seraph-gpu-viz — Minimal HTTP/SSE server for real-time GPU visualization
 * Serves static files, spawns training, streams output via Server-Sent Events
 * Zero dependencies — pure POSIX C
 *
 * Build: gcc -O3 -o seraph-gpu-viz tools/gpu-viz/server.c
 * Usage: ./seraph-gpu-viz [--port 8420]
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <dirent.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <signal.h>
#include <fcntl.h>
#include <errno.h>

#define DEFAULT_PORT 8420
#define MAX_SSE      16
#define REQ_BUF      16384
#define LINE_BUF     8192
#define PID_FILE     "/tmp/seraph-gpu-viz.pid"

/* ═══════════════════════════════════════════════════════════════════════════ */
/* STATE                                                                      */
/* ═══════════════════════════════════════════════════════════════════════════ */

static struct {
    int      listen_fd;
    int      port;
    char     base_dir[512];
    char     static_dir[512];

    pid_t    train_pid;
    int      train_pipe;

    int      sse_fds[MAX_SSE];
    int      n_sse;

    char     line_buf[LINE_BUF];
    int      line_len;

    volatile sig_atomic_t running;
} S = { .listen_fd = -1, .train_pid = -1, .train_pipe = -1, .running = 1 };

/* ═══════════════════════════════════════════════════════════════════════════ */
/* CLEANUP + SIGNALS                                                          */
/* ═══════════════════════════════════════════════════════════════════════════ */

static void cleanup(void) {
    if (S.train_pid > 0) { kill(S.train_pid, SIGTERM); waitpid(S.train_pid, NULL, 0); }
    for (int i = 0; i < S.n_sse; i++) close(S.sse_fds[i]);
    if (S.listen_fd >= 0) close(S.listen_fd);
    S.listen_fd = -1;
    unlink(PID_FILE);
}

static void on_signal(int s) {
    (void)s;
    S.running = 0;
}

static void write_pid_file(void) {
    FILE* f = fopen(PID_FILE, "w");
    if (f) { fprintf(f, "%d\n", getpid()); fclose(f); }
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* UTILITIES                                                                  */
/* ═══════════════════════════════════════════════════════════════════════════ */

static const char* mime_type(const char* path) {
    const char* e = strrchr(path, '.');
    if (!e) return "text/plain";
    if (!strcmp(e, ".html")) return "text/html; charset=utf-8";
    if (!strcmp(e, ".js"))   return "application/javascript";
    if (!strcmp(e, ".css"))  return "text/css";
    if (!strcmp(e, ".json")) return "application/json";
    if (!strcmp(e, ".svg"))  return "image/svg+xml";
    if (!strcmp(e, ".png"))  return "image/png";
    return "text/plain";
}

static int json_str(const char* json, const char* key, char* out, int n) {
    if (!json) { out[0] = 0; return -1; }
    char pat[80];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char* p = strstr(json, pat);
    if (!p) { out[0] = 0; return -1; }
    p += strlen(pat);
    while (*p && (*p == ' ' || *p == ':' || *p == '\t')) p++;
    if (*p != '"') { out[0] = 0; return -1; }
    p++;
    int i = 0;
    while (*p && *p != '"' && i < n - 1) {
        if (*p == '\\' && *(p + 1)) p++;
        out[i++] = *p++;
    }
    out[i] = 0;
    return 0;
}

static int query_param(const char* q, const char* key, char* out, int n) {
    if (!q) { out[0] = 0; return -1; }
    char pat[64];
    snprintf(pat, sizeof(pat), "%s=", key);
    const char* p = strstr(q, pat);
    if (!p) { out[0] = 0; return -1; }
    p += strlen(pat);
    int i = 0;
    while (*p && *p != '&' && i < n - 1) {
        if (*p == '%' && p[1] && p[2]) {
            char h[3] = { p[1], p[2], 0 };
            out[i++] = (char)strtol(h, NULL, 16);
            p += 3;
        } else {
            out[i++] = (*p == '+') ? ' ' : *p;
            p++;
        }
    }
    out[i] = 0;
    return 0;
}

/* case-insensitive substring search (no glibc dependency) */
static char* find_header(const char* haystack, const char* needle) {
    size_t nlen = strlen(needle);
    for (const char* p = haystack; *p; p++) {
        if (strncasecmp(p, needle, nlen) == 0) return (char*)p;
    }
    return NULL;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* HTTP RESPONSES                                                             */
/* ═══════════════════════════════════════════════════════════════════════════ */

static void http_send(int fd, int code, const char* type, const char* body, int len) {
    const char* msg = code == 200 ? "OK" : code == 404 ? "Not Found" : "Error";
    char hdr[512];
    int hl = snprintf(hdr, sizeof(hdr),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %d\r\n"
        "Connection: close\r\n\r\n",
        code, msg, type, len);
    send(fd, hdr, hl, MSG_NOSIGNAL);
    if (len > 0) send(fd, body, len, MSG_NOSIGNAL);
}

static void http_json(int fd, int code, const char* json) {
    http_send(fd, code, "application/json", json, (int)strlen(json));
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* SSE                                                                        */
/* ═══════════════════════════════════════════════════════════════════════════ */

static void sse_broadcast(const char* text) {
    for (int i = S.n_sse - 1; i >= 0; i--) {
        char msg[LINE_BUF + 32];
        int len = snprintf(msg, sizeof(msg), "data: %s\n\n", text);
        int r = send(S.sse_fds[i], msg, len, MSG_NOSIGNAL);
        if (r <= 0) {
            close(S.sse_fds[i]);
            S.sse_fds[i] = S.sse_fds[--S.n_sse];
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* STATIC FILES                                                               */
/* ═══════════════════════════════════════════════════════════════════════════ */

static void serve_file(int fd, const char* url) {
    const char* p = (!strcmp(url, "/")) ? "/index.html"
                   : (!strcmp(url, "/tensor")) ? "/tensor_viewer.html"
                   : (!strcmp(url, "/band")) ? "/band_dashboard.html"
                   : url;
    char path[1024];
    snprintf(path, sizeof(path), "%s%s", S.static_dir, p);
    if (strstr(path, "..")) { http_send(fd, 404, "text/plain", "no", 2); return; }

    struct stat st;
    if (stat(path, &st) || S_ISDIR(st.st_mode)) {
        http_send(fd, 404, "text/plain", "not found", 9);
        return;
    }

    FILE* f = fopen(path, "rb");
    if (!f) { http_send(fd, 404, "text/plain", "not found", 9); return; }

    char hdr[512];
    int hl = snprintf(hdr, sizeof(hdr),
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %ld\r\n"
        "Connection: close\r\n\r\n",
        mime_type(p), (long)st.st_size);
    send(fd, hdr, hl, MSG_NOSIGNAL);

    char buf[8192];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0)
        send(fd, buf, n, MSG_NOSIGNAL);
    fclose(f);
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* FILE BROWSER                                                               */
/* ═══════════════════════════════════════════════════════════════════════════ */

static void serve_browse(int fd, const char* dir_path) {
    char resolved[1024];
    const char* target = (dir_path && dir_path[0]) ? dir_path : "/";

    if (!realpath(target, resolved)) {
        http_json(fd, 400, "{\"error\":\"invalid path\"}");
        return;
    }

    DIR* d = opendir(resolved);
    if (!d) { http_json(fd, 400, "{\"error\":\"cannot open directory\"}"); return; }

    /* build JSON: {"path":"/abs/path","entries":[{"name":"foo","dir":true}, ...]} */
    char out[65536];
    int pos = snprintf(out, sizeof(out), "{\"path\":\"%s\",\"entries\":[", resolved);

    struct dirent* ent;
    int first = 1;
    while ((ent = readdir(d)) && pos < (int)sizeof(out) - 256) {
        if (!strcmp(ent->d_name, ".")) continue;

        /* stat to check if dir */
        char full[2048];
        snprintf(full, sizeof(full), "%s/%s", resolved, ent->d_name);
        struct stat st;
        int is_dir = (stat(full, &st) == 0 && S_ISDIR(st.st_mode));

        if (!first) out[pos++] = ',';
        first = 0;

        /* escape name (skip entries with quotes) */
        if (strchr(ent->d_name, '"')) continue;

        pos += snprintf(out + pos, sizeof(out) - pos,
            "{\"name\":\"%s\",\"dir\":%s}",
            ent->d_name, is_dir ? "true" : "false");
    }
    closedir(d);

    pos += snprintf(out + pos, sizeof(out) - pos, "]}");
    http_send(fd, 200, "application/json", out, pos);
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* GPU INFO                                                                   */
/* ═══════════════════════════════════════════════════════════════════════════ */

static void serve_gpu_info(int fd, const char* model_dir) {
    char cmd[1024];
    if (model_dir[0])
        snprintf(cmd, sizeof(cmd), "%s/tests/gpu-profile-query --defines %s 2>&1",
                 S.base_dir, model_dir);
    else
        snprintf(cmd, sizeof(cmd), "%s/tests/gpu-profile-query 2>&1", S.base_dir);

    FILE* fp = popen(cmd, "r");
    if (!fp) { http_json(fd, 500, "{\"error\":\"gpu-profile-query failed\"}"); return; }

    char out[32768] = {0};
    int total = 0;
    char buf[1024];
    while (fgets(buf, sizeof(buf), fp) && total < (int)sizeof(out) - 1024) {
        int l = (int)strlen(buf);
        memcpy(out + total, buf, l);
        total += l;
    }
    pclose(fp);
    http_send(fd, 200, "text/plain", out, total);
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* TRAINING                                                                   */
/* ═══════════════════════════════════════════════════════════════════════════ */

static char S_log_path[1024] = {0};

static void start_training(int fd, const char* body) {
    if (S.train_pid > 0) { http_json(fd, 400, "{\"error\":\"already running\"}"); return; }

    char model[512] = {0}, data[512] = {0}, flags[1024] = {0};
    json_str(body, "model", model, sizeof(model));
    json_str(body, "data", data, sizeof(data));
    json_str(body, "flags", flags, sizeof(flags));

    if (!model[0] || !data[0]) { http_json(fd, 400, "{\"error\":\"need model and data\"}"); return; }

    /* log file: write all stdout+stderr here, browser polls it */
    snprintf(S_log_path, sizeof(S_log_path), "%s/viz-train.log", S.base_dir);
    truncate(S_log_path, 0);

    pid_t pid = fork();
    if (pid < 0) { http_json(fd, 500, "{\"error\":\"fork failed\"}"); return; }

    if (pid == 0) {
        /* child: redirect stdout+stderr to log file */
        int logfd = open(S_log_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (logfd >= 0) {
            dup2(logfd, STDOUT_FILENO);
            dup2(logfd, STDERR_FILENO);
            close(logfd);
        }
        /* build argv directly — no shell, so kill(pid) kills the actual binary */
        char bin[1024];
        snprintf(bin, sizeof(bin), "%s/seraph-train-cuda", S.base_dir);

        /* tokenize flags into argv (max 64 args) */
        char* argv[64];
        int argc = 0;
        argv[argc++] = bin;
        argv[argc++] = model;
        argv[argc++] = data;
        argv[argc++] = (char*)"--profile";
        argv[argc++] = (char*)"999999999";

        /* split flags string by spaces into argv */
        char flags_buf[1024];
        strncpy(flags_buf, flags, sizeof(flags_buf) - 1);
        flags_buf[sizeof(flags_buf) - 1] = 0;
        char* tok = strtok(flags_buf, " ");
        while (tok && argc < 62) {
            argv[argc++] = tok;
            tok = strtok(NULL, " ");
        }
        argv[argc] = NULL;

        execv(bin, argv);
        _exit(1);
    }

    S.train_pid  = pid;
    S.train_pipe = -1;

    fprintf(stderr, "  training started: pid=%d log=%s\n", pid, S_log_path);
    http_json(fd, 200, "{\"ok\":true}");
}

static void stop_training(int fd) {
    if (S.train_pid <= 0) { http_json(fd, 400, "{\"error\":\"not running\"}"); return; }
    fprintf(stderr, "  stopping training: pid=%d\n", S.train_pid);
    kill(S.train_pid, SIGTERM);
    usleep(300000);
    if (kill(S.train_pid, 0) == 0) kill(S.train_pid, SIGKILL);
    waitpid(S.train_pid, NULL, 0);
    S.train_pid = -1;
    http_json(fd, 200, "{\"ok\":true}");
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* PIPE READER                                                                */
/* ═══════════════════════════════════════════════════════════════════════════ */

static void drain_pipe(void) {
    if (S.train_pipe < 0) return;
    char buf[4096];
    int n = (int)read(S.train_pipe, buf, sizeof(buf) - 1);
    if (n <= 0) {
        if (n == 0 || (errno != EAGAIN && errno != EWOULDBLOCK)) {
            close(S.train_pipe);
            S.train_pipe = -1;
            /* reap child */
            if (S.train_pid > 0) {
                waitpid(S.train_pid, NULL, 0);
                S.train_pid = -1;
            }
            if (S.line_len > 0) {
                S.line_buf[S.line_len] = 0;
                sse_broadcast(S.line_buf);
                S.line_len = 0;
            }
            sse_broadcast("{\"type\":\"finished\"}");
        }
        return;
    }
    for (int i = 0; i < n; i++) {
        if (buf[i] == '\n' || S.line_len >= LINE_BUF - 2) {
            S.line_buf[S.line_len] = 0;
            if (S.line_len > 0) sse_broadcast(S.line_buf);
            S.line_len = 0;
        } else if (buf[i] != '\r') {
            S.line_buf[S.line_len++] = buf[i];
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* REQUEST HANDLING                                                           */
/* ═══════════════════════════════════════════════════════════════════════════ */

static int read_request(int fd, char* buf, int bufsize) {
    struct timeval tv = {0, 200000}; /* 200ms timeout — don't block the pipe */
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    int total = 0;
    while (total < bufsize - 1) {
        int r = (int)recv(fd, buf + total, bufsize - 1 - total, 0);
        if (r <= 0) break;
        total += r;
        buf[total] = 0;

        char* hend = strstr(buf, "\r\n\r\n");
        if (hend) {
            char* cl = find_header(buf, "Content-Length:");
            if (!cl) return total;
            int clen = atoi(cl + 15);
            int hlen = (int)(hend - buf) + 4;
            if (total - hlen >= clen) return total;
        }
    }
    return total;
}

static void handle_connection(int fd) {
    char buf[REQ_BUF];
    memset(buf, 0, 64); /* ensure sscanf has safe input even on short read */
    int n = read_request(fd, buf, sizeof(buf));
    if (n <= 0) { close(fd); return; }

    char method[16] = {0}, url[512] = {0};
    if (sscanf(buf, "%15s %511s", method, url) < 2) {
        close(fd);
        return;
    }

    char* query = strchr(url, '?');
    if (query) *query++ = 0;

    char* body = strstr(buf, "\r\n\r\n");
    if (body) body += 4;

    if (!strcmp(method, "GET")) {
        if (!strcmp(url, "/api/stream")) {
            const char* hdr =
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: text/event-stream\r\n"
                "Cache-Control: no-cache\r\n"
                "Connection: keep-alive\r\n\r\n";
            send(fd, hdr, (int)strlen(hdr), MSG_NOSIGNAL);
            fcntl(fd, F_SETFL, O_NONBLOCK);
            if (S.n_sse < MAX_SSE) {
                S.sse_fds[S.n_sse++] = fd;
                fprintf(stderr, "  SSE client connected (%d total)\n", S.n_sse);
            } else { close(fd); }
            return;
        }
        if (!strcmp(url, "/api/gpu-info")) {
            char model[512] = {0};
            query_param(query, "model", model, sizeof(model));
            serve_gpu_info(fd, model);
            close(fd); return;
        }
        if (!strcmp(url, "/api/browse")) {
            char dir[1024] = {0};
            query_param(query, "path", dir, sizeof(dir));
            serve_browse(fd, dir);
            close(fd); return;
        }
        if (!strcmp(url, "/api/status")) {
            /* check if training process is actually alive */
            if (S.train_pid > 0 && kill(S.train_pid, 0) != 0) {
                waitpid(S.train_pid, NULL, WNOHANG);
                S.train_pid = -1;
            }
            char json[256];
            snprintf(json, sizeof(json), "{\"training\":%s,\"pid\":%d}",
                S.train_pid > 0 ? "true" : "false", S.train_pid);
            http_json(fd, 200, json);
            close(fd); return;
        }
        if (!strcmp(url, "/api/train/output")) {
            struct stat st;
            if (S_log_path[0] && stat(S_log_path, &st) == 0) {
                FILE* lf = fopen(S_log_path, "rb");
                if (lf) {
                    char* buf = (char*)malloc(st.st_size + 1);
                    size_t n = fread(buf, 1, st.st_size, lf);
                    fclose(lf);
                    http_send(fd, 200, "text/plain", buf, (int)n);
                    free(buf);
                } else {
                    http_send(fd, 200, "text/plain", "", 0);
                }
            } else {
                http_send(fd, 200, "text/plain", "", 0);
            }
            close(fd); return;
        }
        serve_file(fd, url);
        close(fd); return;
    }

    if (!strcmp(method, "POST")) {
        if (!strcmp(url, "/api/train/start")) { start_training(fd, body ? body : ""); close(fd); return; }
        if (!strcmp(url, "/api/train/stop"))  { stop_training(fd); close(fd); return; }
    }

    if (!strcmp(method, "OPTIONS")) {
        const char* hdr =
            "HTTP/1.1 204 No Content\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Methods: GET, POST\r\n"
            "Access-Control-Allow-Headers: Content-Type\r\n\r\n";
        send(fd, hdr, (int)strlen(hdr), MSG_NOSIGNAL);
        close(fd); return;
    }

    http_send(fd, 404, "text/plain", "not found", 9);
    close(fd);
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* MAIN                                                                       */
/* ═══════════════════════════════════════════════════════════════════════════ */

int main(int argc, char** argv) {
    S.port = DEFAULT_PORT;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--port") && i + 1 < argc) S.port = atoi(argv[++i]);
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            printf("usage: seraph-gpu-viz [--port %d]\n", DEFAULT_PORT);
            return 0;
        }
    }

    /* discover paths from binary location */
    char resolved[512];
    if (realpath(argv[0], resolved)) {
        char* slash = strrchr(resolved, '/');
        if (slash) *slash = 0;
        snprintf(S.base_dir,   sizeof(S.base_dir),   "%s", resolved);
        snprintf(S.static_dir, sizeof(S.static_dir), "%s/tools/gpu-viz", resolved);
    } else {
        strcpy(S.base_dir, ".");
        strcpy(S.static_dir, "./tools/gpu-viz");
    }

    struct stat st;
    if (stat(S.static_dir, &st) || !S_ISDIR(st.st_mode)) {
        fprintf(stderr, "error: static dir not found: %s\n", S.static_dir);
        return 1;
    }

    signal(SIGINT,  on_signal);
    signal(SIGTERM, on_signal);
    signal(SIGPIPE, SIG_IGN);

    /* create listen socket — IPv6 dual-stack (accepts both ::1 and 127.0.0.1) */
    S.listen_fd = socket(AF_INET6, SOCK_STREAM, 0);
    if (S.listen_fd < 0) { perror("socket"); return 1; }

    int opt = 1;
    setsockopt(S.listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    setsockopt(S.listen_fd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));
    int off = 0;
    setsockopt(S.listen_fd, IPPROTO_IPV6, IPV6_V6ONLY, &off, sizeof(off));

    struct sockaddr_in6 addr = {0};
    addr.sin6_family = AF_INET6;
    addr.sin6_port   = htons(S.port);
    addr.sin6_addr   = in6addr_loopback;

    if (bind(S.listen_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind");
        return 1;
    }
    if (listen(S.listen_fd, 64) < 0) {
        perror("listen");
        return 1;
    }

    write_pid_file();

    printf("\n");
    printf("  \033[36m══════════════════════════════════════════\033[0m\n");
    printf("  \033[36m  SERAPH GPU VISUALIZER\033[0m\n");
    printf("  \033[36m  http://localhost:%d\033[0m\n", S.port);
    printf("  \033[36m══════════════════════════════════════════\033[0m\n");
    printf("\n");
    printf("  base:   %s\n", S.base_dir);
    printf("  static: %s\n\n", S.static_dir);
    fflush(stdout);

    /* ── main event loop ── */
    while (S.running) {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(S.listen_fd, &rfds);
        int maxfd = S.listen_fd;

        if (S.train_pipe >= 0) {
            FD_SET(S.train_pipe, &rfds);
            if (S.train_pipe > maxfd) maxfd = S.train_pipe;
        }

        struct timeval tv = {0, 50000};
        int ready = select(maxfd + 1, &rfds, NULL, NULL, &tv);

        /* reap child */
        if (S.train_pid > 0) {
            int status;
            if (waitpid(S.train_pid, &status, WNOHANG) > 0) {
                fprintf(stderr, "  training finished (exit=%d)\n",
                    WIFEXITED(status) ? WEXITSTATUS(status) : -1);
                S.train_pid = -1;
                drain_pipe();
            }
        }

        if (ready <= 0) continue;

        /* accept all pending connections */
        if (FD_ISSET(S.listen_fd, &rfds)) {
            for (int c = 0; c < 16; c++) {
                struct sockaddr_in6 ca;
                socklen_t cl = sizeof(ca);
                int fd = accept(S.listen_fd, (struct sockaddr*)&ca, &cl);
                if (fd < 0) break;
                handle_connection(fd);
            }
        }
    }

    cleanup();
    printf("\n  GPU Visualizer stopped.\n\n");
    return 0;
}
