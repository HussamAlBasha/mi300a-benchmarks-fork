/*
 * Customized FP64 TRIAD worker.
 *
 * The TRIAD expression and indexing are based on hip-stream/stream.hip from
 * KTH ScaLab mi300a-benchmarks, commit 4e491486ca8313d0e73995c3581ce2ed3686c3dc.
 *
 * Original STREAM work: John D. McCalpin
 * CUDA implementation: Massimiliano Fatica
 * Further modifications: Ben Cumming
 * HIP port: Peng Sun
 *
 * The coordination, tracing, timing, and validation in this file were added
 * for the controlled partition experiment. The upstream copyright and MIT
 * notice remain in ../../hip-stream/stream.hip. See also ../../LICENSE.
 */
#include <hip/hip_runtime.h>
#include <rocprofiler-sdk-roctx/roctx.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sched.h>
#include <unistd.h>
#include <time.h>
#include <signal.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Throw if a HIP call did not return hipSuccess.
static void checked(hipError_t e) {
    if (e != hipSuccess) {
        throw std::runtime_error(hipGetErrorString(e));
    }
}

// Monotonic timestamp in nanoseconds, used for all host-side timing.
static int64_t now_ns() {
    timespec t{};
    if (clock_gettime(CLOCK_MONOTONIC, &t)) {
        throw std::runtime_error("clock_gettime");
    }
    return int64_t(t.tv_sec) * 1000000000LL + t.tv_nsec;
}

// Wrap a string in double quotes and escape it for JSON output.
static std::string quote(const std::string& s) {
    std::string r = "\"";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') {
            r += '\\';
            r += char(c);
        } else if (c < 32) {
            char b[7];
            snprintf(b, sizeof(b), "\\u%04x", c);
            r += b;
        } else {
            r += char(c);
        }
    }
    return r + '"';
}

// Read an environment variable, returning "" when it is unset.
static std::string env(const char* key) {
    const char* v = getenv(key);
    return v ? v : "";
}

// JSON array of the CPU cores this process is allowed to run on.
static std::string affinity() {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set)) {
        throw std::runtime_error("sched_getaffinity");
    }
    std::string r = "[";
    bool first = true;
    for (int i = 0; i < CPU_SETSIZE; i++) {
        if (CPU_ISSET(i, &set)) {
            if (!first) {
                r += ',';
            }
            r += std::to_string(i);
            first = false;
        }
    }
    return r + "]";
}

// Normalized PCI address (domain:bus:device.function) of a HIP device.
static std::string bdf(int device) {
    char raw[64]{};
    checked(hipDeviceGetPCIBusId(raw, sizeof(raw), device));
    unsigned d, b, s, f;
    if (sscanf(raw, "%x:%x:%x.%x", &d, &b, &s, &f) != 4) {
        throw std::runtime_error("invalid HIP BDF");
    }
    char normalized[64];
    snprintf(normalized, sizeof(normalized), "%04x:%02x:%02x.%x", d, b, s, f);
    return normalized;
}

// JSON object describing a HIP device (ordinal, PCI address, CU count, arch, name).
static std::string identity(int d) {
    hipDeviceProp_t p{};
    checked(hipGetDeviceProperties(&p, d));
    std::ostringstream s;
    s << "{\"ordinal\":" << d
      << ",\"bdf\":" << quote(bdf(d))
      << ",\"cus\":" << p.multiProcessorCount
      << ",\"arch\":" << quote(p.gcnArchName)
      << ",\"name\":" << quote(p.name) << "}";
    return s.str();
}

// Seed the three arrays with the STREAM TRIAD starting values: a=0, b=2, c=3.
__global__ void initialize(double* a, double* b, double* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] = 0.0;
        b[i] = 2.0;
        c[i] = 3.0;
    }
}

// STREAM TRIAD kernel: each element does a[i] = b[i] + scalar*c[i], with scalar = 3.0.
__global__ void controlled_triad(double* a, const double* b, const double* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] = b[i] + 3.0 * c[i];
    }
}

// Count elements that are non-finite or differ from the expected result (a=11, b=2, c=3).
__global__ void validate_arrays(const double* a, const double* b, const double* c, int n, unsigned long long* bad) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n &&
        (!isfinite(a[i]) || !isfinite(b[i]) || !isfinite(c[i]) ||
         a[i] != 11.0 || b[i] != 2.0 || c[i] != 3.0)) {
        atomicAdd(bad, 1ULL);
    }
}
// Owns the three arrays and frees each with the API that matches how it was allocated.
struct Arrays {
    std::string allocator;
    double* p[3]{nullptr, nullptr, nullptr};

    Arrays(std::string a, size_t bytes) : allocator(std::move(a)) {
        try {
            for (auto& v : p) {
                if (allocator == "malloc") {
                    v = static_cast<double*>(malloc(bytes));
                    if (!v) {
                        throw std::runtime_error("malloc failed");
                    }
                } else if (allocator == "hipMalloc") {
                    checked(hipMalloc(reinterpret_cast<void**>(&v), bytes));
                } else if (allocator == "hipMallocManaged") {
                    checked(hipMallocManaged(reinterpret_cast<void**>(&v), bytes, hipMemAttachGlobal));
                } else if (allocator == "hipHostMalloc") {
                    checked(hipHostMalloc(reinterpret_cast<void**>(&v), bytes, hipHostMallocDefault));
                } else {
                    throw std::runtime_error("unknown allocator");
                }
            }
        } catch (...) {
            release();
            throw;
        }
    }

    // Free each pointer with the API matching the allocator, then null it out.
    void release() noexcept {
        for (auto& v : p) {
            if (v) {
                if (allocator == "malloc") {
                    free(v);
                } else if (allocator == "hipHostMalloc") {
                    hipHostFree(v);
                } else {
                    hipFree(v);
                }
                v = nullptr;
            }
        }
    }

    ~Arrays() {
        release();
    }
};
struct Calibration {
    std::string label;
    int64_t before, after;
};

struct Batch {
    int64_t before, after;
    int launches;
};

// Record a roctx marker bracketed by host timestamps, for later timeline calibration.
static void marker(std::vector<Calibration>& records, const std::string& label) {
    auto before = now_ns();
    roctxMarkA(label.c_str());
    auto after = now_ns();
    records.push_back({label, before, after});
}

// Send one newline-terminated message over the coordinator socket.
static void send_line(int fd, const std::string& data) {
    std::string wire = data + '\n';
    size_t pos = 0;
    while (pos < wire.size()) {
        auto n = send(fd, wire.data() + pos, wire.size() - pos, MSG_NOSIGNAL);
        if (n <= 0) {
            throw std::runtime_error("coordinator disconnected");
        }
        pos += size_t(n);
    }
}

// Read one newline-terminated message from the coordinator socket.
static std::string receive_line(int fd) {
    std::string s;
    char c;
    while (s.size() < 4096) {
        auto n = recv(fd, &c, 1, 0);
        if (n != 1) {
            throw std::runtime_error("coordinator read failed");
        }
        if (c == '\n') {
            return s;
        }
        s += c;
    }
    throw std::runtime_error("oversized coordinator message");
}

// Write content plus a trailing newline to a file, failing on I/O error.
static void write_file(const std::string& path, const std::string& content) {
    std::ofstream f(path);
    f << content << '\n';
    f.close();
    if (!f) {
        throw std::runtime_error("output write failed");
    }
}

// Parse a positive-integer CLI argument, or return fallback when it is absent.
static int number(const std::map<std::string, std::string>& args, const std::string& key, int fallback) {
    auto i = args.find(key);
    if (i == args.end()) {
        return fallback;
    }
    size_t used;
    long long v = std::stoll(i->second, &used);
    if (used != i->second.size() || v < 1 || v > std::numeric_limits<int>::max()) {
        throw std::runtime_error("invalid " + key);
    }
    return int(v);
}
int main(int argc, char** argv) {
    std::string output;
    int fd = -1;
    try {
        // --probe: print the JSON identity of every visible GPU and exit.
        if (argc == 2 && std::string(argv[1]) == "--probe") {
            int count;
            checked(hipGetDeviceCount(&count));
            std::cout << '[';
            for (int d = 0; d < count; d++) {
                if (d) {
                    std::cout << ',';
                }
                std::cout << identity(d);
            }
            std::cout << "]\n";
            return 0;
        }

        // Parse CLI arguments (each flag takes one value; unknown or duplicate flags are rejected).
        std::map<std::string, std::string> args;
        const std::vector<std::string> allowed = {
            "--socket", "--output", "--allocator", "--elements", "--block-size",
            "--batch-size", "--expected-bdf", "--expected-cus", "--worker-id"};
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 >= argc ||
                std::find(allowed.begin(), allowed.end(), argv[i]) == allowed.end() ||
                args.count(argv[i])) {
                throw std::runtime_error("invalid/duplicate argument");
            }
            args[argv[i]] = argv[i + 1];
        }
        for (const char* key : {"--socket", "--output", "--allocator", "--expected-bdf", "--expected-cus", "--worker-id"}) {
            if (!args.count(key)) {
                throw std::runtime_error(std::string("missing ") + key);
            }
        }
        output = args.at("--output");
        std::string alloc = args.at("--allocator");

        // Require an explicit XNACK setting; plain malloc only works with XNACK=1.
        if (env("HSA_XNACK") != "0" && env("HSA_XNACK") != "1") {
            throw std::runtime_error("explicit HSA_XNACK required");
        }
        if (alloc == "malloc" && env("HSA_XNACK") != "1") {
            throw std::runtime_error("malloc requires XNACK=1");
        }

        // Require exactly one GPU, selected through the ROCr mask only.
        if (env("ROCR_VISIBLE_DEVICES").empty() ||
            !env("HIP_VISIBLE_DEVICES").empty() ||
            !env("CUDA_VISIBLE_DEVICES").empty()) {
            throw std::runtime_error("use one explicit ROCr mask only");
        }
        int count;
        checked(hipGetDeviceCount(&count));
        if (count != 1) {
            throw std::runtime_error("worker must see exactly one GPU");
        }
        checked(hipSetDevice(0));

        // Confirm the selected device matches the identity the coordinator expects.
        hipDeviceProp_t prop{};
        checked(hipGetDeviceProperties(&prop, 0));
        if (bdf(0) != args.at("--expected-bdf") ||
            prop.multiProcessorCount != number(args, "--expected-cus", 0)) {
            throw std::runtime_error("selected HIP identity mismatch");
        }
        if (std::string(prop.gcnArchName).find(env("HSA_XNACK") == "1" ? "xnack+" : "xnack-") == std::string::npos) {
            throw std::runtime_error("selected HIP XNACK architecture mismatch");
        }

        // Launch geometry and array allocation.
        int n = number(args, "--elements", 33554432);
        int block = number(args, "--block-size", 192);
        int batch = number(args, "--batch-size", 8);
        if (block > prop.maxThreadsPerBlock ||
            int64_t(n) + block > std::numeric_limits<int>::max()) {
            throw std::runtime_error("invalid launch geometry");
        }
        int grid = (n + block - 1) / block;
        Arrays arrays(alloc, size_t(n) * sizeof(double));

        // Initialize the arrays, then run 20 warmup TRIAD launches before measuring.
        initialize<<<grid, block>>>(arrays.p[0], arrays.p[1], arrays.p[2], n);
        checked(hipGetLastError());
        checked(hipDeviceSynchronize());
        for (int i = 0; i < 20; i++) {
            controlled_triad<<<grid, block>>>(arrays.p[0], arrays.p[1], arrays.p[2], n);
        }
        checked(hipGetLastError());
        checked(hipDeviceSynchronize());

        // Announce readiness to the coordinator over a Unix-domain socket.
        std::vector<Calibration> calibrations;
        marker(calibrations, "rq5_ready_" + args.at("--worker-id"));
        fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (fd < 0) {
            throw std::runtime_error("socket failed");
        }
        timeval timeout{30, 0};
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        sockaddr_un addr{};
        addr.sun_family = AF_UNIX;
        if (args.at("--socket").size() >= sizeof(addr.sun_path)) {
            throw std::runtime_error("socket path too long");
        }
        strcpy(addr.sun_path, args.at("--socket").c_str());
        if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr))) {
            throw std::runtime_error("connect failed");
        }
        send_line(fd, "{\"state\":\"ready\",\"worker\":" + quote(args.at("--worker-id")) +
                          ",\"device\":" + identity(0) + ",\"affinity\":" + affinity() + "}");

        // Receive the shared measurement schedule and validate its time ordering.
        std::istringstream message(receive_line(fd));
        std::string action;
        int64_t lead, t0, t1, tail;
        message >> action >> lead >> t0 >> t1 >> tail;
        if (!message || action != "START" ||
            !(now_ns() < lead && lead < t0 && t0 < t1 && t1 < tail)) {
            throw std::runtime_error("invalid/late measurement schedule");
        }

        // Spin until the shared lead-in time, then run TRIAD batches until the tail deadline.
        while (now_ns() < lead) {
            timespec nap{0, 100000};
            nanosleep(&nap, nullptr);
        }
        std::vector<Batch> batches;
        uint64_t launches = 0;
        while (now_ns() < tail) {
            int64_t before = now_ns();
            for (int i = 0; i < batch; i++) {
                controlled_triad<<<grid, block>>>(arrays.p[0], arrays.p[1], arrays.p[2], n);
            }
            checked(hipGetLastError());
            checked(hipDeviceSynchronize());
            batches.push_back({before, now_ns(), batch});
            launches += batch;
        }
        marker(calibrations, "rq5_done_" + args.at("--worker-id"));

        // Validate the array contents on the device and copy the error count back to the host.
        unsigned long long* bad = nullptr;
        checked(hipMalloc(reinterpret_cast<void**>(&bad), sizeof(*bad)));
        checked(hipMemset(bad, 0, sizeof(*bad)));
        validate_arrays<<<grid, block>>>(arrays.p[0], arrays.p[1], arrays.p[2], n, bad);
        checked(hipGetLastError());
        checked(hipDeviceSynchronize());
        unsigned long long errors = 0;
        checked(hipMemcpy(&errors, bad, sizeof(errors), hipMemcpyDeviceToHost));
        checked(hipFree(bad));

        // Emit the result JSON, notify the coordinator, and exit.
        std::ostringstream result;
        result << "{\"schema_version\":1,\"worker\":" << quote(args.at("--worker-id"))
               << ",\"status\":" << quote(errors ? "invalid" : "ok")
               << ",\"validation_errors\":" << errors
               << ",\"device\":" << identity(0)
               << ",\"affinity\":" << affinity()
               << ",\"rocr_mask\":" << quote(env("ROCR_VISIBLE_DEVICES"))
               << ",\"xnack\":" << env("HSA_XNACK")
               << ",\"allocator\":" << quote(alloc)
               << ",\"allocator_flag\":" << quote(alloc == "hipHostMalloc" ? "hipHostMallocDefault" : "API default")
               << ",\"elements\":" << n
               << ",\"element_bytes\":8,\"block_size\":" << block
               << ",\"grid_blocks\":" << grid
               << ",\"batch_size\":" << batch
               << ",\"warmup_launches\":20,\"measured_phase_launches\":" << launches
               << ",\"t0_ns\":" << t0
               << ",\"t1_ns\":" << t1
               << ",\"calibrations\":[";
        for (size_t i = 0; i < calibrations.size(); i++) {
            if (i) {
                result << ',';
            }
            auto& c = calibrations[i];
            result << "{\"label\":" << quote(c.label)
                   << ",\"before_ns\":" << c.before
                   << ",\"after_ns\":" << c.after << "}";
        }
        result << "],\"batches\":[";
        for (size_t i = 0; i < batches.size(); i++) {
            if (i) {
                result << ',';
            }
            auto& b = batches[i];
            result << "{\"before_ns\":" << b.before
                   << ",\"after_ns\":" << b.after
                   << ",\"launches\":" << b.launches << "}";
        }
        result << "]}";
        write_file(output, result.str());
        send_line(fd, "{\"state\":\"done\"}");
        close(fd);
        return errors ? 2 : 0;
    } catch (const std::exception& e) {
        if (fd >= 0) {
            close(fd);
        }
        std::cerr << "controlled TRIAD failed: " << e.what() << '\n';
        if (!output.empty()) {
            try {
                write_file(output, "{\"status\":\"failed\",\"error\":" + quote(e.what()) + "}");
            } catch (...) {
            }
        }
        return 1;
    }
}
