#include <hip/hip_runtime.h>
#include <rocrand/rocrand.h>
#include <rocrand/rocrand_xorwow.h>
#include <iostream>
#include <random>
#include <thread>
#include <unistd.h>
#include <stdint.h>
#include <time.h>
#include <signal.h>
#include "../util.h"

const int GRANUL = 200;

template<typename T, bool system>
__global__ void gpu_kernel(T *table, size_t table_size, size_t n, unsigned long long seed, size_t *count)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    rocrand_state_xorwow rand;
    rocrand_init(seed, tid, 0, &rand);

    for (size_t i = 0; i < n; i++) {
        size_t k = rocrand(&rand) % table_size;

        if (system)
            atomicAdd_system(table+k, 1);
        else
            atomicAdd(table+k, 1);

        if (GRANUL && (i % GRANUL == 0))
            atomicAdd_system(count, 1);
    }
}

template<typename T, int order>
void cpu_kernel(T *table, size_t table_size, size_t thread_total, size_t *count)
{
    std::random_device seed;
    std::minstd_rand rand(seed());
    std::uniform_int_distribution<size_t> distrib(0, table_size-1);

    for (size_t i = 0; i < thread_total; i++) {
        size_t k = distrib(rand);
        __atomic_fetch_add(table+k, 1, order);

        if (GRANUL && (i % GRANUL == 0))
            __atomic_fetch_add(count, 1, __ATOMIC_SEQ_CST);
    }
}

struct spec {
    size_t table_size;
    size_t total;
    int threads_cpu;
    int threads_gpu;
    int tpb;
    bool verbose;
    bool system;
};

template<typename T>
bool run(spec spec)
{
    T *table = new T[spec.table_size];

    // Cannot allocate on stack with total=0 since kernel runs past return of
    // this function.
    auto *gpu_count = new size_t(0);
    auto *cpu_count = new size_t(0);

    for (size_t i = 0; i < spec.table_size; i++)
        table[i] = 0;

    if (spec.threads_gpu) {
        int blocks = spec.threads_gpu / spec.tpb;
        CHECK(blocks*spec.tpb == spec.threads_gpu);

        size_t n_per_thread = SIZE_MAX;
        if (spec.total) {
            n_per_thread = spec.total / spec.threads_gpu;
            if (spec.threads_cpu && spec.threads_gpu)
                n_per_thread /= 2;
        }

        std::random_device seed;

        if (spec.system)
            gpu_kernel<T, true> <<<blocks, spec.tpb>>>(table, spec.table_size, n_per_thread, seed(), gpu_count);
        else
            gpu_kernel<T, false><<<blocks, spec.tpb>>>(table, spec.table_size, n_per_thread, seed(), gpu_count);
    }

    std::vector<std::thread> threads;
    if (spec.threads_cpu) {
        size_t thread_total = SIZE_MAX;
        if (spec.total) {
            thread_total = spec.total / spec.threads_cpu;
            if (spec.threads_cpu && spec.threads_gpu)
                thread_total /= 2;
        }

        for (int tid = 0; tid < spec.threads_cpu; tid++) {
            threads.emplace_back(cpu_kernel<T, __ATOMIC_SEQ_CST>, table, spec.table_size, thread_total, cpu_count);

            if (!spec.total)
                threads.back().detach();
        }
    }

    bool ok = true;

    if (spec.total) {
        if (spec.threads_gpu)
            CHECK_HIP(hipDeviceSynchronize());

        for (auto& thread: threads)
            thread.join();

        T sum = 0;
        for (size_t i = 0; i < spec.table_size; i++)
            sum += table[i];

        if (sum != spec.total) {
            std::cerr << "validation failed: sum = " << sum << std::endl;
            ok = false;
        } else {
            std::cerr << "ok" << std::endl;
        }

        if (spec.verbose) {
            for (size_t i = 0; i < spec.table_size; i++)
                std::cout << table[i] << " ";
            std::cout << std::endl;
        }
        
        delete[] table;
    } else {
        const uint64_t NS = 1000000000ULL;

        timespec sleep_time = {.tv_sec = 1};
        CHECK(!nanosleep(&sleep_time, NULL));

        timespec t0;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        size_t gpu0 = __atomic_load_n(gpu_count, __ATOMIC_SEQ_CST);
        size_t cpu0 = __atomic_load_n(cpu_count, __ATOMIC_SEQ_CST);

        CHECK(!nanosleep(&sleep_time, NULL));

        size_t gpu1 = __atomic_load_n(gpu_count, __ATOMIC_SEQ_CST);
        size_t cpu1 = __atomic_load_n(cpu_count, __ATOMIC_SEQ_CST);

        timespec t1;
        clock_gettime(CLOCK_MONOTONIC, &t1);

        uint64_t t_ns = NS*(t1.tv_sec-t0.tv_sec) + (t1.tv_nsec-t0.tv_nsec);
        double t_s = t_ns / (double)NS;

        double gpu_rate = GRANUL*(gpu1-gpu0)/t_s;
        double cpu_rate = GRANUL*(cpu1-cpu0)/t_s;

        std::cout << "gpu: " << gpu_rate << std::endl;
        std::cout << "cpu: " << cpu_rate << std::endl;
    }

    return ok;
}

void usage(const char *argv0)
{
    std::cerr
        << "usage: " << argv0 << " [options]\n"
        << "\n"
        << " -c threads_cpu\n"
        << " -g threads_gpu\n"
        << " -b gpu_tpb\n"
        << " -s table_size\n"
        << " -t total_updates   (0 for timing)\n"
        << " -d data_type       (u32, u64, f32, f64)\n"
        << " -v                 verbose validation\n"
        << " -S                 no system-wide atomics on GPU\n"
        ;
}

int main(int argc, char **argv)
{
    spec spec = {
        .table_size = 1024,
        .total = 1000000,
        .threads_cpu = 0,
        .threads_gpu = 0,
        .tpb = 1,
        .verbose = false,
        .system = true,
    };
    const char *type = "u32";

    const char *opts = "vSc:g:s:t:d:b:";
    for (int opt; (opt = getopt(argc, argv, opts)) != -1;) {
        switch (opt) {
            case 'v':
                spec.verbose = true;
                break;
            case 'S':
                spec.system = false;
                break;
            case 'c':
                spec.threads_cpu = atoi(optarg);
                break;
            case 'g':
                spec.threads_gpu = atoi(optarg);
                break;
            case 'b':
                spec.tpb = atoi(optarg);
                break;
            case 's':
                spec.table_size = strtoul(optarg, NULL, 0);
                break;
            case 't':
                spec.total = strtoul(optarg, NULL, 0);
                break;
            case 'd':
                type = optarg;
                break;
            default:
                usage(argv[0]);
                return 1;
        }
    }

    if (spec.total) {
        size_t mult = 0;
        if (spec.threads_cpu && spec.threads_gpu)
            mult = 2 * std::lcm(spec.threads_cpu, spec.threads_gpu);
        else if (spec.threads_cpu)
            mult = spec.threads_cpu;
        else if (spec.threads_gpu)
            mult = spec.threads_gpu;

        if (mult) {
            size_t new_total = ((spec.total+mult-1) / mult) * mult;
            if (spec.total != new_total)
                std::cerr << "rounding total from " << spec.total << " to " << new_total << std::endl;
            spec.total = new_total;
        }
    }

    bool ok;
    if (!strcmp(type, "u32"))
        ok = run<uint32_t>(spec);
    else if (!strcmp(type, "u64"))
        ok = run<uint64_t>(spec);
    else if (!strcmp(type, "f32"))
        ok = run<float>(spec);
    else if (!strcmp(type, "f64"))
        ok = run<double>(spec);
    else {
        std::cerr << "unknown datatype '" << type << "'" << std::endl;
        return 1;
    }

    if (!ok)
        return 1;

    if (!spec.total && spec.threads_gpu) {
        // HIP runtime waits for kernels to finish as exit handler...
        _exit(0);
    }
}
