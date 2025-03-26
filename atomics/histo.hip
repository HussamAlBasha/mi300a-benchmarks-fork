#include "hip/hip_runtime.h"
#include <iostream>
#include <random>
#include <thread>
#include <unistd.h>
#include "../util.h"

template<typename T, bool system>
__global__ void kernel(T *table, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        size_t k = 0;

        if (system)
            atomicAdd_system(table+k, 1);
        else
            atomicAdd(table+k, 1);
    }
}

template<typename T>
bool run(size_t table_size, size_t total, int threads_cpu, int threads_gpu, int tpb)
{
    T *table = new T[table_size];

    for (size_t i = 0; i < table_size; i++)
        table[i] = 0;

    if (threads_gpu) {
        int blocks = threads_gpu / tpb;
        CHECK(blocks*tpb == threads_gpu);
        size_t n_per_thread = total / threads_gpu;
        if (threads_cpu && threads_gpu)
            n_per_thread /= 2;

        kernel<T, true><<<blocks, tpb>>>(table, n_per_thread);
    }

    int order = __ATOMIC_SEQ_CST;

    std::vector<std::thread> threads;
    if (threads_cpu) {
        size_t thread_total = total / threads_cpu;
        if (threads_cpu && threads_gpu)
            thread_total /= 2;

        for (int tid = 0; tid < threads_cpu; tid++) {
            threads.emplace_back([table, order, table_size](size_t thread_total) {
                std::random_device seed;
                std::minstd_rand rand(seed());
                std::uniform_int_distribution<size_t> distrib(0, table_size-1);

                for (size_t i = 0; i < thread_total; i++) {
                    size_t k = distrib(rand);
                    __atomic_fetch_add(table+k, 1, order);
                }
            }, thread_total);
        }
    }

    if (threads_gpu)
        CHECK_HIP(hipDeviceSynchronize());

    for (auto& thread: threads)
        thread.join();

    T sum = 0;
    for (size_t i = 0; i < table_size; i++)
        sum += table[i];

    bool ok = true;
    if (sum != total) {
        std::cerr << "validation failed: sum = " << sum << std::endl;
        ok = false;
    }
    
    delete[] table;

    return ok;
}

int main(int argc, char **argv)
{
    size_t table_size = 1024;
    size_t total = 1000000;

    int threads_cpu = 0;
    int threads_gpu = 0;

    int tpb = 1;

    const char *type = "u32";

    const char *opts = "c:g:s:t:d:";
    for (int opt; (opt = getopt(argc, argv, opts)) != -1;) {
        switch (opt) {
            case 'c':
                threads_cpu = atoi(optarg);
                break;
            case 'g':
                threads_gpu = atoi(optarg);
                break;
            case 's':
                table_size = strtoul(optarg, NULL, 0);
                break;
            case 't':
                total = strtoul(optarg, NULL, 0);
                break;
            case 'd':
                type = optarg;
                break;
            default:
                std::cerr << "usage: " << argv[0]
                    << " [-c threads_cpu] [-g threads_gpu]"
                    << " [-s table_size] [-t total]"
                    << " [-d datatype]"
                    << std::endl;
                return 1;
        }
    }

    bool ok;
    if (!strcmp(type, "u32"))
        ok = run<uint32_t>(table_size, total, threads_cpu, threads_gpu, tpb);
    else if (!strcmp(type, "u64"))
        ok = run<uint64_t>(table_size, total, threads_cpu, threads_gpu, tpb);
    else if (!strcmp(type, "f32"))
        ok = run<float>(table_size, total, threads_cpu, threads_gpu, tpb);
    else if (!strcmp(type, "f64"))
        ok = run<double>(table_size, total, threads_cpu, threads_gpu, tpb);
    else {
        std::cerr << "unknown datatype '" << type << "'" << std::endl;
        return 1;
    }

    if (!ok)
        return 1;
}
