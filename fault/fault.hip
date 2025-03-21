#include <iostream>
#include <hip/hip_runtime.h>
#include <unistd.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include "../util.h"

// https://theunixzoo.co.uk/blog/2021-10-14-preventing-optimisations.html
// https://github.com/google/benchmark/blob/e451e50e9b8af453f076dec10bd6890847f1624e/include/benchmark/benchmark.h#L349-L356
template <class Tp>
inline void DoNotOptimize(Tp& value) {
#if defined(__clang__)
  asm volatile("" : "+r,m"(value) : : "memory");
#else
  asm volatile("" : "+m,r"(value) : : "memory");
#endif
}

__global__ void kernel_gpu(float *a, size_t n, size_t stride)
{
    size_t id = (blockDim.x * blockIdx.x + threadIdx.x)*stride;
    if (id >= n)
        return;

    a[id] = id;
}

void kernel_cpu(float *a, size_t n, size_t stride)
{
    for (size_t id = 0; id < n; id += stride)
        a[id] = id;
}

void kernel_omp(float *a, size_t n, size_t stride)
{
#pragma omp parallel for
    for (size_t id = 0; id < n; id += stride)
        a[id] = id;
}

void flush_cpu_cache()
{
    static float *a;
    size_t bytes = 256*1024*1024;
    size_t n = bytes/sizeof(float);
    if (!a)
        a = (float *)malloc(bytes);

#pragma omp for
    for (size_t i = 0; i < n; i++)
        a[i] = i;

    DoNotOptimize(a);
}

long clock()
{
    struct timespec ts;
    CHECK(!clock_gettime(CLOCK_MONOTONIC, &ts));
    return 1000000000*ts.tv_sec + ts.tv_nsec;
}

struct spec {
    size_t pages;
    size_t stride;
    bool gpu_register;
    bool gpu_map;
    bool mlock;
    bool unmap;
    bool touch;
    bool gpu_touch;
    bool cpu_kernel;
    bool omp_kernel;
    bool fork;
};

long test(spec spec)
{
    size_t ps = getpagesize();
    size_t bytes = ps*spec.pages;

    // Use mmap to ensure that each test is independent. Otherwise malloc may
    // allocate a larger chunk of address space which is faulted in batch.
    int flags = MAP_PRIVATE|MAP_ANONYMOUS;
    if (spec.fork)
        flags = MAP_SHARED|MAP_ANONYMOUS;

    float *a = (float *)mmap(NULL, bytes, PROT_READ|PROT_WRITE, flags, -1, 0);
    CHECK(a != MAP_FAILED);
    size_t n = bytes / sizeof(float);

    int child = 0;
    if (spec.fork) {
        // mlock a in a child process to get minor faults?
        int sync_pipe[2];
        CHECK(!pipe(sync_pipe));

        child = fork();
        if (child == 0) {
            CHECK(!close(sync_pipe[0]));
            CHECK(!mlock(a, bytes));

            char x = 0;
            CHECK(write(sync_pipe[1], &x, 1) == 1);
            CHECK(!close(sync_pipe[1]));

            // Wait to be killed
            for (;;)
                pause();
        } else {
            CHECK(!close(sync_pipe[1]));

            // Wait for mlock to complete in child
            char x = 0;
            CHECK(read(sync_pipe[0], &x, 1) == 1);
            CHECK(!close(sync_pipe[0]));
        }
    }

    if (spec.mlock)
        CHECK(!mlock(a, bytes));

    if (spec.touch)
#pragma omp parallel for
        for (size_t i = 0; i < n; i++)
            a[i] = 0;

    if (spec.gpu_map) {
        void *b;
        CHECK_HIP(hipHostRegister(a, bytes, hipHostRegisterMapped));
        CHECK_HIP(hipHostGetDevicePointer(&b, a, 0));
        a = (float *)b;
    } else if (spec.gpu_register) {
        CHECK_HIP(hipHostRegister(a, bytes, hipHostRegisterDefault));
    }

    size_t threads = (n+spec.stride-1) / spec.stride;
    CHECK(threads > 0);

    unsigned tpb = 256;
    unsigned blocks = (threads+tpb-1)/tpb;

    if (spec.gpu_touch) {
        kernel_gpu<<<blocks,tpb>>>(a, n, spec.stride);
        CHECK_HIP(hipDeviceSynchronize());
    }

    flush_cpu_cache();

    long t0 = clock();

    if (spec.omp_kernel) {
        kernel_omp(a, n, spec.stride);
    } else if (spec.cpu_kernel) {
        kernel_cpu(a, n, spec.stride);
    } else {
        kernel_gpu<<<blocks,tpb>>>(a, n, spec.stride);
        CHECK_HIP(hipDeviceSynchronize());
    }
    DoNotOptimize(a);

    long t1 = clock();

    if (child) {
        CHECK(!kill(child, SIGTERM));
        CHECK(child == waitpid(child, NULL, 0));
    }

    // If we don't munmap, we will run out of memory eventually
    if (spec.unmap)
        CHECK(!munmap(a, bytes));

    return (t1-t0);
}

int main(int argc, char **argv)
{
    spec spec = {
        .pages = 1,
        .stride = (size_t)getpagesize() / sizeof(float),
        .gpu_register = false,
        .gpu_map = false,
        .mlock = false,
        .unmap = true,
        .touch = false,
        .gpu_touch = false,
        .cpu_kernel = false,
        .omp_kernel = false,
        .fork = false,
    };

    for (int opt; (opt = getopt(argc, argv, "n:s:rlutTmcof")) != -1;) {
        switch (opt) {
            case 'n':
                spec.pages = atoi(optarg);
                break;
            case 's':
                spec.stride = atoi(optarg);
                break;
            case 'r':
                spec.gpu_register = true;
                break;
            case 'm':
                spec.gpu_map = true;
                break;
            case 'l':
                spec.mlock = true;
                break;
            case 'u':
                spec.unmap = false;
                break;
            case 't':
                spec.touch = true;
                break;
            case 'T':
                spec.gpu_touch = true;
                break;
            case 'c':
                spec.cpu_kernel = true;
                break;
            case 'o':
                spec.omp_kernel = true;
                break;
            case 'f':
                spec.fork = true;
                break;
            default:
                std::cerr << "usage: " << argv[0] << " [-n pages] [-s stride] [-rlutmcof]" << std::endl;
                return 1;
        }
    }

    CHECK(!(spec.cpu_kernel && spec.omp_kernel));

    // Warmup
    for (int i = 0; i < 10; i++)
        test(spec);

    for (int i = 0; i < 100; i++)
        std::cout << test(spec) << std::endl;
}
