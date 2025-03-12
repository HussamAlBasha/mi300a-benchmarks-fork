#include <iostream>
#include <hip/hip_runtime.h>
#include <unistd.h>
#include <sys/mman.h>
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

__global__ void kernel(float *a, size_t n, size_t stride)
{
    size_t id = (blockDim.x * blockIdx.x + threadIdx.x)*stride;
    if (id >= n)
        return;

    a[id] = id;
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
    bool mlock;
    bool unmap;
};

long test(spec spec)
{
    size_t ps = getpagesize();
    size_t bytes = ps*spec.pages;

    // Use mmap to ensure that each test is independent. Otherwise malloc may
    // allocate a larger chunk of address space which is faulted in batch.
    //float *a = (float *)malloc(bytes);
    float *a = (float *)mmap(NULL, bytes, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    CHECK(a != MAP_FAILED);

    if (spec.mlock)
        CHECK(!mlock(a, bytes));

    if (spec.gpu_register)
        CHECK_HIP(hipHostRegister(a, bytes, hipHostRegisterDefault));

    size_t n = bytes / sizeof(float);
    size_t threads = (n+spec.stride-1) / spec.stride;
    CHECK(threads > 0);

    unsigned tpb = 256;
    unsigned blocks = (threads+tpb-1)/tpb;

    long t0 = clock();

    kernel<<<blocks,tpb>>>(a, n, spec.stride);
    CHECK_HIP(hipDeviceSynchronize());
    DoNotOptimize(a);

    long t1 = clock();

    // If we don't munmap, we will run out of memory eventually
    if (spec.unmap)
        CHECK(!munmap(a, bytes));

    return (t1-t0);
}

int main(int argc, char **argv)
{
    spec spec = {
        .pages = 1,
        .stride = (size_t)getpagesize(),
        .gpu_register = false,
        .mlock = false,
        .unmap = true,
    };

    for (int opt; (opt = getopt(argc, argv, "n:s:rlu")) != -1;) {
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
            case 'l':
                spec.mlock = true;
                break;
            case 'u':
                spec.unmap = false;
                break;
            default:
                std::cerr << "usage: " << argv[0] << " [-n pages] [-s stride] [-rlu]" << std::endl;
                return 1;
        }
    }

    // Warmup
    for (int i = 0; i < 10; i++)
        test(spec);

    for (int i = 0; i < 100; i++)
        std::cout << test(spec) << std::endl;
}
