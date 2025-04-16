#include <iostream>
#include <chrono>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include "../util.h"

void test(void *(*alloc_fn)(size_t n), void (*free_fn)(void *), size_t n, int iter, void **ps, double *atime, double *ftime)
{
    using namespace std::chrono;

    auto t0 = high_resolution_clock::now();
    for (int i = 0; i < iter; i++)
        ps[i] = alloc_fn(n);

    auto t1 = high_resolution_clock::now();

    for (int i = 0; i < iter; i++)
        free_fn(ps[i]);

    auto t2 = high_resolution_clock::now();

    *atime = duration_cast<duration<double>>(t1-t0).count()/iter;
    *ftime = duration_cast<duration<double>>(t2-t1).count()/iter;
}

void *wrap_hipMalloc(size_t n)
{
    void *p = NULL;
    CHECK_HIP(hipMalloc(&p, n));
    return p;
}

void *wrap_hipHostMalloc(size_t n)
{
    void *p = NULL;
    CHECK_HIP(hipHostMalloc(&p, n));
    return p;
}

void *wrap_hipMallocManaged(size_t n)
{
    void *p = NULL;
    CHECK_HIP(hipMallocManaged(&p, n));
    return p;
}

void wrap_hipFree(void *p)
{
    CHECK_HIP(hipFree(p));
}

int main(int argc, char **argv)
{
    size_t n = strtoul(argv[1], NULL, 0);
    int iter = atoi(argv[2]);
    int warmup = 10;

    void **ps = (void **)malloc(std::max(iter, warmup) * sizeof(void *));
    memset(ps, 0, std::max(iter, warmup) * sizeof(void *));

    void *(*alloc_fn)(size_t n);
    void (*free_fn)(void *);

    if (!strcmp(argv[3], "malloc")) {
        alloc_fn = malloc;
        free_fn = free;
    } else if (!strcmp(argv[3], "hipMalloc")) {
        alloc_fn = wrap_hipMalloc;
        free_fn = wrap_hipFree;
    } else if (!strcmp(argv[3], "hipHostMalloc")) {
        alloc_fn = wrap_hipHostMalloc;
        free_fn = wrap_hipFree;
    } else if (!strcmp(argv[3], "hipMallocManaged")) {
        alloc_fn = wrap_hipMallocManaged;
        free_fn = wrap_hipFree;
    } else {
        CHECK(!"unknown alloc method");
    }

    double atime, ftime;

    test(alloc_fn, free_fn, n, warmup, ps, &atime, &ftime);
    test(alloc_fn, free_fn, n, iter, ps, &atime, &ftime);

    std::cout << atime << "\t" << ftime << std::endl;
}
