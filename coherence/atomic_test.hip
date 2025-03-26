#include "hip/hip_runtime.h"
#include <iostream>
#include <unistd.h>
#include "../util.h"

template<typename T> __global__ void atomic_sum(T *x)
{
    atomicAdd(x, 1);
}

template<typename T> __global__ void basic_sum(T *x)
{
    x += 1;
}

template<typename T> void test_basic(T *x, const char *s)
{
    *x = 0;

    int blocks = 1024;
    int tpb = 1024;
    basic_sum<<<blocks, tpb>>>(x);
    CHECK_HIP(hipDeviceSynchronize());

    std::cout << s << "\t" << *x << std::endl;
}

template<typename T> void test_atomic(T *x, const char *s)
{
    *x = 0;

    int blocks = 1024;
    int tpb = 1024;
    atomic_sum<<<blocks, tpb>>>(x);
    CHECK_HIP(hipDeviceSynchronize());

    std::cout << s << "\t" << *x << std::endl;
}

#define TEST_BASIC(x) test_basic((x), #x)
#define TEST_ATOMIC(x) test_atomic((x), #x)

int main(int argc, char **argv)
{
    int ps = getpagesize();

    float *sp_fine = (float *)malloc(ps);
    float *sp_coarse = (float *)malloc(ps);
    CHECK_HIP(hipMemAdvise(sp_coarse, ps, hipMemAdviseSetCoarseGrain, 0));
    //float *sp_coarse;
    //CHECK_HIP(hipMalloc((void**)&sp_coarse, sizeof(float)));

    double *dp_fine = (double *)malloc(ps);
    double *dp_coarse = (double *)malloc(ps);
    CHECK_HIP(hipMemAdvise(dp_coarse, ps, hipMemAdviseSetCoarseGrain, 0));

    std::cout << "-- basic --" << std::endl;
    TEST_BASIC(sp_fine);
    TEST_BASIC(sp_coarse);
    TEST_BASIC(dp_fine);
    TEST_BASIC(dp_coarse);

    std::cout << "-- atomic --" << std::endl;
    TEST_ATOMIC(sp_fine);
    TEST_ATOMIC(sp_coarse);
    TEST_ATOMIC(dp_fine);
    TEST_ATOMIC(dp_coarse);
}
