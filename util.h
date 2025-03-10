#pragma once

#include <iostream>
#include <errno.h>
#include <hip/hip_runtime.h>

#define FAIL(a) do { std::cerr << "FAIL: " << a << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl; abort(); } while (0)
#define CHECK(a) do { if (!(a)) FAIL("check " #a); } while (0)
#define CHECK_ERRNO(a) do { if ((a) != 0) FAIL(strerror(errno) << " in " #a); } while (0)

#define CHECK_HIP(x)                                     \
do{                                                      \
    hipError_t err = x;                                  \
    if(hipSuccess != err)                                \
         FAIL("HIP error: " << hipGetErrorString(err));  \
}while(0)
