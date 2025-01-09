#include <unistd.h>
#include "hip/hip_runtime.h"

#define CHECK_HIP(x)                                     \
do{                                                      \
    hipError_t err = x;                                  \
    if(hipSuccess != err){                               \
        printf("HIP Error (%s:%d): %s\n",          	 \
         __FILE__, __LINE__, hipGetErrorString(err));    \
        abort();                                         \
    }                                                    \
}while(0)

int main(int argc, char **argv)
{
    if (argc != 2) {
        printf("give waste size\n");
        return 1;
    }

    size_t size = strtoull(argv[1], NULL, 0);

    if (!size) {
        printf("give waste size\n");
        return 1;
    }

    size_t free = 0, total = 0;
    CHECK_HIP(hipMemGetInfo(&free, &total));
    printf("Free:   %zu\n", free);
    printf("Total:  %zu\n", total);

    void *p;
    CHECK_HIP(hipMalloc(&p, size));
    CHECK_HIP(hipMemset(p, 0, size));

    printf("Wasting %zu bytes\n", size);

    for (;;)
        pause();
}
