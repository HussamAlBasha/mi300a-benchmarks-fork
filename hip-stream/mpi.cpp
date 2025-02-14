#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define CHECK_MPI(x)                                     \
do{                                                      \
    int _status = x;                                     \
    if (MPI_SUCCESS != _status) {                        \
        printf("MPI Error (%s:%d): %d\n",          	 \
         __FILE__, __LINE__, _status);                   \
        abort();                                         \
    }                                                    \
}while(0)

void wrap_mpi_init()
{
    CHECK_MPI(MPI_Init(NULL, NULL));
}

void wrap_mpi_alloc(size_t n, void **p)
{
    CHECK_MPI(MPI_Alloc_mem(n, MPI_INFO_NULL, p));
}
