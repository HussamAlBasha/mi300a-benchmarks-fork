/*
Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/*
  STREAM benchmark implementation in CUDA.

    COPY:       a(i) = b(i)
    SCALE:      a(i) = q*b(i)
    SUM:        a(i) = b(i) + c(i)
    TRIAD:      a(i) = b(i) + q*c(i)

  It measures the memory system on the device.
  The implementation is in single precision.

  Code based on the code developed by John D. McCalpin
  http://www.cs.virginia.edu/stream/FTP/Code/stream.c

  Written by: Massimiliano Fatica, NVIDIA Corporation

  Further modifications by: Ben Cumming, CSCS

  Ported to HIP by: Peng Sun, AMD
*/

#include "hip/hip_runtime.h"
#define NTIMES  20

#include <string>
#include <vector>

#include <stdio.h>
#include <float.h>
#include <limits.h>
#include <unistd.h>
#include <sys/time.h>

#include <sys/time.h>

# ifndef MIN
# define MIN(x,y) ((x)<(y)?(x):(y))
# endif
# ifndef MAX
# define MAX(x,y) ((x)>(y)?(x):(y))
# endif

#include <iostream>

#define FAIL(a) do { std::cerr << "FAIL: " << a << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl; abort(); } while (0)
#define CHECK(a) do { if (!(a)) FAIL("check " #a); } while (0)

#define CHECK_HIP(x)                                     \
do{                                                      \
    hipError_t err = x;                                  \
    if(hipSuccess != err)                                \
         FAIL("HIP error: " << hipGetErrorString(err));  \
}while(0)

typedef double real;

static double   avgtime[4] = {0}, maxtime[4] = {0},
        mintime[4] = {FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX};

#ifdef DEVICE_ARRAY
#define STATIC_ARRAY "__device__"
#define array_qualifier __device__
#endif

#ifdef CONSTANT_ARRAY
#define STATIC_ARRAY "__constant__"
#define array_qualifier __constant__
#endif

#ifdef MANAGED_ARRAY
#define STATIC_ARRAY "__managed__"
#define array_qualifier __managed__
#endif

#ifdef STATIC_ARRAY
const size_t STATIC_SIZE = 1<<26;

array_qualifier real d_a[STATIC_SIZE];
array_qualifier real d_b[STATIC_SIZE];
array_qualifier real d_c[STATIC_SIZE];
#endif

#ifndef STATIC_ARRAY
void wrap_mpi_init();
void wrap_mpi_alloc(size_t n, void **p);
#endif

void print_help()
{
    printf(
        "Usage: stream [-s] [-n <elements>] [-b <blocksize>]\n\n"
        "  -s\n"
        "        Print results in SI units (by default IEC units are used)\n\n"
        "  -n <elements>\n"
        "        Put <elements> values in the arrays\n"
        "        (defaults to 1<<26)\n\n"
        "  -b <blocksize>\n"
        "        Use <blocksize> as the number of threads in each block\n"
        "        (defaults to 192)\n"
    );
}

void parse_options(int argc, char** argv, bool& SI, int& N, int& blockSize, bool& cpuinit, bool& validate)
{
    // Default values
    SI = false;
    N = 1<<26;
    blockSize = 192;
    cpuinit = false;

    int c;

    while ((c = getopt (argc, argv, "sn:b:hcv")) != -1)
        switch (c)
        {
            case 's':
                SI = true;
                break;
            case 'n':
                N = std::atoi(optarg);
                break;
            case 'b':
                blockSize = std::atoi(optarg);
                break;
            case 'c':
                cpuinit = true;
                break;
            case 'v':
                validate = true;
                break;
            case 'h':
                print_help();
                std::exit(0);
                break;
            default:
                print_help();
                std::exit(1);
        }
}

/* A gettimeofday routine to give access to the wall
   clock timer on most UNIX-like systems.  */


double mysecond()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp,&tzp);
    return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}


#ifdef STATIC_ARRAY
__global__ void set_array_a(real value, int len)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        d_a[idx] = value;
    }
}
__global__ void set_array_b(real value, int len)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        d_b[idx] = value;
    }
}
__global__ void set_array_c(real value, int len)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        d_c[idx] = value;
    }
}

__global__ void STREAM_Copy(int len)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len)
        d_c[idx] = d_a[idx];
}

__global__ void STREAM_Scale(real scale, int len)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len)
        d_b[idx] = scale* d_c[idx];
}

__global__ void STREAM_Add(int len)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len)
        d_c[idx] = d_a[idx]+d_b[idx];
}

__global__ void STREAM_Triad(real scalar, int len)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len)
        d_a[idx] = d_b[idx]+scalar*d_c[idx];
}

#else

template <typename T>
__global__ void set_array(T *a,  T value, int len)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len)
        a[idx] = value;
}

template <typename T>
__global__ void STREAM_Copy(T *a, T *c, int len)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len)
        c[idx] = a[idx];
}

template <typename T>
__global__ void STREAM_Scale(T *b, T *c, T scale,  int len)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len)
        b[idx] = scale* c[idx];
}

template <typename T>
__global__ void STREAM_Add(T *a, T *b, T *c,  int len)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len)
        c[idx] = a[idx]+b[idx];
}

template <typename T>
__global__ void STREAM_Triad(T *a, T *b, T *c, T scalar, int len)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len)
        a[idx] = b[idx]+scalar*c[idx];
}
#endif

#ifndef abs
#define abs(a) ((a) >= 0 ? (a) : -(a))
#endif

#ifdef MANAGED_ARRAY
void checkSTREAMresults (size_t N)
#else
void checkSTREAMresults (size_t N, real *d_a, real *d_b, real *d_c)
#endif
{
	real aj,bj,cj,scalar;
	real aSumErr,bSumErr,cSumErr;
	real aAvgErr,bAvgErr,cAvgErr;
	double epsilon;
	ssize_t	j;
	int	k,ierr,err;

    /* reproduce initialization */
	aj = 1.0;
	bj = 2.0;
	cj = 0.0;
	//aj = 2.0;
	//bj = 0.5;
	//cj = 0.5;
    /* a[] is modified during timing check */
	//aj = 2.0E0 * aj;
    /* now execute timing loop */
	scalar = 3.0;
	for (k=0; k<NTIMES; k++)
        {
            cj = aj;
            bj = scalar*cj;
            cj = aj+bj;
            aj = bj+scalar*cj;
        }

    /* accumulate deltas between observed and expected results */
	aSumErr = 0.0;
	bSumErr = 0.0;
	cSumErr = 0.0;
	for (j=0; j<N; j++) {
		aSumErr += abs(d_a[j] - aj);
		bSumErr += abs(d_b[j] - bj);
		cSumErr += abs(d_c[j] - cj);
		// if (j == 417) printf("Index 417: c[j]: %f, cj: %f\n",c[j],cj);	// MCCALPIN
	}
	aAvgErr = aSumErr / (real) N;
	bAvgErr = bSumErr / (real) N;
	cAvgErr = cSumErr / (real) N;

	if (sizeof(real) == 4) {
		epsilon = 1.e-6;
	}
	else if (sizeof(real) == 8) {
		epsilon = 1.e-13;
	}
	else {
		printf("WEIRD: sizeof(real) = %lu\n",sizeof(real));
		epsilon = 1.e-6;
	}

	err = 0;
	if (abs(aAvgErr/aj) > epsilon) {
		err++;
		printf ("Failed Validation on array a[], AvgRelAbsErr > epsilon (%e)\n",epsilon);
		printf ("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n",aj,aAvgErr,abs(aAvgErr)/aj);
		ierr = 0;
		for (j=0; j<N; j++) {
			if (abs(d_a[j]/aj-1.0) > epsilon) {
				ierr++;
#ifdef VERBOSE
				if (ierr < 10) {
					printf("         array a: index: %ld, expected: %e, observed: %e, relative error: %e\n",
						j,aj,d_a[j],abs((aj-d_a[j])/aAvgErr));
				}
#endif
			}
		}
		printf("     For array a[], %d errors were found.\n",ierr);
	}
	if (abs(bAvgErr/bj) > epsilon) {
		err++;
		printf ("Failed Validation on array b[], AvgRelAbsErr > epsilon (%e)\n",epsilon);
		printf ("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n",bj,bAvgErr,abs(bAvgErr)/bj);
		printf ("     AvgRelAbsErr > Epsilon (%e)\n",epsilon);
		ierr = 0;
		for (j=0; j<N; j++) {
			if (abs(d_b[j]/bj-1.0) > epsilon) {
				ierr++;
#ifdef VERBOSE
				if (ierr < 10) {
					printf("         array b: index: %ld, expected: %e, observed: %e, relative error: %e\n",
						j,bj,d_b[j],abs((bj-d_b[j])/bAvgErr));
				}
#endif
			}
		}
		printf("     For array b[], %d errors were found.\n",ierr);
	}
	if (abs(cAvgErr/cj) > epsilon) {
		err++;
		printf ("Failed Validation on array c[], AvgRelAbsErr > epsilon (%e)\n",epsilon);
		printf ("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n",cj,cAvgErr,abs(cAvgErr)/cj);
		printf ("     AvgRelAbsErr > Epsilon (%e)\n",epsilon);
		ierr = 0;
		for (j=0; j<N; j++) {
			if (abs(d_c[j]/cj-1.0) > epsilon) {
				ierr++;
#ifdef VERBOSE
				if (ierr < 10) {
					printf("         array c: index: %ld, expected: %e, observed: %e, relative error: %e\n",
						j,cj,d_c[j],abs((cj-d_c[j])/cAvgErr));
				}
#endif
			}
		}
		printf("     For array c[], %d errors were found.\n",ierr);
	}
	if (err == 0) {
		printf ("Solution Validates: avg error less than %e on all three arrays\n",epsilon);
	}
#ifdef VERBOSE
	printf ("Results Validation Verbose Results: \n");
	printf ("    Expected a(1), b(1), c(1): %f %f %f \n",aj,bj,cj);
	printf ("    Observed a(1), b(1), c(1): %f %f %f \n",d_a[1],d_b[1],d_c[1]);
	printf ("    Rel Errors on a, b, c:     %e %e %e \n",abs(aAvgErr/aj),abs(bAvgErr/bj),abs(cAvgErr/cj));
#endif
}

int main(int argc, char** argv)
{
#ifndef STATIC_ARRAY
    real *d_a, *d_b, *d_c;
#endif
    int j,k;
    double times[4][NTIMES];
    real scalar;
    std::vector<std::string> label{"Copy:      ", "Scale:     ", "Add:       ", "Triad:     "};

    // Parse arguments
    bool SI;
    int N, blockSize;
    bool cpuinit, validate;
    parse_options(argc, argv, SI, N, blockSize, cpuinit, validate);

#ifdef STATIC_ARRAY
    CHECK(N == STATIC_SIZE);

    printf("STATIC_ARRAY: %s\n", STATIC_ARRAY);
#else
    if (optind+1 != argc) {
        printf("select alloc method\n");
        return 1;
    }

    if (!strcmp(argv[optind], "malloc")) {
        d_a = (real *)malloc(sizeof(real)*N);
        d_b = (real *)malloc(sizeof(real)*N);
        d_c = (real *)malloc(sizeof(real)*N);
    } else if (!strcmp(argv[optind], "hipMalloc")) {
        CHECK_HIP(hipMalloc((void**)&d_a, sizeof(real)*N));
        CHECK_HIP(hipMalloc((void**)&d_b, sizeof(real)*N));
        CHECK_HIP(hipMalloc((void**)&d_c, sizeof(real)*N));
    } else if (!strcmp(argv[optind], "hipHostMalloc")) {
        CHECK_HIP(hipHostMalloc((void**)&d_a, sizeof(real)*N));
        CHECK_HIP(hipHostMalloc((void**)&d_b, sizeof(real)*N));
        CHECK_HIP(hipHostMalloc((void**)&d_c, sizeof(real)*N));
    } else if (!strcmp(argv[optind], "hipHostMallocNonCoherent")) {
        CHECK_HIP(hipHostMalloc((void**)&d_a, sizeof(real)*N, hipHostMallocNonCoherent));
        CHECK_HIP(hipHostMalloc((void**)&d_b, sizeof(real)*N, hipHostMallocNonCoherent));
        CHECK_HIP(hipHostMalloc((void**)&d_c, sizeof(real)*N, hipHostMallocNonCoherent));
    } else if (!strcmp(argv[optind], "hipMallocManaged")) {
        CHECK_HIP(hipMallocManaged((void**)&d_a, sizeof(real)*N));
        CHECK_HIP(hipMallocManaged((void**)&d_b, sizeof(real)*N));
        CHECK_HIP(hipMallocManaged((void**)&d_c, sizeof(real)*N));
    } else if (!strcmp(argv[optind], "hipMallocManagedCpu")) {
        CHECK_HIP(hipMallocManaged((void**)&d_a, sizeof(real)*N));
        CHECK_HIP(hipMallocManaged((void**)&d_b, sizeof(real)*N));
        CHECK_HIP(hipMallocManaged((void**)&d_c, sizeof(real)*N));
        for (int i = 0; i < N; i++) {
            d_a[i] = 2.f;
            d_b[i] = .5f;
            d_c[i] = .5f;
        }
    } else if (!strcmp(argv[optind], "malloc+register")) {
        d_a = (real *)malloc(sizeof(real)*N);
        d_b = (real *)malloc(sizeof(real)*N);
        d_c = (real *)malloc(sizeof(real)*N);
        CHECK_HIP(hipHostRegister(d_a, sizeof(real)*N, hipHostRegisterDefault));
        CHECK_HIP(hipHostRegister(d_b, sizeof(real)*N, hipHostRegisterDefault));
        CHECK_HIP(hipHostRegister(d_c, sizeof(real)*N, hipHostRegisterDefault));
    } else if (!strcmp(argv[optind], "malloc+coarseGrain")) {
        d_a = (real *)malloc(sizeof(real)*N);
        d_b = (real *)malloc(sizeof(real)*N);
        d_c = (real *)malloc(sizeof(real)*N);
        CHECK_HIP(hipMemAdvise(d_a, sizeof(real)*N, hipMemAdviseSetCoarseGrain, 0));
        CHECK_HIP(hipMemAdvise(d_b, sizeof(real)*N, hipMemAdviseSetCoarseGrain, 0));
        CHECK_HIP(hipMemAdvise(d_c, sizeof(real)*N, hipMemAdviseSetCoarseGrain, 0));
    } else if (!strcmp(argv[optind], "MPI")) {
        wrap_mpi_init();
        wrap_mpi_alloc(sizeof(real)*N, (void**)&d_a);
        wrap_mpi_alloc(sizeof(real)*N, (void**)&d_b);
        wrap_mpi_alloc(sizeof(real)*N, (void**)&d_c);
    } else {
        printf("select alloc method\n");
        return 1;
    }
#endif

    printf(" STREAM Benchmark implementation in HIP\n");
    printf(" Array size (%s precision) =%7.2f MB\n", sizeof(double)==sizeof(real)?"double":"single", double(N)*double(sizeof(real))/1.e6);

    /* Compute execution configuration */
    dim3 dimBlock(blockSize);
    dim3 dimGrid(N/dimBlock.x );
    if( N % dimBlock.x != 0 ) dimGrid.x+=1;

    printf(" using %d threads per block, %d blocks\n",dimBlock.x,dimGrid.x);

    if (SI)
        printf(" output in SI units (KB = 1000 B)\n");
    else
        printf(" output in IEC units (KiB = 1024 B)\n");

    if (cpuinit) {
        printf(" init on cpu\n");

#if defined(STATIC_ARRAY) && !defined(MANAGED_ARRAY)
        void *a, *b, *c;
        CHECK_HIP(hipGetSymbolAddress(&a, HIP_SYMBOL(d_a)));
        CHECK_HIP(hipGetSymbolAddress(&b, HIP_SYMBOL(d_b)));
        CHECK_HIP(hipGetSymbolAddress(&c, HIP_SYMBOL(d_c)));

        memset(a, 0, sizeof(real)*N);
        memset(b, 0, sizeof(real)*N);
        memset(c, 0, sizeof(real)*N);
#else
        memset(d_a, 0, sizeof(real)*N);
        memset(d_b, 0, sizeof(real)*N);
        memset(d_c, 0, sizeof(real)*N);
#endif
    }

    /* Initialize memory on the device */
#ifdef STATIC_ARRAY
    set_array_a<<<dimGrid, dimBlock, 0, 0>>>(1.0, N);
    set_array_b<<<dimGrid, dimBlock, 0, 0>>>(2.0, N);
    set_array_c<<<dimGrid, dimBlock, 0, 0>>>(0.0, N);
#else
    set_array<real><<<dimGrid, dimBlock, 0, 0>>>(d_a, 1.0, N);
    set_array<real><<<dimGrid, dimBlock, 0, 0>>>(d_b, 2.0, N);
    set_array<real><<<dimGrid, dimBlock, 0, 0>>>(d_c, 0.0, N);
#endif

    /*  --- MAIN LOOP --- repeat test cases NTIMES times --- */

    scalar=3.0f;
    for (k=0; k<NTIMES; k++)
    {
        times[0][k]= mysecond();
#ifdef STATIC_ARRAY
        STREAM_Copy<<<dimGrid, dimBlock, 0, 0>>>(N);
#else
        STREAM_Copy<real><<<dimGrid, dimBlock, 0, 0>>>(d_a, d_c, N);
#endif
        CHECK_HIP(hipDeviceSynchronize());
        times[0][k]= mysecond() -  times[0][k];

        times[1][k]= mysecond();
#ifdef STATIC_ARRAY
        STREAM_Scale<<<dimGrid, dimBlock, 0, 0>>>(scalar, N);
#else
        STREAM_Scale<real><<<dimGrid, dimBlock, 0, 0>>>(d_b, d_c, scalar, N);
#endif
        CHECK_HIP(hipDeviceSynchronize());
        times[1][k]= mysecond() -  times[1][k];

        times[2][k]= mysecond();
#ifdef STATIC_ARRAY
        STREAM_Add<<<dimGrid, dimBlock, 0, 0>>>(N);
#else
        STREAM_Add<real><<<dimGrid, dimBlock, 0, 0>>>(d_a, d_b, d_c, N);
#endif
        CHECK_HIP(hipDeviceSynchronize());
        times[2][k]= mysecond() -  times[2][k];

        times[3][k]= mysecond();
#ifdef STATIC_ARRAY
        STREAM_Triad<<<dimGrid, dimBlock, 0, 0>>>(scalar, N);
#else
        STREAM_Triad<real><<<dimGrid, dimBlock, 0, 0>>>(d_a, d_b, d_c, scalar, N);
#endif
        CHECK_HIP(hipDeviceSynchronize());
        times[3][k]= mysecond() -  times[3][k];
    }

    /*  --- SUMMARY --- */

    for (k=1; k<NTIMES; k++) /* note -- skip first iteration */
    {
        for (j=0; j<4; j++)
        {
            avgtime[j] = avgtime[j] + times[j][k];
            mintime[j] = MIN(mintime[j], times[j][k]);
            maxtime[j] = MAX(maxtime[j], times[j][k]);
        }
    }

    double bytes[4] = {
        2 * sizeof(real) * (double)N,
        2 * sizeof(real) * (double)N,
        3 * sizeof(real) * (double)N,
        3 * sizeof(real) * (double)N
    };

    // Use right units
    const double G = SI ? 1.e9 : static_cast<double>(1<<30);

    printf("\nFunction      Rate %s  Avg time(s)  Min time(s)  Max time(s)\n",
           SI ? "(GB/s) " : "(GiB/s)" );
    printf("-----------------------------------------------------------------\n");
    for (j=0; j<4; j++) {
        avgtime[j] = avgtime[j]/(double)(NTIMES-1);

        printf("%s%11.4f     %11.8f  %11.8f  %11.8f\n", label[j].c_str(),
                bytes[j]/mintime[j] / G,
                avgtime[j],
                mintime[j],
                maxtime[j]);
    }

    if (validate) {
#ifdef STATIC_ARRAY
#ifdef MANAGED_ARRAY
        checkSTREAMresults(N);
#else
        real *a = (real *)malloc(sizeof(real)*N);
        real *b = (real *)malloc(sizeof(real)*N);
        real *c = (real *)malloc(sizeof(real)*N);
        CHECK_HIP(hipMemcpyToSymbol(HIP_SYMBOL(d_a), a, sizeof(real)*N, hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpyToSymbol(HIP_SYMBOL(d_b), b, sizeof(real)*N, hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpyToSymbol(HIP_SYMBOL(d_c), c, sizeof(real)*N, hipMemcpyHostToDevice));
        /* CHECK_HIP(hipMemcpyFromSymbol(a, HIP_SYMBOL(d_a), sizeof(real)*N)); */
        /* CHECK_HIP(hipMemcpyFromSymbol(b, HIP_SYMBOL(d_b), sizeof(real)*N)); */
        /* CHECK_HIP(hipMemcpyFromSymbol(c, HIP_SYMBOL(d_c), sizeof(real)*N)); */
        checkSTREAMresults(N, a, b, c);
#endif
#else
        checkSTREAMresults(N, d_a, d_b, d_c);
#endif
    }

    /* Free memory on device */
    //hipFree(d_a);
    //hipFree(d_b);
    //hipFree(d_c);
}

