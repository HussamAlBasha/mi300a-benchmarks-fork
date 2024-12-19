#include <iostream>
#include <hip/hip_runtime.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <errno.h>

const size_t SIZE = 10*1024*1024;

const char *FILENAME = "getmem_file";
int file_fd = -1;

bool prefix(const char *str, const char *pre)
{
    return strncmp(pre, str, strlen(pre)) == 0;
}

void measure()
{
    auto& out = std::cout;

    size_t ps = getpagesize();
    size_t kb = 1024;

    out << "hipMemGetInfo\t";
    size_t free, total;
    if (hipSuccess != hipMemGetInfo(&free, &total))
        abort();
    out << free << std::endl;

    FILE *fp = fopen("/proc/meminfo", "r");
    if (!fp)
        abort();
    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        const char *name = NULL;
        if (prefix(line, "MemFree:"))
            name = "MemFree";
        else if (prefix(line, "MemAvailable:"))
            name = "MemAvailable";

        if (name) {
            char *p = line;
            while (!isspace(*p)) p++;
            while (isspace(*p)) p++;
            char *end = NULL;
            size_t n = strtoul(p, &end, 10);
            if (end == NULL)
                abort();

            out << name << "\t" << n*kb << std::endl;
        }
    }
    fclose(fp);

    fp = fopen("/proc/self/statm", "r");
    if (!fp)
        abort();
    size_t vm_size = 0;
    size_t vm_rss = 0;
    if (2 != fscanf(fp, "%zu %zu", &vm_size, &vm_rss))
        abort();
    out << "VmSize\t" << vm_size*ps << std::endl;
    out << "VmRss\t" << vm_rss*ps << std::endl;
    fclose(fp);

    fp = fopen("/proc/self/smaps_rollup", "r");
    if (!fp)
        abort();
    while (fgets(line, sizeof(line), fp)) {
        const char *name = NULL;
        if (prefix(line, "Rss:"))
            name = "Rss";

        if (name) {
            char *p = line;
            while (!isspace(*p)) p++;
            while (isspace(*p)) p++;
            char *end = NULL;
            size_t n = strtoul(p, &end, 10);
            if (end == NULL)
                abort();

            out << name << "\t" << n*kb << std::endl;
        }
    }
    fclose(fp);
}

void *alloc_malloc()
{
    return malloc(SIZE);
}

void *alloc_mmap_anon()
{
    void *p = mmap(NULL, SIZE, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED)
        abort();
    return p;
}

void *alloc_mmap_file()
{
    //void *p = mmap(NULL, SIZE, PROT_READ, MAP_SHARED, file_fd, 0);
    void *p = mmap(NULL, SIZE, PROT_READ, MAP_PRIVATE, file_fd, 0);
    if (p == MAP_FAILED) {
        perror("mmap");
        abort();
    }
    return p;
}

void *alloc_hipMalloc()
{
    void *p;
    if (hipSuccess != hipMalloc(&p, SIZE))
        abort();
    return p;
}

void *alloc_hipHostMalloc()
{
    void *p;
    if (hipSuccess != hipHostMalloc(&p, SIZE))
        abort();
    return p;
}

void *alloc_hipMallocManaged()
{
    void *p;
    if (hipSuccess != hipMallocManaged(&p, SIZE))
        abort();
    return p;
}

void cpu_read(void *p)
{
    char *q = (char *)p;
    size_t s = 0;
    for (size_t i = 0; i < SIZE; i++)
        s += q[i];
}

void cpu_write(void *p)
{
    memset(p, 123, SIZE);
}

void gpu_read(void *p)
{
    /* TBD */
    abort();
}

void gpu_write(void *p)
{
    if (hipSuccess != hipMemset(p, 123, SIZE))
        abort();
}

struct allocator {
    const char *name;
    void *(*alloc)();
    bool touch_write;
};

struct allocator allocs[] = {
    {"malloc", &alloc_malloc, true},
    {"mmap_anon", &alloc_mmap_anon, true},
    {"mmap_file", &alloc_mmap_file, false},
    {"hipMalloc", &alloc_hipMalloc, true},
    {"hipHostMalloc", &alloc_hipHostMalloc, true},
    {"hipMallocManaged", &alloc_hipMallocManaged, true},
    {},
};

int main(int argc, char **argv)
{
    allocator *alloc = NULL;
    bool touch_cpu = false;
    bool touch_gpu = false;

    // Ensure everything is initialized before we start experimenting
    if (hipSuccess != hipInit(0))
        abort();
    void *q = alloc_hipMalloc();
    if (hipSuccess != hipMemset(q, 312, SIZE))
        abort();

    if (argc > 1) {
        for (allocator *a = allocs; a->name; a++)
            if (!strcmp(argv[1], a->name)) {
                alloc = a;
                break;
            }

        if (alloc == NULL)
            abort();

        if (!strcmp(alloc->name, "mmap_file")) {
            char *buf = (char *)malloc(SIZE);
            for (int i = 0; i < SIZE; i++)
                buf[i] = i;
            file_fd = open(FILENAME, O_RDWR|O_CREAT|O_TRUNC, 0644);
            if (!file_fd)
                abort();
            size_t n = SIZE;
            while (n) {
                size_t m = write(file_fd, buf+SIZE-n, n);
                if (m < 0)
                    abort();
                n -= m;
            }
            free(buf);
        }
    }
    if (argc > 2) {
        if (!strcmp(argv[2], "cpu"))
            touch_cpu = true;
        if (!strcmp(argv[2], "gpu"))
            touch_gpu = true;
    }

    measure();

    if (alloc) {
        std::cout << std::endl;
        void *p = alloc->alloc();
        measure();

        if (touch_cpu) {
            if (alloc->touch_write)
                cpu_write(p);
            else
                cpu_read(p);

            std::cout << std::endl;
            measure();
        }

        if (touch_gpu) {
            if (alloc->touch_write)
                gpu_write(p);
            else
                gpu_read(p);

            std::cout << std::endl;
            measure();
        }
    }

    if (file_fd != -1)
        unlink(FILENAME);
}
