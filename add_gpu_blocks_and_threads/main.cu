#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <limits.h>

#define OPERATOR *
#define OPERATOR_NAME "multiplication"

#define DTYPE float

void random_ints(int* a, int N)
{
   int i;
   for (i = 0; i < N; ++i)
    a[i] = rand();
}

void random_floats(float* a, int N)
{
    for (int i = 0; i < N; ++i)
        a[i] = (float)rand()/(float)(RAND_MAX/a[i]);
}

__global__ void add_gpu_blocks(DTYPE *a, DTYPE *b, DTYPE *c) {
    c[blockIdx.x] = a[blockIdx.x] OPERATOR b[blockIdx.x];
}

__global__ void add_gpu_threads(DTYPE *a, DTYPE *b, DTYPE *c) {
    c[threadIdx.x] = a[threadIdx.x] OPERATOR b[threadIdx.x];
}

__global__ void add_gpu_both(DTYPE *a, DTYPE *b, DTYPE *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n)
        c[index] = a[index] OPERATOR b[index];
}

void add_cpu(DTYPE *a, DTYPE *b, DTYPE *c, int size) {
    for (int i=0; i<size; ++i)
        c[i] = a[i] OPERATOR b[i];
}

#define N (int)2048*2048*120
#define THREADS_PER_BLOCK 512

int main(void) {

    int device;
    cudaGetDevice(&device);

    struct cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    printf("Using %s.\n\n", props.name);

    DTYPE *a, *b, *c; // host copies of a, b, c
    DTYPE *d_a, *d_b, *d_c; // device copies of a, b, c
    int size = N*sizeof(DTYPE);

    // Allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Alloc space for host copies of a, b, c and setup input values
    a = (DTYPE *)malloc(size); random_floats(a, N);
    b = (DTYPE *)malloc(size); random_floats(b, N);
    c = (DTYPE *)malloc(size);

    printf("Calculating vector %s with %d elements...\n", OPERATOR_NAME, N);

    clock_t t_cpu;
    t_cpu = clock();
    add_cpu(a, b, c, N);
    t_cpu = clock() - t_cpu;
    double time_taken_cpu = 1000*((double)t_cpu)/CLOCKS_PER_SEC; // in micro-second
    printf("CPU: %.2fms\n", time_taken_cpu);

    // Calculate the time taken by calculation
    clock_t t_start;
    t_start = clock();

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    clock_t t_mem_in = clock();

    // Launch add() kernel on GPU
    add_gpu_both<<<N/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    clock_t t_calc= clock();
    
    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    clock_t t_mem_out = clock();

    clock_t t_gpu = t_mem_out-t_start;
    clock_t t_gpu_mem_in = t_mem_in-t_start;
    clock_t t_gpu_calc = t_calc-t_mem_in;
    clock_t t_gpu_mem_out = t_mem_out-t_calc;

    double time_taken_gpu = 1000*((double)t_gpu)/CLOCKS_PER_SEC; // in micro-second
    double time_taken_mem_in = 1000*((double)t_gpu_mem_in)/CLOCKS_PER_SEC; // in micro-second
    double time_taken_calc = 1000*((double)t_gpu_calc)/CLOCKS_PER_SEC; // in micro-second
    double time_taken_mem_out = 1000*((double)t_gpu_mem_out)/CLOCKS_PER_SEC; // in micro-second

    printf("GPU: %.2fms.\n", time_taken_gpu);
    printf("\tmemory in: %.2fms.\n", time_taken_mem_in);
    printf("\tkernel: %.2fms.\n", time_taken_calc);
    printf("\tmemory out: %.2fms.\n", time_taken_mem_out);

    // Cleanup
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}