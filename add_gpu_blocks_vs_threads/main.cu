#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

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

void add_cpu(DTYPE *a, DTYPE *b, DTYPE *c, int size) {
    for (int i=0; i<size; ++i)
        c[i] = a[i] OPERATOR b[i];
}

#define N (int)10e7

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
    clock_t t_gpu;
    t_gpu = clock();

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch add() kernel on GPU
    add_gpu_blocks<<<N,1>>>(d_a, d_b, d_c);
    
    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    t_gpu = clock() - t_gpu;
    double time_taken_gpu = 1000*((double)t_gpu)/CLOCKS_PER_SEC; // in micro-second

    printf("GPU (blocks): %.2fms.\n", time_taken_gpu);

    // ------------------------------------------------

    // Calculate the time taken by calculation
    clock_t t_gpu_2;
    t_gpu_2 = clock();

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch add() kernel on GPU
    add_gpu_threads<<<1,N>>>(d_a, d_b, d_c);
    
    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    t_gpu_2 = clock() - t_gpu_2;
    double time_taken_gpu_2 = 1000*((double)t_gpu_2)/CLOCKS_PER_SEC; // in micro-second

    printf("GPU (threads): %.2fms.\n", time_taken_gpu_2);

    // Cleanup
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}