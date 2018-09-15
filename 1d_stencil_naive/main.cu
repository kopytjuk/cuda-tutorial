#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#define R 3

__global__ void oneD_stencil_naive(int *in_arr, int *out_arr) {
    int in_index = blockIdx.x + threadIdx.x;
    int out_index = blockIdx.x;
    // guaranteed to be performed without interference from other threads 
    atomicAdd(out_arr+out_index, in_arr[in_index]);
    __syncthreads();
}

int main(void) {

    int device;
    cudaGetDevice(&device);

    struct cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    printf("Using %s.\n\n", props.name);

    // host copies of input and output array
    int arr_in[13] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
    int in_elements = 13;
    int in_size = in_elements*sizeof(int);

    int *arr_out;
    int out_elements = in_elements-(2*R+1)+1;
    int out_size = out_elements*sizeof(int);
    arr_out = (int *)malloc(out_size);

    int *d_arr_in, *d_arr_out; // host copies of input and output array

    // Allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_arr_in, in_size);
    cudaMalloc((void **)&d_arr_out, out_size);

    // Copy inputs to device
    cudaMemcpy(d_arr_in, arr_in, in_size, cudaMemcpyHostToDevice);

    // Launch stencil() kernel on GPU
    int n_blocks = out_elements;
    int n_threads_per_block = 2*R+1;
    oneD_stencil_naive<<<n_blocks, n_threads_per_block>>>(d_arr_in, d_arr_out);
    
    // Copy result back to host
    cudaMemcpy(arr_out, d_arr_out, out_size, cudaMemcpyDeviceToHost);
    std::cout << "[";
    for(int i=0; i<n_blocks; ++i){
        std::cout << arr_out[i] << ", ";
    }
    std::cout << "]" << std::endl;

    // Cleanup
    cudaFree(d_arr_in); cudaFree(d_arr_out);
    return 0;
}