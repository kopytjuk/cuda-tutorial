#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#define R 3
#define BLOCK_SIZE 5 // number of output elements calculated in one block

int iDivUp(int a, int b){ return ((a % b) != 0) ? (a / b + 1) : (a / b); }

__global__ void oneD_stencil_shared(int *in_arr, int *out_arr, int n_input) {
    
    // we declare a shared memory within a block, s.t. input elements don't have to be loaded from global memory several times
    // reading from shared memory is faster!
    __shared__ int temp_buffer[BLOCK_SIZE + 2*R];
    int gindex = R + blockDim.x*blockIdx.x + threadIdx.x;
    int lindex = R + threadIdx.x;

    if(gindex < n_input){
        temp_buffer[lindex] = in_arr[gindex];

        if(threadIdx.x < R){
            temp_buffer[lindex-R] = in_arr[gindex-R];
            temp_buffer[lindex + BLOCK_SIZE] = in_arr[gindex + BLOCK_SIZE];
        }
    }

    // until this point we want the shared buffer to be filled, we wait
    __syncthreads();

    int res = 0;

    for(int i=-R; i<=R; ++i){
        res += temp_buffer[lindex + i];
    }

    int oindex = gindex - R;
    out_arr[oindex] = res;
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
    int n_blocks = iDivUp(in_elements, BLOCK_SIZE);
    int n_threads_per_block = BLOCK_SIZE;
    oneD_stencil_shared<<<n_blocks, n_threads_per_block>>>(d_arr_in, d_arr_out, in_elements);
    
    // Copy result back to host
    cudaMemcpy(arr_out, d_arr_out, out_size, cudaMemcpyDeviceToHost);
    std::cout << "[";
    for(int i=0; i<out_elements; ++i){
        std::cout << arr_out[i] << ", ";
    }
    std::cout << "]" << std::endl;

    // Cleanup
    cudaFree(d_arr_in); cudaFree(d_arr_out);
    return 0;
}