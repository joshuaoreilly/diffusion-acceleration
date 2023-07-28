#include<chrono>
#include <stdio.h>
#include <assert.h>
#include "diffusioncuda.h"


inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}


void swap(double* &a, double* &b){
    double *temp = a;
    a = b;
    b = temp;
}


__global__
void step_kernel(double *c, double *c_tmp, const double aux, const size_t M) {
    int index = threadIdx.x + blockIdx.x * blockDim.x + M + 1;
    int stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < (M * M) - M; i += stride) {
        if (i % M != 0 && i % M != M - 1) {
            c_tmp[i] = c[i] + aux * (
                    c[i + 1] + c[i - 1] +
                    c[i + M] + c[i - M] -
                    4 * c[i]);
        }
    }
}


std::chrono::nanoseconds diffuse_cuda(double *c_h,
                                    const double T,
                                    const double dt,
                                    const double aux,
                                    const size_t M) {
    int deviceId;
    cudaDeviceProp props;
    cudaGetDevice(&deviceId);
    cudaGetDeviceProperties(&props, deviceId);
    const int multiProcessorCount = props.multiProcessorCount;

    const size_t bytes = M * M * sizeof(double);

    double *c;
    double *c_tmp;

    checkCuda(cudaMalloc(&c, bytes));
    checkCuda(cudaMalloc(&c_tmp, bytes));

    auto time_cpy_htod_start = std::chrono::steady_clock::now();
    cudaMemcpy(c, c_h, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(c_tmp, c_h, bytes, cudaMemcpyHostToDevice);
    auto time_cpy_htod_stop = std::chrono::steady_clock::now();
    auto time_cpy_htod_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(time_cpy_htod_stop - time_cpy_htod_start);

    int threadsPerBlock = 32;
    int numberOfBlocks = 32 * multiProcessorCount;

    const size_t num_steps = (size_t) ((T / dt) + 1);
    auto time_start = std::chrono::steady_clock::now();
    for (size_t step = 0; step < num_steps; ++step) {
        step_kernel<<<numberOfBlocks, threadsPerBlock>>>(c, c_tmp, aux, M);
        checkCuda(cudaGetLastError());
        checkCuda(cudaDeviceSynchronize());
        swap(c, c_tmp);
    }
    auto time_stop = std::chrono::steady_clock::now();
    auto time_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(time_stop - time_start);

    auto time_cpy_dtoh_start = std::chrono::steady_clock::now();
    cudaMemcpy(c_h, c, bytes, cudaMemcpyDeviceToHost);
    auto time_cpy_dtoh_stop = std::chrono::steady_clock::now();
    auto time_cpy_dtoh_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(time_cpy_dtoh_stop - time_cpy_dtoh_start);

    printf("IO time: %ldns\n", time_cpy_htod_elapsed.count() + time_cpy_dtoh_elapsed.count());

    cudaFree(c);
    cudaFree(c_tmp);
    return time_elapsed;
}
