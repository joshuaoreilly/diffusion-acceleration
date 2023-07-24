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
    for (size_t i = 1; i < M - 1; ++i) {
        for (size_t j = 1; j < M - 1; ++j) {
            c_tmp[i * M + j] = c[i * M + j] + aux * (
                c[i * M + (j + 1)] + c[i * M + (j - 1)] +
                c[(i + 1) * M + j] + c[(i - 1) * M + j] -
                4 * c[i * M + j]);
        }
    }
}


std::chrono::nanoseconds diffuse(double *c_h,
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

    cudaMemcpy(c, c_h, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(c_tmp, c_h, bytes, cudaMemcpyHostToDevice);


    const size_t num_steps = (size_t) ((T / dt) + 1);
    auto time_start = std::chrono::steady_clock::now();
    for (size_t step = 0; step < num_steps; ++step) {
        step_kernel<<<1, 1>>>(c, c_tmp, aux, M);
        checkCuda(cudaGetLastError());
        checkCuda(cudaDeviceSynchronize());
        swap(c, c_tmp);
    }
    auto time_stop = std::chrono::steady_clock::now();
    auto time_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(time_stop - time_start);

    cudaMemcpy(c_h, c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(c);
    cudaFree(c_tmp);
    return time_elapsed;
}


std::chrono::nanoseconds diffuse_cuda(double *c_h,
                    const double T,
                    const double dt,
                    const double aux,
                    const size_t M) {
    return diffuse(c_h, T, dt, aux, M);
}
