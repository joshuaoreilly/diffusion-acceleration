#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "diffusioncuda.h"


void diffuse_cuda(double *c,
                    double *c_tmp,
                    const double T,
                    const double dt,
                    const double aux) {
    printf("T: %f, dt: %f, aux: %f\n", T, dt, aux);
    int deviceId;
    cudaDeviceProp props;
    cudaGetDevice(&deviceId);
    cudaGetDeviceProperties(&props, deviceId);
    int multiProcessorCount = props.multiProcessorCount;
    printf("Number of streaming multiprocessors: %d\n", multiProcessorCount);
}
