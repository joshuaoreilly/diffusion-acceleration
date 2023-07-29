#include <iostream>
#include <chrono>
#include <cmath>
#include <fstream>
#include <vector>
#include <omp.h>
#include "diffusioncuda.h"


/*
* TODO:
*   - Find out why std::abs() works but abs() doesn't in initialization
*
* Compile with:
* g++ -O3 -Wall -Wextra -pedantic -Wall -Wconversion -Wextra -pedantic -std=c++14 -fopenmp -o diffusion diffusion.cpp
*/


void save_results(const std::vector<double> &c,
                    const std::string &implementation) {
    std::string filename = "cpp_" + implementation + ".txt";
    // M = N + 2, the length with padding
    size_t M = (size_t) sqrt(c.size());
    std::ofstream ofs(filename, std::ofstream::out);
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < M; ++j) {
            ofs << c[i * M + j];
            if (j == M - 1) {
                ofs << "\n";
            } else {
                ofs << ",";
            }
        }
    }
    ofs.close();
}


void vector_to_arr(std::vector<double> &c,
                    double *arr,
                    const size_t M) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < M; ++j) {
            arr[i * M + j] = c[i * M + j];
        }
    }
}


void arr_to_vector(std::vector<double> &c,
                    double *arr,
                    const size_t M) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < M; ++j) {
            c[i * M + j] = arr[i * M + j];
        }
    }
}


std::chrono::nanoseconds diffuse_openmp(std::vector<double> &c,
                            std::vector<double> &c_tmp,
                            const double aux,
                            const size_t num_steps) {
    // M = N + 2, the length with padding
    const size_t M = (size_t) sqrt(c.size());
    auto time_start = std::chrono::steady_clock::now();
    for (size_t step = 0; step < num_steps; ++step) {
#pragma omp parallel for collapse(2)
        for (size_t i = 1; i < M - 1; ++i) {
            for (size_t j = 1; j < M - 1; ++j) {
                c_tmp[i * M + j] = c[i * M + j] + aux * (
                    c[i * M + (j + 1)] + c[i * M + (j - 1)] +
                    c[(i + 1) * M + j] + c[(i - 1) * M + j] -
                    4 * c[i * M + j]);
            }
        }
        std::swap(c, c_tmp);
    }
    auto time_stop = std::chrono::steady_clock::now();
    auto time_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(time_stop - time_start);
    return time_elapsed;
}


__inline__ void step_const_c(const std::vector<double> &__restrict c,
                    std::vector<double> &c_tmp,
                    const size_t &M,
                    const double &aux) {
    for (size_t i = 1; i < M - 1; ++i) {
        for (size_t j = 1; j < M - 1; ++j) {
            c_tmp[i * M + j] = c[i * M + j] + aux * (
                c[i * M + (j + 1)] + c[i * M + (j - 1)] +
                c[(i + 1) * M + j] + c[(i - 1) * M + j] -
                4 * c[i * M + j]);
        }
    }
}


std::chrono::nanoseconds diffuse_const_c(std::vector<double> &c,
                            std::vector<double> &c_tmp,
                            const double aux,
                            const size_t num_steps) {
    // M = N + 2, the length with padding
    const size_t M = (size_t) sqrt(c.size());
    auto time_start = std::chrono::high_resolution_clock::now();
    for (size_t step = 0; step < num_steps; ++step) {
        step_const_c(c, c_tmp, M, aux);
        std::swap(c, c_tmp);
    }
    auto time_stop = std::chrono::high_resolution_clock::now();
    auto time_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(time_stop - time_start);
    return time_elapsed;
}


std::chrono::nanoseconds diffuse_naive(std::vector<double> &c,
                            std::vector<double> &c_tmp,
                            const double aux,
                            const size_t num_steps) {
    // M = N + 2, the length with padding
    const size_t M = (size_t) sqrt(c.size());
    auto time_start = std::chrono::steady_clock::now();
    for (size_t step = 0; step < num_steps; ++step) {
        for (size_t i = 1; i < M - 1; ++i) {
            for (size_t j = 1; j < M - 1; ++j) {
                c_tmp[i * M + j] = c[i * M + j] + aux * (
                    c[i * M + (j + 1)] + c[i * M + (j - 1)] +
                    c[(i + 1) * M + j] + c[(i - 1) * M + j] -
                    4 * c[i * M + j]);
            }
        }
        std::swap(c, c_tmp);
    }
    auto time_stop = std::chrono::steady_clock::now();
    auto time_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(time_stop - time_start);
    return time_elapsed;
}


void initialize_concentration(std::vector<double> &c,
                                const double &L,
                                const size_t &N,
                                const double &h) {
    const double bound = 0.125 * L;
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            if (std::abs(i * h - 0.5 * L) < bound &&
                std::abs(j * h - 0.5 * L) < bound) {
                c[(i + 1) * (N + 2) + (j + 1)] = 1.0;
            }
        }
    }
}


int main(int argc, char *argv[]) {
    if (argc < 7) {
        fprintf(stderr, "Usage: %s D L N T out implementation\n", argv[0]);
        return 1;
    }

    const double D = std::stod(argv[1]);        // diffusion constant
    const double L = std::stod(argv[2]);        // domain size
    const size_t N = std::stoul(argv[3]);       // number of grid points along each dim
    const double T = std::stod(argv[4]);        // simulation time in seconds
    const size_t output = std::stoul(argv[5]);  // whether to store results, yes == 1
    const std::string implementation = argv[6]; // version of program to run

    const double h = L / (N - 1);               // length between grid elements
    const double dt = h * h / (4 * D);          // maximum timestamp for numeric stability
    const double aux = dt * D / (h * h);        // all constant terms in diffusion equation
    const size_t num_steps = (size_t) ((T / dt) + 1);

    std::vector<double> c((N+2) * (N+2), 0.0);
    std::vector<double> c_tmp((N+2) * (N+2), 0.0);
    initialize_concentration(c, L, N, h);
    std::chrono::nanoseconds time_elapsed;
    if (implementation.compare("naive") == 0) {
        time_elapsed = diffuse_naive(c, c_tmp, aux, num_steps);
    } else if (implementation.compare("const") == 0) {
        time_elapsed = diffuse_const_c(c, c_tmp, aux, num_steps);
    } else if (implementation.compare("openmp") == 0) {
        time_elapsed = diffuse_openmp(c, c_tmp, aux, num_steps);
    } else if (implementation.compare("cuda") == 0) {
        double *arr = (double *) malloc((N + 2) * (N + 2) * sizeof(double));
        vector_to_arr(c, arr, N + 2);
        time_elapsed = diffuse_cuda(arr, aux, N + 2, num_steps);
        arr_to_vector(c, arr, N + 2);
    }  else {
        std::cout << "Implementation " << implementation << " not found\n";
        exit(1);
    }
    printf("Elapsed time: %luns\n", time_elapsed.count());
    if (output == 1) {
        save_results(c, implementation);
    }
}
