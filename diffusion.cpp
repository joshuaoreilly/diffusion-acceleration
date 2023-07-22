#include <chrono>
#include <cmath>
#include <fstream>
#include <vector>


/*
* TODO:
*   - Find out why std::abs() works but abs() doesn't in initialization
*/


void save_results(const std::vector<double> &c,
                    const std::string &implementation) {
    std::string filename = "cpp_" + implementation + ".txt";
    size_t length = sqrt(c.size());
    std::ofstream ofs(filename, std::ofstream::out);
    for (size_t i = 0; i < length; ++i) {
        for (size_t j = 0; j < length; ++j) {
            ofs << c[i * length + j];
            if (j == length - 1) {
                ofs << "\n";
            } else {
                ofs << ",";
            }
        }
    }
    ofs.close();
}


std::chrono::nanoseconds diffuse_naive(std::vector<double> &c,
                            std::vector<double> &c_tmp,
                            const double T,
                            const double dt,
                            const double aux) {
    const size_t num_steps = (T / dt) + 1;
    const size_t length = sqrt(c.size());
    // M = N + 2, the length with padding
    const int M = (int) length;
    auto time_start = std::chrono::high_resolution_clock::now();
    for (size_t step = 0; step < num_steps; ++step) {
        for (size_t i = 1; i < length - 1; ++i) {
            for (size_t j = 1; j < length - 1; ++j) {
                c_tmp[i * M + j] = c[i * M + j] + aux * (
                    c[i * M + (j + 1)] + c[i * M + (j - 1)] +
                    c[(i + 1) * M + j] + c[(i - 1) * M + j] -
                    4 * c[i * M + j]);
            }
        }
        std::swap(c, c_tmp);
    }
    auto time_stop = std::chrono::high_resolution_clock::now();
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

    std::vector<double> c((N+2) * (N+2), 0.0);
    std::vector<double> c_tmp((N+2) * (N+2), 0.0);
    initialize_concentration(c, L, N, h);
    std::chrono::nanoseconds time_elapsed;
    if (implementation.compare("naive") == 0) {
        time_elapsed = diffuse_naive(c, c_tmp, T, dt, aux);
    }
    printf("Elapsed time: %luns\n", time_elapsed.count());
    if (output == 1) {
        save_results(c, implementation);
    }
}
