#ifndef DIFFUSIONCUDA_H
#define DIFFUSIONCUDA_H

std::chrono::nanoseconds diffuse_cuda(double *c,
                                        const double aux,
                                        const size_t M,
                                        const size_t num_steps);

#endif
