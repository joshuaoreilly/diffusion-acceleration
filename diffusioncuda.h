#ifndef DIFFUSIONCUDA_H
#define DIFFUSIONCUDA_H

std::chrono::nanoseconds diffuse_cuda(double *c,
                                        const double T,
                                        const double dt,
                                        const double aux,
                                        const size_t M);

#endif
