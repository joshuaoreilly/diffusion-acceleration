#ifndef DIFFUSIONCUDA_H
#define DIFFUSIONCUDA_H

void diffuse_cuda(double *c,
                    double *c_tmp,
                    const double T,
                    const double dt,
                    const double aux);

#endif
