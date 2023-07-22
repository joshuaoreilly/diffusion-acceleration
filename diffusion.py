import argparse
from pathlib import Path
from typing import List
from time import perf_counter_ns
import numpy as np
import matplotlib.pyplot as plt


"""
TODO:
- Profile to verify why numpy is slower (hypothesis: no benefit of underlying C++ data structure since doing single accesses at a time)
"""


def save_results(c, implementation):
     if implementation == 'naive':
          with open(Path('python_' + implementation + '.txt'), 'w') as file:
               for row in c:
                    file.writelines(','.join([str(i) for i in row]) + '\n')
     elif implementation == 'numpy':
          np.savetxt('python_' + implementation + '.txt', c, fmt='%f', delimiter=',')
     else:
          raise ValueError(f'Implementation {implementation} does\'t exist')


def diffuse_numpy(c : List[List[float]],
                  c_tmp : List[List[float]],
                  T : float,
                  dt : float,
                  aux : float):
     c = np.array(c)
     c_tmp = np.array(c_tmp)
     num_steps = int(T / dt) + 1
     time_start = perf_counter_ns()
     for _ in range(num_steps):
          for i in range(1, len(c) - 1):
               for j in range(1, len(c[0]) - 1):
                    c_tmp[i][j] = (c[i][j] + aux *
                                   (c[i - 1][j] + c[i + 1][j] +
                                    c[i][j - 1] + c[i][j + 1] -
                                    4 * c[i][j]))
          c, c_tmp = c_tmp, c # like a C++ swap() :D
     time_stop = perf_counter_ns()
     return c, time_stop - time_start


def diffuse_naive(c : List[List[float]],
                  c_tmp : List[List[float]],
                  T : float,
                  dt : float,
                  aux : float):
     num_steps = int(T / dt) + 1
     time_start = perf_counter_ns()
     for _ in range(num_steps):
          for i in range(1, len(c) - 1):
               for j in range(1, len(c[0]) - 1):
                    c_tmp[i][j] = (c[i][j] + aux *
                                   (c[i - 1][j] + c[i + 1][j] +
                                    c[i][j - 1] + c[i][j + 1] -
                                    4 * c[i][j]))
          c, c_tmp = c_tmp, c # like a C++ swap() :D
     time_stop = perf_counter_ns()
     return c, time_stop - time_start


def initialize_concentration(L : float, N : float, h : float):
     bound = 0.125 * L
     # +2 to add a border of boundary cells (boundary condition)
     c = [[0.0] * (N+2) for _ in range(N+2)]
     # Initialize non-boundary cells
     for i in range(N):
          for j in range(N):
               if (abs(i * h - 0.5 * L) < bound and
                   abs(j * h - 0.5 * L) < bound):
                    c[i+1][j+1] = 1.0
     c_tmp = [row[:] for row in c]
     return c, c_tmp


def get_constants(D : float, L : float, N : int):
     h = L / (N - 1)          # length between grid elements
     dt = h * h / (4 * D)     # maximum timestamp for numeric stability
     aux = dt * D / (h * h)   # all constant terms in diffusion equation
     return h, dt, aux


if __name__ == '__main__':
     parser = argparse.ArgumentParser()
     parser.add_argument('-D', type=float, required=False, default=1,
                         help='Diffusion constant')
     parser.add_argument('-L', type=float, required=False, default = 2,
                         help='Domain size')
     parser.add_argument('-N', type=int, required=False, default=100,
                         help='Number of grid points along each dim')
     parser.add_argument('-T', type=float, required=False, default=1.0,
                         help='Simulation time in seconds')
     parser.add_argument('--implementation', type=str, required=False, default='naive',
                         help='Diffusion implementation to execute')
     parser.add_argument('--output', action='store_true', required=False,
                         help='Whether to store results')
     args = parser.parse_args()
     D = args.D
     L = args.L
     N = args.N
     T = args.T
     implementation = args.implementation
     output = args.output
     h, dt, aux = get_constants(D, L, N)
     c, c_tmp = initialize_concentration(L, N, h)
     if implementation == 'naive':
          c, time_ns = diffuse_naive(c, c_tmp, T, dt, aux)
     elif implementation == 'numpy':
          c, time_ns = diffuse_numpy(c, c_tmp, T, dt, aux)
     else:
          raise ValueError(f'Implementation {implementation} does\'t exist')
     if output:
          save_results(c, implementation)
     print(f'Elapsed time: {time_ns}ns')
     plt.imshow(np.array(c, dtype=float), vmin=0, vmax=1, interpolation='none')
     plt.colorbar()
     plt.show()
