import argparse
from pathlib import Path
from typing import List
from time import perf_counter_ns
import numpy as np
import scipy as sp
import cupyx.scipy.signal as sgx
import cupy as cp
import torch
import cv2


"""
TODO:
- Profile to verify why numpy is slower (hypothesis: no benefit of underlying C++ data structure since doing single accesses at a time)
"""


def save_results(c, implementation):
     if implementation == 'naive':
          with open(Path('python_' + implementation + '.txt'), 'w') as file:
               for row in c:
                    file.writelines(','.join([str(i) for i in row]) + '\n')
     elif implementation in {'numpy', 'scipy', 'opencv', 'cupy', 'torch'}:
          np.savetxt('python_' + implementation + '.txt', c, fmt='%f', delimiter=',')
     else:
          raise ValueError(f'Implementation {implementation} does\'t exist')


def diffuse_torch(c : List[List[float]],
                 c_tmp : List[List[float]],
                 aux : float,
                 num_steps : int):
     width = len(c)
     gpu = torch.device('cuda')
     c = torch.tensor(c, dtype=torch.float64, device=gpu)
     c = torch.unsqueeze(torch.unsqueeze(c, 0), 0)
     c_tmp = torch.tensor(c_tmp, dtype=torch.float64, device=gpu)
     c_tmp = torch.unsqueeze(torch.unsqueeze(c_tmp, 0), 0)
     kernel = aux * torch.tensor([[0, 1, 0],
                                  [1, -4, 1],
                                  [0, 1, 0]],
                                  dtype=torch.float64,
                                  device=gpu)
     kernel = torch.unsqueeze(torch.unsqueeze(kernel, 0), 0)
     assert c.is_cuda
     assert c_tmp.is_cuda
     assert kernel.is_cuda
     time_start = perf_counter_ns()
     for _ in range(num_steps):
          c_tmp = c + torch.nn.functional.pad(
               torch.nn.functional.conv2d(c, kernel, padding='valid'),
               (1,1,1,1),
               'constant',
               0
          )
          c, c_tmp = c_tmp, c
     time_stop = perf_counter_ns()
     c_reshaped = torch.reshape(c, (width, width)).cpu().numpy()
     return  c_reshaped, time_stop - time_start


def diffuse_cupy(c : List[List[float]],
                 c_tmp : List[List[float]],
                 aux : float,
                 num_steps : int):
     c = cp.array(c)
     c_tmp = cp.array(c_tmp)
     kernel = aux * cp.array([[0, 1, 0],
                              [1, -4, 1],
                              [0, 1, 0]])
     time_start = perf_counter_ns()
     for _ in range(num_steps):
          c_tmp = c + cp.pad(
               sgx.convolve2d(c, kernel, mode='valid'),
                    1,
                    'constant'
               )
          c, c_tmp = c_tmp, c
     time_stop = perf_counter_ns()
     return c, time_stop - time_start


def diffuse_opencv(c : List[List[float]],
                   c_tmp : List[List[float]],
                   aux : float,
                   num_steps : int):
     c = np.array(c)
     c_tmp = np.array(c_tmp)
     kernel = aux * np.array([[0, 1, 0],
                              [1, -3, 1],
                              [0, 1, 0]])
     time_start = perf_counter_ns()
     for _ in range(num_steps):
          c_tmp = cv2.filter2D(src=c, kernel=kernel, borderType=cv2.BORDER_CONSTANT, value=0)
          c, c_tmp = c_tmp, c
     time_stop = perf_counter_ns()
     c = np.pad(c, 1,'constant')
     return c, time_stop - time_start


def diffuse_scipy(c : List[List[float]],
                  c_tmp : List[List[float]],
                  aux : float,
                  num_steps : int):
     c = np.array(c)
     c_tmp = np.array(c_tmp)
     kernel = aux * np.array([[0, 1, 0],
                              [1, -4, 1],
                              [0, 1, 0]])
     time_start = perf_counter_ns()
     for _ in range(num_steps):
          c_tmp = c + np.pad(
               sp.signal.convolve2d(c, kernel, mode='valid'),
                    1,
                    'constant'
               )
          c, c_tmp = c_tmp, c
     time_stop = perf_counter_ns()
     return c, time_stop - time_start


def diffuse_numpy(c : List[List[float]],
                  c_tmp : List[List[float]],
                  aux : float,
                  num_steps : int):
     c = np.array(c)
     c_tmp = np.array(c_tmp)
     time_start = perf_counter_ns()
     for _ in range(num_steps):
          for i in range(1, len(c) - 1):
               for j in range(1, len(c[0]) - 1):
                    c_tmp[i][j] = (c[i][j] + aux *
                                   (c[i - 1][j] + c[i + 1][j] +
                                    c[i][j - 1] + c[i][j + 1] -
                                    4 * c[i][j]))
          c, c_tmp = c_tmp, c
     time_stop = perf_counter_ns()
     return c, time_stop - time_start


def diffuse_naive(c : List[List[float]],
                  c_tmp : List[List[float]],
                  aux : float,
                  num_steps : int):
     time_start = perf_counter_ns()
     for _ in range(num_steps):
          for i in range(1, len(c) - 1):
               for j in range(1, len(c[0]) - 1):
                    c_tmp[i][j] = (c[i][j] + aux *
                                   (c[i - 1][j] + c[i + 1][j] +
                                    c[i][j - 1] + c[i][j + 1] -
                                    4 * c[i][j]))
          c, c_tmp = c_tmp, c
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
     num_steps = int(T / dt) + 1
     c, c_tmp = initialize_concentration(L, N, h)
     if implementation == 'naive':
          c, time_ns = diffuse_naive(c, c_tmp, aux, num_steps)
     elif implementation == 'numpy':
          c, time_ns = diffuse_numpy(c, c_tmp, aux, num_steps)
     elif implementation == 'scipy':
          c, time_ns = diffuse_scipy(c, c_tmp, aux, num_steps)
     elif implementation == 'opencv':
          c, time_ns = diffuse_torch(c, c_tmp, aux, num_steps)
     elif implementation == 'cupy':
          c, time_ns = diffuse_cupy(c, c_tmp, aux, num_steps)
     elif implementation == 'torch':
          c, time_ns = diffuse_torch(c, c_tmp, aux, num_steps)
     else:
          raise ValueError(f'Implementation {implementation} does\'t exist')
     if output:
          save_results(c, implementation)
     print(f'Elapsed time: {time_ns}ns')
