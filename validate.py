import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


"""
Compares output of two diffusion simulations

TODO:
- Ensure all values in both are in [0, 1]
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', type=Path, required=True,
                        help='Name/path of first .txt file')
    parser.add_argument('--b', type=Path, required=True,
                        help='Name/path of second .txt file')
    args = parser.parse_args()
    file_a = args.a
    file_b = args.b
    if not file_a.is_file():
        raise IOError(f'File doesn\'t exist: {file_a}')
    if not file_b.is_file():
        raise IOError(f'File doesn\'t exist: {file_b}')
    a = np.loadtxt(file_a, delimiter=',')
    b = np.loadtxt(file_b, delimiter=',')
    allclose = np.allclose(a, b)
    if allclose:
        print('Within acceptable margin!')
    else:
        deltas = np.multiply(np.logical_not(np.isclose(a, b)), np.abs(a - b))
        i, j = np.unravel_index(deltas.argmax(), deltas.shape)
        print(f'Unacceptable margin, worst instance is where a is {a[i,j]}, b is {b[i,j]}, delta is {deltas[i,j]}')
