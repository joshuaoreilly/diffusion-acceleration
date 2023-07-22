import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=Path, required=True,
                        help='Name/path of file to load and visualize')
    args = parser.parse_args()
    file = args.file
    if not file.is_file():
        raise IOError(f'File doesn\'t exist: {file}')
    c = np.loadtxt(file, delimiter=',')
    plt.imshow(np.array(c, dtype=float), vmin=0, vmax=1, interpolation='none')
    plt.colorbar()
    plt.show()
