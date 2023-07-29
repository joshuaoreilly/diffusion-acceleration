import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_overall(df : pd.DataFrame):
    _, ax = plt.subplots()
    implementations = df['Implementation'].unique()
    for implementation in implementations:
        df_imp = df.loc[df['Implementation'] == implementation]
        ax.plot(df_imp['N (M = N + 2)'],
                df_imp['Time (ns)'].apply(lambda x: x/1000000000),
                label=implementation)
    ax.set_yscale('log')
    ax.set_title('Execution time as a function of array width')
    ax.set_xlabel('N')
    ax.set_ylabel('Execution time (s)')
    ax.set_xticks([16, 32, 64, 128, 256, 512, 1024])
    ax.legend()
    ax.grid(True)
    plt.show()


def plot_speedup(df : pd.DataFrame):
    df2 = df.copy()
    _, ax = plt.subplots()
    implementations = df2['Implementation'].unique()
    df_cpp_naive = df2.loc[df2['Implementation'] == 'cpp_naive']
    for _, row in df_cpp_naive.iterrows():
        N = row['N (M = N + 2)']
        time = row['Time (ns)']
        df2.loc[df2['N (M = N + 2)'] == N, 'Time (ns)'] = time / df2.loc[df2['N (M = N + 2)'] == N, 'Time (ns)']
    df2 = df2.rename(columns={'Time (ns)' : 'Relative speedup'})
    for implementation in implementations:
        df_imp = df2.loc[df['Implementation'] == implementation]
        ax.plot(df_imp['N (M = N + 2)'],
                df_imp['Relative speedup'],
                label=implementation)
    ax.set_title('Speedup relative to naive C++ implementation')
    ax.set_xlabel('N')
    ax.set_ylabel('Relative speedup')
    ax.set_xticks([16, 32, 64, 128, 256, 512, 1024])
    #ax.set_yticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ax.legend()
    ax.grid(True)
    plt.show()

    # Smaller arrays only for the slow implementations
    df2 = df2.loc[(df2['N (M = N + 2)'] <= 256) & (df2['Relative speedup'] <= 1.0)]
    fig, ax = plt.subplots()
    for implementation in implementations:
        df_imp = df2.loc[df2['Implementation'] == implementation]
        ax.plot(df_imp['N (M = N + 2)'],
                df_imp['Relative speedup'],
                label=implementation)
    ax.set_yscale('log')
    ax.set_title('Speedup relative to naive C++ implementation')
    ax.set_xlabel('N')
    ax.set_ylabel('Relative speedup')
    ax.set_xticks([16, 32, 64, 128, 256])
    #ax.set_yticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ax.legend()
    ax.grid(True)
    plt.show()


def plot_cuda_IO(df : pd.DataFrame):
    _, ax = plt.subplots()
    df_cuda = df.loc[df['Implementation'] == 'cuda']
    rows = df_cuda.shape[0]
    bottom = np.zeros(rows)
    ax.bar(df_cuda['N (M = N + 2)'].astype(str),
           df_cuda['Time (ns)'].apply(lambda x: x/1000000000),
           label='Compute Time',
           bottom=bottom)
    bottom += df_cuda['Time (ns)'].apply(lambda x: x/1000000000)
    ax.bar(df_cuda['N (M = N + 2)'].astype(str),
           df_cuda['IO Time (ns) if applicable'].apply(lambda x: x/1000000000),
           label='IO Time',
           bottom=bottom)
    ax.set_yscale('log')
    ax.set_title('Compute time and IO Time of CUDA implementation')
    ax.set_xlabel('N')
    ax.set_ylabel('Time (s)')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=Path, required=False, default='results.txt')
    args = parser.parse_args()
    filepath = args.file
    if not filepath.is_file():
        raise IOError(f"File {filepath} doesn't exist")
    df = pd.read_csv(filepath, skipinitialspace=True)
    plot_overall(df)
    plot_speedup(df)
    plot_cuda_IO(df)
