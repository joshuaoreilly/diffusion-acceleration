import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_overall(df : pd.DataFrame):
    fig, ax = plt.subplots()
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
    fig, ax = plt.subplots()
    implementations = df['Implementation'].unique()
    df_cpp_naive = df.loc[df['Implementation'] == 'cpp_naive']
    for _, row in df_cpp_naive.iterrows():
        N = row['N (M = N + 2)']
        time = row['Time (ns)']
        df.loc[df['N (M = N + 2)'] == N, 'Time (ns)'] = time / df.loc[df['N (M = N + 2)'] == N, 'Time (ns)']
    df.rename(columns={'Time (ns)' : 'Relative speedup'}, inplace=True)
    for implementation in implementations:
        df_imp = df.loc[df['Implementation'] == implementation]
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
    df = df.loc[(df['N (M = N + 2)'] <= 256) & (df['Relative speedup'] <= 1.0)]
    fig, ax = plt.subplots()
    for implementation in implementations:
        df_imp = df.loc[df['Implementation'] == implementation]
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
