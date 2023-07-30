# Accelerating a Diffusion Simulation

Experiments in accelerating a simple 2D heat diffusion simulation.
For implementation details and results, visit [here](https://joshuaoreilly.com/Projects/diffusion-acceleration.html).

https://github.com/joshuaoreilly/diffusion-acceleration/assets/26682773/6c8c0ce9-a591-4ade-807a-33fff83ef2ed

## Requirements

- g++ or another compiler that supports OpenMP
- numpy
- matplotlib
- scipy
- cupy
- torch
- cuda/nvcc

I don't think the particular version matters all that much.
Something recent-ish, at least?

## Usage

### Python

To run a single Python implementation:

```
python diffusion.py -D 0.1 -L 2.0 -N 200 -T 0.5 --implementation torch --output
```

where `D` is the diffusion coefficient, `L` is the width of the system we're working on, `N` is the width of the array (calculations scale to the power of four with width), `T` is the simulation time in seconds, `implementation` is either `naive`, `numpy`, `scipy`, `cupy`, or `torch`, and `--output` can be selected if you'd like the final array to be output to a text file for visualization with `visualize.py` or compared to the output of another for accuracy with `validate.py`.

### C++/CUDA

The C++/CUDA executable can be built by running:

```
make clean # (if there are leftover build files from a previous build of yours)
make
```

To run a single C++/CUDA implementation:

```
./diffusion 0.1 2.0 N 0.5 0 implementation
```

where the arguments are in order: `D`, `L`, `N`, `T`, `output` (0 or 1), and `implementation`, which can be one of `naive`, `openmp`, or `cuda`.

### Benchmark

The full benchmark can be run by:

```
bash benchmark.bash your-file-name-here.txt
```

This will produce a file called `your-file-name-here.txt` (or `results.txt`, if no argument given) which can be visualized by running:

```
python plot.py --file your-file-name-here.txt
```

## The Math

I've left out most of the math, since I'm more here for speed-ups then theory, but in short, the heat-flow in a 2D medium can be described by

$$
\frac{\partial c(x, t)}{\partial t} = D \nabla^2 c(x,t)
$$

where $c(x,t)$ is the concentration of heat at position $x = (x_1,x_2)$ and $D$ is a constant diffusion coefficient $\frac{\partial x^2}{\partial t}$.
Following some derivations you can find more details on [here](https://hplgit.github.io/fdm-book/doc/pub/diffu/pdf/diffu-4print.pdf), [here](https://hplgit.github.io/fdm-book/doc/pub/book/sphinx/._book011.html), and [on Wikipedia](https://en.wikipedia.org/wiki/Finite_difference), we get the discritized diffusion process using a central finite difference scheme in space and forward Euler time integration:

$$
\frac{c_{i,j}^{n+1} - c_{i,j}^{n}}{\Delta t} = D \left[ \frac{c_{i-1,j}^n - 2c_{i,j}^n + c_{i+1,j}^n}{\Delta x^2} + \frac{ c_{i,j-1}^n - 2c_{i,j}^n + c_{i,j+1}^n}{\Delta y^2} \right]
$$

where $c_{i,j}^n$ is the heat concentration at position $i, j$ at time n.
Since we're working with a square grid, $h = \Delta x = \Delta y$.
Isolating the new concentration $c_{i,j}^{n+1}$:

$$
c_{i,j}^{n+1} = c_{i,j}^{n} +  \frac{D \Delta t}{h^2}  \left[ c_{i-1,j}^n + c_{i+1,j}^n + c_{i,j-1}^n + c_{i,j+1}^n - 4c_{i,j}^n \right]
$$

where $\frac{D \Delta t}{h^2}$ is the `aux` term which shows up in the code.

One of my roommates pointed out that this equation is effectively convoluting the 2D array by a $3 \times 3$ kernel, namely:

$$
c^{n+1} = c^{n} + c^{n} * \left( \frac{D \Delta t}{h^2}
    \begin{bmatrix}
    0 & 1 & 0\\
    1 & -4 & 1 \\
    0 & 1 & 0
    \end{bmatrix}
    \right)
$$

where $*$ is the convolution operation and $c^{n}$ is the entire 2D array at time $n$.

While exlicit Euler has the advantage of not needing to solve a system of equations to perform a step forward in time, it can be unstable if the time step is too large.
The maximum stable step size can be computed using von Neumann analysis the process of which is in [this pdf](https://joshuaoreilly.com/static/diffusion-von-neumann.pdf), and the result of which is the following:

$$
\Delta t \leq \frac{h^2}{4D}
$$

If we fix $D$, then as the array width $x = y = h$ doubles, the maximum timestep shrinks by a factor of 4.

Finally, I've set the boundary condition of $c_{i,0} = c_{0,j} = 0$: the borders are always equal to zero, meaning heat is slowly being leeched from the system.

## Implementations and Results

I've detailed the implementations and results on my [website](https://joshuaoreilly.com/Projects/diffusion-acceleration.html)
