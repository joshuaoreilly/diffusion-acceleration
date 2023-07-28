# Accelerating Diffusion

Experiments in accelerating simple 2D diffusion problem.

## Explanation of Optimizations

All the times in this section are done with the following arguments:

- D = 0.1
- L = 2.0
- N = 200
- T = 0.5

`g++ -O3 -Wall -Wextra -pedantic -Wall -Wextra -pedantic -std=c++14` is used to compile `diffusion.cpp`, except when specified otherwise.

### Python Naive

`diffuse_naive()`

Runtime: 17545110978ns = 17.5s

### Python Numpy

`diffuse_numpy()`

Using numpy for storing arrays -> since we're mostly performing memory accesses instead of actual compute, the additional overhead of querying the underlying C/C++ objects increases execution time.

Runtime: 94771412432ns = 94.7s

### C++ Naive - no O3

`diffuse_naive()`

Logically identical to the Python implementation, but in C++.

Runtime: 1059068632ns = 1.1s

### C++ Const, Restrict, and Inline - no O3

`diffuse_const_c`

The step loops have been moved to their own function so that the main array can be declared as `const` and `__restrict`, the temporary array as `__restrict`, and the entire function as `__inline__`
By declaring our main array as `const`, the compiler knows we won't edit it during this function.
By declaring them as `__restrict`, we tell the compiler the two arrays don't overlap, so it doesn't need to worry about them overlapping.

Runtime (just `const`): 1119420297ns = 1.1s

Runtime (just `__restrict`): 1116635178ns = 1.1s

Runtime (just `__inline__`): 1130495914ns = 1.1s

Runtime (all): 1148817841ns = 1.1s

### C++ Naive - with O3

Now we let the compiler work its magic.

Runtime: 48826736ns = 0.0488s

Huge improvement, and a big lesson: let the compiler optimize if you're not doing anything funky!

This is our *real* baseline for improvment.
I think writing a diffusion sim in Python is probably a pretty terrible idea even on the surface, and `-O3` is also an effortless improvment, so 0.05s is the time to beat from now on.

### C++ Const, Restrict, and Inline - with O3

Okay, so it was a little silly to expect telling the compiler to optimize, without letting it actually optimize, to speed things up.
Adding the `-O3` flag back to our compilation (where it will stay from now on):

Runtime (both): 54355785ns = 0.0543s

A little slower than the `-O3` version, showing just how much smarter the optimizer is than I am (also, `__restrict` apparently [doesn't affect](https://stackoverflow.com/questions/76747148/why-does-moving-for-loops-to-their-own-function-slow-down-the-program?noredirect=1#comment135304575_76747148) `std::vector` anyways); it probably already inlined the function, noticed `c` isn't edited in the inner for loops and did the equivalent of `const`, and a thousand other optimizations I never considered or know about.

### OpenMP

[OpenMP](https://en.wikipedia.org/wiki/OpenMP) is an API that allows for incredibly simple multithreading; instead of having to spawn threads, divide the workload manually based on the thread-id, then joint them at the end, we can `#include <omp.h>`, add a single `#pragma parallel for collapse(2)` statement which'll automatically spawn threads, distribute the work over the nested loops, then join them at the end for us, and finally at the compilation flag `fopenmp`, and we're off the the races, which so far, it's winning.

Runtime (`export OMP_NUM_THREADS=16`): 21742075ns = 0.021s

Over twice as fast as our reference implementation; not the 16x speedup we might naively hope for, but still useful.

TODO: add explanation of why non-linear scaling.

### CUDA Naive

First, a naive kernel that (TODO: add explanation, image from [here](https://stackoverflow.com/questions/32226993/understanding-streaming-multiprocessors-sm-and-streaming-processors-sp)).

```
int threadsPerBlock = 32;
int numberOfBlocks = 32 * multiProcessorCount;
```

Runtime: 23572605ns = 0.023s

Incrementally slower than our parallelized C++ version for a array width of 200, but once the array gets larger, the performance gap widens; with $N = 1500$, C++ naive took 223 seconds, C++ OpenMP took 131 seconds, and CUDA took 21 seconds.

## Requirements

- `g++`
- numpy
- matplotlib
- cuda/nccc (get an actual requirement and put it here)
- openmp? (get an actual requirement and put it here)

## Lessons Learned

- CUDA will throw `CUDA Runtime Error: an illegal memory access was encountered` if a reference is passed to a kernel
- Declaring arguments as `const` in a kernel is unecessary, as [all values](https://stackoverflow.com/questions/65015858/cuda-kernel-do-i-need-to-put-const-in-all-pass-by-value-parameters) are [passed as const](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#function-parameters)
- It's probably better to work directly with C-style arrays than vectors, since CUDA doesn't support them, and we need to do some annoying conversions to get it onto the GPU

## To Do

- Compare different choices of thread counts with OpenMP
- Compare different choices of thread and block count with CUDA
- CUDA version with dynamic shared memory -> 'stride' within a contiguous chunk of the array, separate x and y derivatives
- omp simd to vectorize C++ implementation
- Manually vectorize C++ implementation (but don't multithread)
- Manually thread and vectorize C++ implementation
- Look into pinning threads on certain CPUs?
- Parallelize Python code
- Add memcpy from host to device to time, explain why it's also probably not as relevant in this case, since we could have performed the allocation of c on the GPU direclty if I wasn't running everything/allocating from my C++ file
- Run as float instead of double maybe add the image from here? It's techincally for a different GPU, but illustrates quite well the availability of registers for different floating point number sizes (https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)
