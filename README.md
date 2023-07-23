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

## Requirements

- `g++`
- numpy
- matplotlib
- cuda/nccc (get an actual requirement and put it here)
- openmp? (get an actual requirement and put it here)

## To Do

- Find ways to speed up python code