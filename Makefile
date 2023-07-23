# TODO: figure out why placing LIBDIRS and LIBS before source files
# causes compilation to fail
CC= g++
NVCC=nvcc
CXXFLAGS= -O3 -Wall -Wconversion -Wextra -pedantic -Wall -Wextra -pedantic -std=c++11
CUDAFLAGS=-std=c++11 -c
LIBS= -fopenmp -lcudart -lcuda
LIBDIRS= -L /usr/local/cuda/lib64

all: diffusion

diffusion: diffusioncuda.o
	$(CC) $(CXXFLAGS) diffusion.cpp diffusioncuda.o $(LIBDIRS) $(LIBS) -o diffusion

diffusioncuda.o:
	$(NVCC) $(CUDAFLAGS) diffusion.cu -o diffusioncuda.o

clean:
	rm -f *.o diffusion