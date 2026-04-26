CXX    := g++
CXXFLAGS := -O3 -march=native -fopenmp -std=c++17
LIBS   := -lm

NVCC   := nvcc
NVFLAGS := -O3 -arch=sm_70 -std=c++17

.PHONY: all clean

all: attention conv conv_cuda

attention: attention.cpp timing.h
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBS)

conv: conv.cpp timing.h
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBS)

conv_cuda: conv.cu timing.h
	$(NVCC) $(NVFLAGS) $< -o $@

clean:
	rm -f attention conv conv_cuda
