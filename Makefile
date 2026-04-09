CXX    := g++
CXXFLAGS := -O3 -march=native -fopenmp -std=c++17
LIBS   := -lm

.PHONY: all clean

all: attention conv

attention: attention.cpp timing.h
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBS)

conv: conv.cpp timing.h
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBS)

clean:
	rm -f attention conv
