CXX = g++
CXX_FLAGS = -O3 -std=c++11 -stdlib=libc++ -Wc++11-extensions -Wall -Wextra -I.
LD_FLAGS	= -pthread

OBJECTS = main.cpp TSPReader.cpp Aco.cpp Ant.cpp common.hpp
OBJECTS_GPU = GPUAco.cu TSPReader.cpp


acogpu: $(OBJECTS_GPU) 
	nvcc -O3 -g -lineinfo $< -o $@

acocpu: $(OBJECTS)
	$(CXX) $(CXX_FLAGS) $< -o $@ $(LD_FLAGS)

clean:	
	rm -f acogpu acocpu
