CXX = g++
CXX_FLAGS = -O3 -std=c++11 -stdlib=libc++ -Wc++11-extensions -Wall -g -I ~/Projects/fastflow/
LD_FLAGS	= -pthread

OBJECTS = main.cpp TSPReader.cpp Aco.cpp Ant.cpp common.hpp
OBJECTS_GPU = GPUAco.cu TSPReader.cpp common.hpp
OBJECTS_FF = FFAco.cpp TSPReader.cpp common.hpp


acogpu: $(OBJECTS_GPU) 
	nvcc -O3 -g -lineinfo $< -o $@

acocpu: $(OBJECTS)
	$(CXX) $(CXX_FLAGS) $< -o $@ $(LD_FLAGS)

acoff: $(OBJECTS_FF)
	$(CXX) $(CXX_FLAGS) $< -o $@ $(LD_FLAGS) -DTRACE_FASTFLOW

clean:	
	rm -f acogpu acocpu acoff
