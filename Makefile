CXX = g++
CXX_FLAGS = -O3 -std=c++14 -Wall -I ~/Projects/fastflow/
LD_FLAGS	= -pthread

OBJECTS_CPU = main.cpp TSPReader.cpp AcoCpu.cpp AcoFF.cpp common.hpp random.hpp
OBJECTS_GPU = GPUAco.cu TSPReader.cpp common.hpp
OBJECTS_FF = main.cpp TSPReader.cpp AcoCpu.cpp AcoFF.cpp common.hpp random.hpp

acocpu: $(OBJECTS_CPU)
	$(CXX) $(CXX_FLAGS) $< -o $@ $(LD_FLAGS) -DACO_CPU

acogpu: $(OBJECTS_GPU) 
	nvcc -O3 -g -lineinfo $< -o $@

acoff: $(OBJECTS_CPU)
	$(CXX) $(CXX_FLAGS) $< -o $@ $(LD_FLAGS) -DACO_FF -DTRACE_FASTFLOW

clean:	
	rm -f acogpu acocpu acoff
