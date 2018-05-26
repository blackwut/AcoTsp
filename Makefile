CXX 		= g++
CPP7 		= /usr/local/Cellar/gcc\@7/7.3.0/bin/c++-7
CXX_FLAGS	= -O3 -g -std=c++14 -pedantic -Wall -Waligned-new=none -I ~/Projects/fastflow/
LD_FLAGS	= -pthread

OBJECTS_CPU = main.cpp TSPReader.cpp AcoCpu.cpp AcoFF.cpp common.hpp random.hpp
OBJECTS_GPU = GPUAco.cu TSPReader.cpp common.hpp

acocpu: $(OBJECTS_CPU)
	$(CXX) $(CXX_FLAGS) $< -o $@ $(LD_FLAGS)

acocpu7: $(OBJECTS_CPU)
	$(CPP7) $(CXX_FLAGS) $< -o $@ $(LD_FLAGS)


acogpu: $(OBJECTS_GPU) 
	nvcc -O3 -g -lineinfo $< -o $@

acocpu_san: $(OBJECTS_CPU)
	$(CXX) $(CXX_FLAGS) -fsanitize=thread $< -o $@ $(LD_FLAGS)

clean:	
	rm -f acocpu acogpu
