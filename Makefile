CXX			:= g++ 
CXX_FLAGS	:= -O3 -g -std=c++14 -pedantic -Wall -Waligned-new=none 
INCLUDES	:=-I . ~/Projects/fastflow/
LD_FLAGS	:= -pthread

OBJECTS_CPU = main.cpp ACO.o TSP.o AcoCpu.o AcoFF.o
OBJECTS_GPU = GPUAco.cu ACO.o TSP.o

acocpu: $(OBJECTS_CPU)
	$(CXX) $(CXX_FLAGS) $< -o $@ $(LD_FLAGS)

acogpu: $(OBJECTS_GPU)
	nvcc -Xptxas="-v" -O3 -g -lineinfo $< -o $@

acocpu_san: $(OBJECTS_CPU)
	$(CXX) $(CXX_FLAGS) -fsanitize=thread $< -o $@ $(LD_FLAGS)

%.o: %.cxx
	$(CXX) $(CXX_FLAGS) -c $(input) -o $(output)

clean:
	rm -f acocpu acogpu *.o

# CXX 		= g++
# CPP7 		= /usr/local/Cellar/gcc\@7/7.3.0/bin/c++-7
# CXX_FLAGS	= -O3 -g -std=c++14 -pedantic -Wall -Waligned-new=none -I ~/Projects/fastflow/
# LD_FLAGS	= -pthread

# OBJECTS_CPU = main.cpp ACO.cpp TSP.cpp AcoCpu.cpp AcoFF.cpp common.hpp
# OBJECTS_GPU = GPUAco.cu common.hpp TSP.cpp Aco.cpp

# acocpu: $(OBJECTS_CPU)
# 	$(CXX) $(CXX_FLAGS) $< -o $@ $(LD_FLAGS)

# acocpu7: $(OBJECTS_CPU)
# 	$(CPP7) $(CXX_FLAGS) $< -o $@ $(LD_FLAGS)


# acogpu: $(OBJECTS_GPU)
# 	nvcc -Xptxas="-v" -O3 -g -lineinfo $< -o $@

# acocpu_san: $(OBJECTS_CPU)
# 	$(CXX) $(CXX_FLAGS) -fsanitize=thread $< -o $@ $(LD_FLAGS)

# clean:	
# 	rm -f acocpu acogpu
