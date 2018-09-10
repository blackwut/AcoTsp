# CXX			= g++ 
# CPPFLAGS	= -O3 -g -std=c++14 -pedantic -Wall
# INCLUDES	= -I . ~/Projects/fastflow/
# LD_FLAGS 	= -pthread

# OBJECTS_CPU = main.cpp common.hpp TSP.o AcoCpu.o
# OBJECTS_GPU = GPUAco.cu ACO.o TSP.o

# acogpu: $(OBJECTS_GPU)
# 	nvcc -Xptxas="-v" -O3 -g -lineinfo $< -o $@

# acocpu_san: $(OBJECTS_CPU)
# 	$(CXX) $(CPPFLAGS) -fsanitize=thread $< -o $@ $(LD_FLAGS)

CXX			= g++
CXXFLAGS	= -std=c++14 -O3 -Wall -pedantic
INCLUDES	= -I . -I ~/Projects/fastflow
LIBS		= -pthread

SRCS	= main.cpp TSP.cpp Environment.cpp Parameters.cpp Ant.cpp AcoCpu.cpp
OBJS	= $(SRCS:.cpp=.o)
ACOCPU	= acocpu

$(ACOCPU): $(OBJS) 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(ACOCPU) $(OBJS) $(LIBS)

acogpu: GPUAco.cu TSP.o
	nvcc -Xptxas="-v" -O3 -g -lineinfo $< -o $@

AcoCpu.o : Environment.cpp Parameters.cpp Ant.cpp

%.o : %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<  -o $@

.PHONY: clean
clean:
	$(RM) -f *.o *~ $(ACOCPU) 

# DO NOT DELETE THIS LINE -- make depend needs it

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
