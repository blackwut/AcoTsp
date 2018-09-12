CXX			= g++
CXXFLAGS	= -std=c++14 -O3 -Wall -pedantic #-faligned-new #-fsanitize=thread #-Waligned-new=none
INCLUDES	= -I . -I ~/Projects/fastflow
LIBS		= -lpthread

OBJS	= main.cpp TSP.o Environment.o Parameters.o Ant.o AcoCpu.o AcoFF.o
ACOCPU	= acocpu
ACOGPU	= acogpu

$(ACOCPU): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.cpp -o $(ACOCPU) $(LIBS)

$(ACOGPU): AcoGPU.cu TSP.cpp
	nvcc -Xptxas="-v" -O3 -lineinfo -c TSP.cpp -o TSP.o
	nvcc -Xptxas="-v" -O3 -lineinfo GPUAco.cu -o $@

%.o : %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

.PHONY: clean
clean:
	$(RM) *.o *~ $(ACOCPU) $(ACOGPU)
