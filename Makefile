CXX			= g++
# CPP7 		= /usr/local/Cellar/gcc\@7/7.3.0/bin/c++-7
CXXFLAGS	= -std=c++14 -O3 -Wall -pedantic #-fsanitize=thread #-Waligned-new=none
INCLUDES	= -I . -I ~/Projects/fastflow
LIBS		= -lpthread

SRCS	= main.cpp TSP.cpp Environment.cpp Parameters.cpp Ant.cpp AcoCpu.cpp AcoFF.cpp
OBJS	= main.cpp TSP.o Environment.o Parameters.o Ant.o AcoCpu.o AcoFF.o
ACOCPU	= acocpu
ACOGPU	= acogpu

$(ACOCPU): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) -o $(ACOCPU) $(LIBS)

$(ACOGPU): GPUAco.cu TSP.o
	nvcc -Xptxas="-v" -O3 -lineinfo $< -o $@

%.o : %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<  -o $@

.PHONY: clean
clean:
	$(RM) -f *.o *~ $(ACOCPU) $(ACOGPU)