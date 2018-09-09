# CXX			= g++ 
# CPPFLAGS	= -O3 -g -std=c++14 -pedantic -Wall
# INCLUDES	= -I . ~/Projects/fastflow/
# LD_FLAGS 	= -pthread

# OBJECTS_CPU = main.cpp common.hpp TSP.o AcoCpu.o
# OBJECTS_GPU = GPUAco.cu ACO.o TSP.o

# acocpu: $(OBJECTS_CPU)
# 	$(CXX) $(CPPFLAGS) $< -o $@ $(LD_FLAGS)

# acogpu: $(OBJECTS_GPU)
# 	nvcc -Xptxas="-v" -O3 -g -lineinfo $< -o $@

# acocpu_san: $(OBJECTS_CPU)
# 	$(CXX) $(CPPFLAGS) -fsanitize=thread $< -o $@ $(LD_FLAGS)

CC			= g++
CPPFLAGS	= -Wall -pedantic -std=c++14 -O3
INCLUDES	= -I . -I ~/Projects/fastflow
LIBS		= -pthread

SRCS	= main.cpp TSP.cpp Environment.cpp Parameters.cpp Ant.cpp AcoCpu.cpp
OBJS	= $(SRCS:.cpp=.o)
MAIN	= acocpu

$(MAIN): $(OBJS) 
	$(CC) $(CPPFLAGS) $(INCLUDES) -o $(MAIN) $(OBJS) $(LIBS)

# this is a suffix replacement rule for building .o's from .c's
# it uses automatic variables $<: the name of the prerequisite of
# the rule(a .c file) and $@: the name of the target of the rule (a .o file) 
# (see the gnu make manual section about automatic variables)
.cpp.o:
	$(CC) $(CPPFLAGS) $(INCLUDES) -c $<  -o $@

# %.o: %.cxx
# 	$(CXX) -O3 -g -std=c++14 -c $(input) -o $(output)

clean:
	$(RM) *.o *~ $(MAIN)

depend: $(SRCS)
	makedepend $(INCLUDES) $^

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
