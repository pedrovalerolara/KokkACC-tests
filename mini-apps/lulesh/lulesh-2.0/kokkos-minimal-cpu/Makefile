#Set your Kokkos path to something appropriate
#KOKKOS_PATH = ${HOME}/kokkos
KOKKOS_PATH = ../../../../../
#KOKKOS_DEVICES = "Serial"
#KOKKOS_DEVICES = "OpenMP"
KOKKOS_DEVICES = "Cuda"
#KOKKOS_ARCH = 
#KOKKOS_ARCH = "Power9"
KOKKOS_ARCH = "Volta70"
KOKKOS_CUDA_OPTIONS = "enable_lambda"

SRC = $(wildcard *.cc)

default: build
	echo "Start Build"
	
CXXFLAGS = -g -O3 -DUSE_OMP=1 -DRAJA_USE_GETTIME -I./includes -I./

#ifeq ( $(USE_MPI), yes) 
#CXX = mpicxx
#CXXFLAGS += -DUSE_MPI=1
#else
#CXX = clang++
#CXX = g++
#CXX = g++
CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
CXXFLAGS += -DUSE_MPI=0
#endif

LINK = ${CXX}

LINKFLAGS =  
EXE = lulesh.host

DEPFLAGS = -M

OBJ = $(SRC:.cc=.o)
LIB = -ldl

include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean 
	rm -f *.o 

# Compilation rules

%.o:%.cc $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $<



