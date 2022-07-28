KOKKOS_PATH = $(KOKKOS_ROOT)
EXE_NAME = "Stream"
# For OpenACC backend
KOKKOS_DEVICES ?= "OpenACC"
# For OpenMPTarget backend
#KOKKOS_DEVICES ?= "OpenMPTarget"
# For Cuda backend
#KOKKOS_DEVICES ?= "Cuda"

SRC = $(wildcard *.cpp)

default: build
	echo "Start Build"

ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
CXXFLAGS = -O3
LINK = ${CXX}
LINKFLAGS =
EXE = ${EXE_NAME}.${KOKKOS_DEVICES}
KOKKOS_ARCH = "Volta70"
##KOKKOS_ARCH = "Pascal60"
KOKKOS_CUDA_OPTIONS = "enable_lambda"
#ifneq (,$(findstring Openmp,$(KOKKOS_DEVICES)))
#CXX = g++
#EXE = ${EXE_NAME}.host
#KOKKOS_ARCH = "BDW"
else ifneq (,$(findstring OpenMPTarget,$(KOKKOS_DEVICES)))
#For Clang compiler
CXX ?= clang++ 
CXXFLAGS = -Wall
#For NVHPC compiler 
#CXX ?= nvc++
#CXXFLAGS = -Wall 
LINK = ${CXX}
LINKFLAGS = 
EXE = ${EXE_NAME}.${KOKKOS_DEVICES}
KOKKOS_ARCH = "Volta70"
KOKKOS_CXX_STANDARD = c++17
else 
CXX ?= nvc++ 
#CXXFLAGS = 
LINK = ${CXX}
EXE = ${EXE_NAME}.${KOKKOS_DEVICES}
KOKKOS_ARCH = "Volta70"
KOKKOS_CXX_STANDARD = c++17
KOKKOS_OPTIONS = disable_desul_atomics
endif


DEPFLAGS = -M

OBJ = $(SRC:.cpp=.o)
LIB =

include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean
	rm -f *.o

purge: clean
	rm -f $(EXE)

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $<

test: $(EXE)
	./$(EXE)

