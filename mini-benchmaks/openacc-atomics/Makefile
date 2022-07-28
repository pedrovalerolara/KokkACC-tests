KOKKOS_PATH = $(KOKKOS_ROOT)
KOKKOS_DEVICES = "OpenACC"
EXE_NAME = "OpenACC"
#KOKKOS_DEVICES = "Cuda"
#EXE_NAME = "Cuda"

SRC = $(wildcard *.cpp)

default: build
	echo "Start Build"

ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
CXXFLAGS = -O3
LINK = ${CXX}
LINKFLAGS =
EXE = ${EXE_NAME}.cuda
KOKKOS_ARCH = "Volta70"
##KOKKOS_ARCH = "Pascal60"
KOKKOS_CUDA_OPTIONS = "enable_lambda"
#ifneq (,$(findstring Openmp,$(KOKKOS_DEVICES)))
#CXX = g++
#EXE = ${EXE_NAME}.host
#KOKKOS_ARCH = "BDW"
else 
CXX = nvc++ 
CXXFLAGS = -O3 -Minfo=acc -acc
KOKKOS_CXX_STANDARD = c++17
#KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS = 0
LINK = ${CXX}
LINKFLAGS = -acc
EXE = ${EXE_NAME}.acc
KOKKOS_ARCH = "Volta70"
KOKKOS_CUDA_OPTIONS = "enable_lambda"
#KOKKOS_OPTIONS = disable_desul_atomics
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
	rm -f *.cuda *.acc *.host

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $<

test: $(EXE)
	./$(EXE)

