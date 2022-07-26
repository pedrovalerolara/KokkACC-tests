# Kokkos minimally requires 3.10 right now,
# but your project can set it higher
cmake_minimum_required(VERSION 3.10)

# Projects can safely mix languages - must have C++ support
# Kokkos flags will only apply to C++ files
project(Example CXX Fortran)

# You need this for using Kokkos_ROOT variable
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.12.0")
  message(STATUS "Setting policy CMP0074 to use <Package>_ROOT variables")
  cmake_policy(SET CMP0074 NEW)
endif()

# Look for an installed Kokkos
find_package(Kokkos REQUIRED)

add_executable(openacctest openacc-test.cpp) 

# This is the only thing required to set up compiler/linker flags
target_link_libraries(openacctest Kokkos::kokkos)

enable_testing()
add_test(NAME KokkosInTree_Verify COMMAND openacctest)
