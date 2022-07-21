#ifndef miniFE_info_hpp
#define miniFE_info_hpp

#define MINIFE_HOSTNAME "leconte"
#define MINIFE_KERNEL_NAME "'Linux'"
#define MINIFE_KERNEL_RELEASE "'4.18.0-240.1.1.el8_3.ppc64le'"
#define MINIFE_PROCESSOR "'ppc64le'"

#define MINIFE_CXX "'/opt/nvidia/hpc_sdk/Linux_ppc64le/21.3/compilers/bin/nvc++'"
#define MINIFE_CXX_VERSION "'nvc++ 21.3-0 linuxpower target on Linuxpower '"
#define MINIFE_CXXFLAGS "'-O3  -acc -DKOKKOS_HAVE_OPENACC -DUSE_MPI=0 -acc -DMPICH_IGNORE_CXX_SEEK -fPIC -DMINIFE_SCALAR=double -DMINIFE_LOCAL_ORDINAL=int -DMINIFE_GLOBAL_ORDINAL=int -DMINIFE_CSR_MATRIX -I./ -I/home/f3g/KOKKOS/kokkos-openacc/mini-apps/miniFE/miniFE/kokkos/src/.. -I/home/f3g/KOKKOS/kokkos-openacc/mini-apps/miniFE/miniFE/kokkos/src/../src -I/home/f3g/KOKKOS/kokkos-openacc/mini-apps/miniFE/miniFE/kokkos/src/../kokkos/linalg/src  -I/home/f3g/KOKKOS/kokkos-openacc/mini-apps/miniFE/miniFE/kokkos/src/../fem -I/home/f3g/KOKKOS/kokkos-openacc/mini-apps/miniFE/miniFE/kokkos/src/../utils -I/home/f3g/KOKKOS/kokkos-openacc/mini-apps/miniFE/miniFE/kokkos/src/../common   -DMINIFE_INFO=1 -DMINIFE_KERNELS=0 -DUSE_MPI_WTIME'"

#endif
