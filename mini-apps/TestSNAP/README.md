<!----------------BEGIN-HEADER------------------------------------>
Instructions to build Kokkos OpenMPTarget backend with various compilers
and then build TestSNAP with the said backend. 

## Clone Kokkos develop branch

```
git clone --single-branch --branch develop https://github.com/kokkos/kokkos.git 
```

## Build OpenMPTarget backend 

### LLVM compiler
```
mkdir build_ompt_clang && cd build_ompt_clang

cmake -D CMAKE_INSTALL_PREFIX=$PWD/../install_ompt_clang/ \
    -D CMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_STANDARD=17 \
    -D CMAKE_CXX_FLAGS="-Wno-unknown-cuda-version -Werror -Wno-undefined-internal -Wno-pass-failed" \
    -D Kokkos_ARCH_VOLTA70=ON -D Kokkos_ENABLE_OPENMPTARGET=ON -D Kokkos_ENABLE_SERIAL=ON \
    ../ 

    make -j8
    make install
```

### NVHPC compiler

```
mkdir build_ompt_nvhpc && cd build_ompt_nvhpc

cmake -D CMAKE_INSTALL_PREFIX=$PWD/../install_ompt_nvhpc/ \
    -D CMAKE_CXX_COMPILER=nvc++ -D CMAKE_CXX_STANDARD=17 \
    -D Kokkos_ARCH_VOLTA70=ON -D Kokkos_ENABLE_OPENMPTARGET=ON -D Kokkos_ENABLE_SERIAL=ON \
    ../ 

    make -j8
    make install
```

## Clone TestSNAP kokkos-nvhpc branch

```
git clone --single-branch --branch kokkos-nvhpc https://github.com/rgayatri23/TestSNAP.git
```

## Build TestSNAP with OpenMPTarget backend 

### LLVM compiler
```
mkdir build_ompt_clang
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_COMPILER=clang++ \
    -D Kokkos_DIR=$PATH-To-Kokkos-clang-install/lib64/cmake/Kokkos/ \
    -D ref_data=14 ../

    make 

    ./test_snap
```

### NVHPC compiler

```
mkdir build_ompt_clang
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_COMPILER=nvc++ \
    -D Kokkos_DIR=$PATH-To-Kokkos-nvhpc-install/lib64/cmake/Kokkos/ \
    -D ref_data=14 ../

    make 

    ./test_snap
```
<!-----------------END-HEADER------------------------------------->

