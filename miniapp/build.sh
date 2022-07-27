#!/bin/sh

rm -rfv build
mkdir build
cd build
#cmake ../ && make
export CC=/spack/apps/gcc/8.3.0/bin/gcc
export CXX=/spack/apps/gcc/8.3.0/bin/g++
export CMAKE_CXXFLAGS=-D_GLIBCXX_USE_CXX11_ABI=0
#cmake ../ && make
cmake  -DCMAKE_PREFIX_PATH=/project/priyav_216/tlinker/libtorch/ ..
cmake   --build . --config Release
cp example-app ..
