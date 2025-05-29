#!/bin/bash

export CUDA_TOOLKIT_ROOT=/usr/local/cuda

workdir=`pwd`

./build.sh -DPPLNN_USE_X86_64=ON \
           -DPPLNN_USE_CUDA=ON \
           -DPPLNN_ENABLE_PYTHON_API=ON \
           -DPPLNN_ENABLE_CUDA_JIT=ON
