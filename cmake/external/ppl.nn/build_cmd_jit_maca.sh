#!/bin/bash

#export MACA_PATH=/path/to/C500/maca/maca-current
export PATH=$MACA_PATH/mxgpu_llvm/bin:$MACA_PATH/bin:${MACA_PATH}/tools/wcuda/tools:$PATH
export LD_LIBRARY_PATH=$MACA_PATH/lib:$LD_LIBRARY_PATH
#export CUDA_PATH=$MACA_PATH/tools/wcuda
export CUDA_PATH=~/wcuda/CUDA_DIR
export CUDA_TOOLKIT_ROOT=$CUDA_PATH
export WCUDA_DEBUG=1


workdir=`pwd`

./build.sh -DPPLNN_USE_X86_64=ON \
                -DPPLNN_USE_CUDA=ON \
                -DPPLNN_ENABLE_PYTHON_API=ON \
                -DPPLNN_ENABLE_CUDA_JIT=ON\
                -DPPLNN_USE_MACA=ON

