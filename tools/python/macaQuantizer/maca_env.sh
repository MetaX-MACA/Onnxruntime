# export MACA_QUANTIZER_USING_MXGPU=0   # 0:Disable  1:Enable(default)
# export MACA_PATH=/opt/maca
export MACA_PATH=${1}
export CUCC_PATH=${MACA_PATH}/tools/cu-bridge

export MACA_CLANG_PATH=${MACA_PATH}/mxgpu_llvm/bin/
export DEVINFO_ROOT=${MACA_PATH}
export LD_LIBRARY_PATH=${MACA_PATH}/lib/:${MACA_PATH}/mxgpu_llvm/lib/:${LD_LIBRARY_PATH}
# export PATH=${MACA_PATH}/mxgpu_llvm/bin:${MACA_PATH}/bin:${PATH}
export PATH=${MACA_PATH}/mxgpu_llvm/bin:${MACA_PATH}/bin:${CUCC_PATH}/tools:${CUCC_PATH}/bin:${PATH}

export ISU_FASTMODEL=1  # must be set, otherwise may induce precision error
export USE_TDUMP=OFF    # optional, use to control whether generating debug file
export TMEM_LOG=OFF     # optional, use to control whether generating debug file
export DEBUG_ITRACE=0   # optional, use to control whether generating debug file

