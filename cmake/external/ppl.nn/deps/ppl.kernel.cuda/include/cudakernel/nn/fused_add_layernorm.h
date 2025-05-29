#ifndef PPL_CUDA_KERNEL_INCLUDE_FUSED_ADD_LAYERNORM_H_
#define PPL_CUDA_KERNEL_INCLUDE_FUSED_ADD_LAYERNORM_H_
#include "ppl/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_runtime.h>

ppl::common::RetCode PPLCUDAFusedAddLayerNormForwardImp(
    cudaStream_t stream,
    ppl::common::TensorShape* input_shape_0,
    ppl::common::TensorShape* input_shape_1,
    const void* input_0,
    const void* input_1,
    const void* scale,
    const void* shift,
    void* output,
    ppl::common::TensorShape* output_shape,
    int outer,
    int inner,
    bool elementwise_affine,
    float epsilon,
    float in_scale_0,
    float in_scale_1,
    float out_scale);

ppl::common::RetCode PPLCUDAFusedAddLayerNormForwardImp(
    cudaStream_t stream,
    ppl::common::TensorShape* input_shape_0,
    ppl::common::TensorShape* input_shape_1,
    const void* input_0,
    const void* input_1,
    const void* scale,
    const void* shift,
    void* output_0,
    void* output_1,
    ppl::common::TensorShape* output_shape,
    int outer,
    int inner,
    bool elementwise_affine,
    float eps,
    float in_scale_0,
    float in_scale_1,
    float out_scale,
    float out_scale_add);
#endif
