#include "ppl/nn/engines/cuda/kernels/pmx/fused_add_layernorm_kernel.h"
#include "cudakernel/nn/fused_add_layernorm.h"

namespace ppl { namespace nn { namespace cuda {


ppl::common::RetCode FusedAddLayerNormKernel::DoExecute(KernelExecContext* ctx) {

    auto input0 = ctx->GetInput<TensorImpl>(0);
    auto in_shape0 = input0->GetShape();

    auto input1 = ctx->GetInput<TensorImpl>(1);
    auto in_shape1 = input1->GetShape();

    auto output_layernorm = ctx->GetOutput<TensorImpl>(0); //add + layernorm output
    if(ctx->GetOutputCount() == 2){
        output_layernorm = ctx->GetOutput<TensorImpl>(1);
    }
    auto output_layernorm_shape = output_layernorm->GetShape();

    void* scale_ptr = nullptr;
    void* shift_ptr = nullptr;
    if (param_->elementwise_affine) {
        scale_ptr = ctx->GetInput<TensorImpl>(2)->GetBufferPtr();
        shift_ptr = ctx->GetInput<TensorImpl>(3)->GetBufferPtr();
    }

    int axis = param_->axis;
    int outer = 1;
    int inner = 1;

    if (axis < 0) {
        axis += in_shape0->GetDimCount();
    }

    for (int32_t i = 0; i < axis; ++i) {
        outer *= output_layernorm_shape->GetDim(i);
    }
    for (uint32_t i = axis; i < output_layernorm_shape->GetDimCount(); ++i) {
        inner *= output_layernorm_shape->GetDim(i);
    }

    auto input_id0 = input0->GetEdge()->GetId();
    auto input_quant0 = GetCommonParam()->cuda_tensor_info->at(input_id0);

    auto input_id1 = input1->GetEdge()->GetId();
    auto input_quant1 = GetCommonParam()->cuda_tensor_info->at(input_id1);

    auto output_layernorm_id = output_layernorm->GetEdge()->GetId();
    auto output_layernorm_quant = GetCommonParam()->cuda_tensor_info->at(output_layernorm_id);

    LOG(DEBUG) << "Run FusedAddLayerNormKernel with datatype " << in_shape0->GetDataType() << " dataformat " << in_shape0->GetDataFormat();
    ppl::common::RetCode status = 0;
    if(ctx->GetOutputCount() == 1){
        status = PPLCUDAFusedAddLayerNormForwardImp(GetStream(), in_shape0, in_shape1, input0->GetBufferPtr(),input1->GetBufferPtr(), scale_ptr, shift_ptr,
                            output_layernorm->GetBufferPtr(), output_layernorm_shape, outer, inner, param_->elementwise_affine, param_->eps, input_quant0.scale[0], input_quant1.scale[0], 1.0f / output_layernorm_quant.scale[0]);
    }else{
        auto output_add = ctx->GetOutput<TensorImpl>(0); //two shape
        auto output_add_id = output_add->GetEdge()->GetId();
        auto output_add_quant = 1.0f / GetCommonParam()->cuda_tensor_info->at(output_add_id).scale[0];
        status = PPLCUDAFusedAddLayerNormForwardImp(GetStream(), in_shape0, in_shape1, input0->GetBufferPtr(), input1->GetBufferPtr(), scale_ptr, shift_ptr, output_layernorm->GetBufferPtr(), output_add->GetBufferPtr(),
            output_layernorm_shape, outer, inner, param_->elementwise_affine, param_->eps, input_quant0.scale[0], input_quant1.scale[0], 1.0f / output_layernorm_quant.scale[0], output_add_quant);
    }
    return status;
}

}}} // namespace ppl::nn::cuda
