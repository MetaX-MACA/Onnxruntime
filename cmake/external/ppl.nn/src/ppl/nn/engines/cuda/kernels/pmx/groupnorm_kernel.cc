// 2024 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#include "ppl/nn/engines/cuda/kernels/pmx/groupnorm_kernel.h"
#include "cudakernel/nn/group_normalization.h"

namespace ppl { namespace nn { namespace cuda {


ppl::common::RetCode GroupNormKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto scale = ctx->GetInput<TensorImpl>(1);
    auto bias = ctx->GetInput<TensorImpl>(2);
    auto output = ctx->GetOutput<TensorImpl>(0);

    int64_t n_groups_i = param_->param.num_groups == -1 ? input->GetShape()->GetDim(2)/scale->GetShape()->GetDim(0) : param_->param.num_groups;
    int n_groups = static_cast<int>(n_groups_i);
    float epsilon = param_->param.epsilon;
    auto input_id0 = input->GetEdge()->GetId();
    float in_scale = GetCommonParam()->cuda_tensor_info->at(input_id0).scale[0];
    auto output_id0 = output->GetEdge()->GetId();
    float out_scale = GetCommonParam()->cuda_tensor_info->at(output_id0).scale[0];
    bool with_relu = param_->extra_param.has_relu;
    bool with_swish = param_->extra_param.has_swish;

    auto output_shape = output->GetShape();
    LOG(DEBUG) << "Run GroupNormKernel with datatype " << output_shape->GetDataType() << " dataformat " << output_shape->GetDataFormat();
    if (output_shape->GetDataFormat() != ppl::common::DATAFORMAT_NDARRAY) {
        LOG(ERROR) << "GroupNormKernel Just support NCHW. However DataFormat: " << output_shape->GetDataFormat();
        return ppl::common::RC_UNSUPPORTED;
    }

    if (output_shape->GetDataType() != ppl::common::DATATYPE_FLOAT32
        && output_shape->GetDataType() != ppl::common::DATATYPE_FLOAT16
        && output_shape->GetDataType() != ppl::common::DATATYPE_INT8) {
        LOG(ERROR) << "GroupNormKernel Just support INT8/FP16/FP32. However DataType: " << output_shape->GetDataType();
        return ppl::common::RC_UNSUPPORTED;
    }

    auto status = PPLCUDAGroupNormalizationForwardImp(GetStream(),
                                                    input->GetShape(),
                                                    input->GetBufferPtr(),
                                                    scale->GetShape(),
                                                    scale->GetBufferPtr(),
                                                    bias->GetBufferPtr(),
                                                    output->GetShape(),
                                                    output->GetBufferPtr(),
                                                    n_groups,
                                                    epsilon,
                                                    in_scale,
                                                    out_scale,
                                                    with_relu,
                                                    with_swish);
    return status;
}

}}} // namespace ppl::nn::cuda
