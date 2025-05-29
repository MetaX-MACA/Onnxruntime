#include "ppl/nn/engines/cuda/kernels/pmx/fused_gathernd_reshape_concat_kernel.h"


namespace ppl { namespace nn { namespace cuda {


ppl::common::RetCode FusedGatherNDReshapeConcatKernel::DoExecute(KernelExecContext* ctx) {

    auto GatherND_data_input = ctx->GetInput<TensorImpl>(0);
    auto GatherND_data_input_shape = GatherND_data_input->GetShape();

    auto GatherND_indices_input = ctx->GetInput<TensorImpl>(1);
    auto GatherND_indices_input_shape = GatherND_indices_input->GetShape();

    auto Reshape_shape_input = ctx->GetInput<TensorImpl>(2);
    auto Reshape_shape_input_shape = Reshape_shape_input->GetShape();

    auto Concat_input = ctx->GetInput<TensorImpl>(3);
    auto Concat_input_shape = Concat_input->GetShape();

    auto output = ctx->GetOutput<TensorImpl>(0);
    auto output_shape = output->GetShape();

    int gather_nd_batch_dims = param_->gathernd_param.batch_dims;
    int concat_axis = param_->concat_param.axis;

    LOG(DEBUG) << "Run FusedGatherNDReshapeConcatKernel with datatype " << GatherND_data_input_shape->GetDataType() << " dataformat " << GatherND_data_input_shape->GetDataFormat();
    //auto status = PPLCUDAFusedGatherNDReshapeConcatForwardImp(GetStream(), .....,output->GetBufferPtr(), param_->axis);
    return 0;
}

}}} // namespace ppl::nn::cuda
