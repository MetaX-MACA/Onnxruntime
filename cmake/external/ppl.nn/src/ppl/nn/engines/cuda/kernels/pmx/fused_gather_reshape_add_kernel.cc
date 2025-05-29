#include "ppl/nn/engines/cuda/kernels/pmx/fused_gather_reshape_add_kernel.h"
#include "cudakernel/memory/gather_reshape_add.h"

namespace ppl { namespace nn { namespace cuda {


ppl::common::RetCode FusedGatherReshapeAddKernel::DoExecute(KernelExecContext* ctx) {
    auto input_data = ctx->GetInput<TensorImpl>(0);
    auto input_data_shape = input_data->GetShape();

    auto input_indices = ctx->GetInput<TensorImpl>(1);
    auto input_indices_shape = input_indices->GetShape();

    // auto input_shape = ctx->GetInput<TensorImpl>(2);

    auto input_add = ctx->GetInput<TensorImpl>(3);
    auto input_add_shape = input_add->GetShape();

    auto output = ctx->GetOutput<TensorImpl>(0);
    auto output_shape = output->GetShape();

    int axis = param_->axis;

    ppl::common::RetCode status = PPLCUDAGatResAddForwardImp(GetStream(), input_data_shape, input_data->GetBufferPtr(), input_indices_shape, input_indices->GetBufferPtr(),
                                                                input_add_shape, input_add->GetBufferPtr(), output_shape, output->GetBufferPtr(), axis);

    return status;
}

}}} // namespace ppl::nn::cuda
