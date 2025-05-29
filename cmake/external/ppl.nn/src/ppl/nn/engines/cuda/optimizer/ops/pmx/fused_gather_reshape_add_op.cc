#include "ppl/nn/engines/cuda/optimizer/ops/pmx/fused_gather_reshape_add_op.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/pmx/fused_gather_reshape_add_kernel.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::onnx;


namespace ppl { namespace nn { namespace cuda {

RetCode FusedGatherReshapeAddOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam<GatherParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }
    status = Finalize(options);
    return RC_SUCCESS;
}


FusedGatherReshapeAddOp::FusedGatherReshapeAddOp(const ir::Node* node) : CudaOptKernel(node) {

  infer_type_func_ = [](InputOutputInfo* info, std::vector<CudaTensorQuant>* quant, datatype_t type) -> RetCode {
        ppl::common::RetCode status;
        // type = ppl::common::DATATYPE_FLOAT16; // only support fp16 for now
        type = info->GetInput<TensorImpl>(0)->GetShape()->GetDataType();
        if (type == DATATYPE_UNKNOWN) {
            status = InferInheritedType(info);
        } else {
            status = InferDefaultType(info, type);
        }
        return status;
    };

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        auto input_data = info->GetInput<TensorImpl>(0);
        auto input_data_shape = input_data->GetShape();

        auto input_indices = info->GetInput<TensorImpl>(1);
        auto input_indices_shape = input_indices->GetShape();

        std::vector<int64_t> output_shape;
        int64_t m = input_indices_shape->GetDim(0);

        if(output_shape_.empty()){
            output_shape = {m , input_data_shape->GetDim(1)};
        }else{
            int64_t total_data_len = m * input_data_shape->GetDim(1);
            output_shape = output_shape_;
            output_shape[dynamic_axis_] = total_data_len / kown_shape_len_;
        }

        int dynamic_axis = 0;
        for(size_t i=0; i<info->GetOutputCount(); i++){
            auto output = info->GetOutput<TensorImpl>(i);
            output->GetShape()->Reshape(output_shape);
        }

        return RC_SUCCESS;
    };

}
RetCode FusedGatherReshapeAddOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* FusedGatherReshapeAddOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<FusedGatherReshapeAddKernel>(&param_);
}

}}} // namespace ppl::nn::cuda
