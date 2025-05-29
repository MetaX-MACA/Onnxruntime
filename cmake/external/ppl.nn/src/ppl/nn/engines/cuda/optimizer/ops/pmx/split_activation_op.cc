// 2024 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#include "ppl/nn/engines/cuda/optimizer/ops/pmx/split_activation_op.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/pmx/split_activation_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_split_activation.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::pmx;

namespace ppl { namespace nn { namespace cuda {
RetCode SplitActivationOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam<SplitActivationParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }
    return RC_SUCCESS;
}

SplitActivationOp::SplitActivationOp(const ir::Node* node) : CudaOptKernel(node) {
    infer_type_func_ = [](InputOutputInfo* info, std::vector<CudaTensorQuant>* quant, datatype_t type) -> RetCode {
        ppl::common::RetCode status;
        if (type == DATATYPE_UNKNOWN) {
            status = InferInheritedType(info);
        } else if (type == DATATYPE_INT8) {
            status = CopyQuantType(info, quant);
        } else {
            status = InferDefaultType(info, type);
        }
        return status;
    };

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return ppl::nn::onnx::ReshapeSplitActivation(info, &param_);
    };
}

RetCode SplitActivationOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* SplitActivationOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<SplitActivationKernel>(&param_);
}

}}} // namespace ppl::nn::cuda
