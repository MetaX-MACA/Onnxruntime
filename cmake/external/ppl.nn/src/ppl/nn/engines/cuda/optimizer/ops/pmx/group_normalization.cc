// 2024 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#include "ppl/nn/engines/cuda/optimizer/ops/pmx/group_normalization.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/pmx/groupnorm_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_group_normalization.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::pmx;

namespace ppl { namespace nn { namespace cuda {
RetCode GroupNormalizationOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam<ppl::nn::pmx::GroupNormalizationParam>(options, &param_.param);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }
    return RC_SUCCESS;
}

GroupNormalizationOp::GroupNormalizationOp(const ir::Node* node) : CudaOptKernel(node) {
    infer_type_func_ = [](InputOutputInfo* info, std::vector<CudaTensorQuant>* quant, datatype_t type) -> RetCode {
        ppl::common::RetCode status;
        if (type == DATATYPE_UNKNOWN) {
            //status = InferInheritedType(info);
            status = InferDefaultType(info, info->GetInput<TensorImpl>(0)->GetShape()->GetDataType());
        } else if (type == DATATYPE_INT8) {
            status = CopyQuantType(info, quant);
            info->GetInput<TensorImpl>(1)->GetShape()->SetDataType(ppl::common::DATATYPE_FLOAT32);
            info->GetInput<TensorImpl>(2)->GetShape()->SetDataType(ppl::common::DATATYPE_FLOAT32);
        } else {
            //status = InferDefaultType(info, type);
            status = InferDefaultType(info, info->GetInput<TensorImpl>(0)->GetShape()->GetDataType());
        }
        return status;
    };

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return ppl::nn::onnx::ReshapeGroupNormalization(info, &param_.param);
    };
}

RetCode GroupNormalizationOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* GroupNormalizationOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<GroupNormKernel>(&param_);
}

}}} // namespace ppl::nn::cuda
