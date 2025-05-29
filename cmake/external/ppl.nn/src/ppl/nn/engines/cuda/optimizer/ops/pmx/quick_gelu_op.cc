#include "ppl/nn/engines/cuda/optimizer/ops/pmx/quick_gelu_op.h"
#include "ppl/nn/engines/cuda/kernels/pmx/quick_gelu_kernel.h"
#include "ppl/nn/common/logger.h"


using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {
RetCode QuickGeluOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam<ppl::nn::pmx::QuickGeluParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }
    return RC_SUCCESS;
}

QuickGeluOp::QuickGeluOp(const ir::Node* node) : CudaOptKernel(node) {
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

    infer_dims_func_ = GenericInferDims;
}


RetCode QuickGeluOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* QuickGeluOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<QuickGeluKernel>(&param_);
}


}}} // namespace ppl::nn::cuda