#include "ppl/nn/engines/cuda/optimizer/ops/pmx/batch_gemm_op.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/pmx/batch_gemm_kernel.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::pmx;


namespace ppl { namespace nn { namespace cuda {

RetCode BatchGemmOp::Init(const OptKernelOptions& options) {
    // auto status = GenericLoadParam<LayerNormParam>(options, &param_);
    // if (status != RC_SUCCESS) {
    //     LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
    //     return status;
    // }
    // status = Finalize(options);
    return RC_SUCCESS;
}

BatchGemmOp::BatchGemmOp(const ir::Node* node) : CudaOptKernel(node) {

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
        auto input0 = info->GetInput<TensorImpl>(0);
        auto input0_shape = input0->GetShape();

        auto input1 = info->GetInput<TensorImpl>(1);
        auto input1_shape = input1->GetShape();

        std::vector<int64_t> output_shape;
        int64_t m = input0_shape->GetDim(0);
        if(input0_shape->GetDimCount()==3){
            m *= input0_shape->GetDim(1);
        }
        if(output_shape_.empty()){
            output_shape = {m , input1_shape->GetDim(1)};
        }else{
            int64_t total_data_len = m * input1_shape->GetDim(2);
            output_shape = output_shape_;
            if(dynamic_axis_ > -1){
                output_shape[dynamic_axis_] = total_data_len / kown_shape_len_;
            }
        }
        for(size_t i=0; i<info->GetOutputCount(); i++){
            auto output = info->GetOutput<TensorImpl>(i);
            output->GetShape()->Reshape(output_shape);
        }

        return RC_SUCCESS;
    };

}
RetCode BatchGemmOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* BatchGemmOp::CreateKernelImpl() const {

    return CreateKernelImplWithoutParam<BatchGemmKernel>();
}

}}} // namespace ppl::nn::cuda
