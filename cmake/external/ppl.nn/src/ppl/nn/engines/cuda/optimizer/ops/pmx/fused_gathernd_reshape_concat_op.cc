#include "ppl/nn/engines/cuda/optimizer/ops/pmx/fused_gathernd_reshape_concat_op.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/pmx/fused_gathernd_reshape_concat_kernel.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::pmx;


namespace ppl { namespace nn { namespace cuda {

RetCode FusedGatherNDReshapeConcatOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam<FusedGatherNDReshapeConcatParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }
    status = Finalize(options);
    return RC_SUCCESS;
}

FusedGatherNDReshapeConcatOp::FusedGatherNDReshapeConcatOp(const ir::Node* node) : CudaOptKernel(node) {

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

        // get gathernd output element count
        auto input_data = info->GetInput<TensorImpl>(0)->GetShape();
        auto input_indices = info->GetInput<TensorImpl>(1)->GetShape();

        const uint32_t r = input_data->GetRealDimCount();
        const uint32_t q = input_indices->GetRealDimCount();
        if (r < 1 || q < 1) {
            LOG(DEBUG) << "ERROR: input[0]'s dim count[" << r << "] < 1 or input[1]'s dim count[" << q << "] < 1.";
            return RC_INVALID_VALUE;
        }
        const uint32_t last_indices_dim = input_indices->GetDim(q - 1);
        if (last_indices_dim < 1 || last_indices_dim > (r - param_.gathernd_param.batch_dims)) {
            LOG(DEBUG) << "ERROR: last index dim[" << last_indices_dim << "] is out of range[1, " << (r - param_.gathernd_param.batch_dims) << "].";
            return RC_INVALID_VALUE;
        }
        const uint32_t gathernd_output_dim_count = q + r - last_indices_dim - 1 - param_.gathernd_param.batch_dims;
        int64_t gathernd_output_element_count = 1;

        for (size_t i = 0; i < gathernd_output_dim_count; i++) {
            if(i < q-1){
                gathernd_output_element_count *= input_indices->GetDim(i);
            }else{
                gathernd_output_element_count *= input_data->GetDim(i - (q - 1) + last_indices_dim + param_.gathernd_param.batch_dims);
            }
        }

        // get reshape output shape
        auto shape_input = info->GetInput<TensorImpl>(2);
        if (!shape_input->GetBufferPtr()) {
            return RC_NOT_FOUND;
        }

        const TensorShape& dst_desc = *shape_input->GetShape();
        vector<int64_t> reshape_output_shape(dst_desc.CalcElementsIncludingPadding());

        auto status = shape_input->CopyToHost(reshape_output_shape.data());
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "Copy shape data failed: " << GetRetCodeStr(status);
            return status;
        }

        // get concat output shape
        int unkown_axis = -1;
        int kown_axis_ele_count = 1;
        for(size_t i=0; i<reshape_output_shape.size(); i++){
            if(reshape_output_shape[i]<1){
                if(unkown_axis == -1){
                    unkown_axis = i;
                }else{
                    LOG(ERROR) << "shape data not support unkown axis>=2";
                    return RC_INVALID_VALUE;
                }
            }else{
                kown_axis_ele_count *= reshape_output_shape[i];
            }
        }

        if(unkown_axis != -1 ){
            reshape_output_shape[unkown_axis] = gathernd_output_element_count / kown_axis_ele_count;
            kown_axis_ele_count *= reshape_output_shape[unkown_axis];
        }

        if(kown_axis_ele_count != gathernd_output_element_count){
            LOG(ERROR) << "shape data not match input elements : "<< kown_axis_ele_count << " vs " << gathernd_output_element_count;
            return RC_INVALID_VALUE;
        }

        // get output shape
        auto concat_input = info->GetInput<TensorImpl>(3);
        const TensorShape& concat_input_shape = *concat_input->GetShape();

        std::vector output_shape = reshape_output_shape;
        if(reshape_output_shape.size() != concat_input_shape.GetDimCount()){
            LOG(ERROR) << "concat op input dim count not match";
            return RC_INVALID_VALUE;
        }

        for(size_t i =0; i<reshape_output_shape.size(); i++){
            if(i == param_.concat_param.axis){
                output_shape[i] = reshape_output_shape[i] + concat_input_shape.GetDim(i);
            }else if(reshape_output_shape[i] !=  concat_input_shape.GetDim(i)){
                LOG(ERROR) << "concat op input dim not match : " << reshape_output_shape[i] <<  " vs " << concat_input_shape.GetDim(i);
                return RC_INVALID_VALUE;
            }
        }

        auto output = info->GetOutput<TensorImpl>(0);
        output->GetShape()->Reshape(output_shape.data(), output_shape.size());

        return RC_SUCCESS;

    };

}
RetCode FusedGatherNDReshapeConcatOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* FusedGatherNDReshapeConcatOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<FusedGatherNDReshapeConcatKernel>(&param_);
}

}}} // namespace ppl::nn::cuda
