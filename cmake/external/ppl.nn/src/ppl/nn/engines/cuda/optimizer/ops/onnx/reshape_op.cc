// 2024 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "ppl/nn/engines/cuda/optimizer/ops/onnx/reshape_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/kernels/onnx/reshape_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_reshape.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

RetCode ReshapeOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    auto node = GetNode();
    auto graph_data = options.graph->data.get();
    auto preedge_id = node->GetInput(1);
    auto constant_ref = graph_data->constants.find(preedge_id);
    auto preshape_ref = graph_data->shapes.find(preedge_id);
    if (constant_ref != graph_data->constants.end()) {
        auto shape_buffer = static_cast<int64_t*>(constant_ref->second.data.GetData());
        if (preshape_ref != graph_data->shapes.end()) {
            int64_t output_shape = preshape_ref->second.dims[0];
            for (int i = 0; i < output_shape; i++) {
                shape_data_.push_back(shape_buffer[i]);
            }
            is_constant_shape_ = true;
        }
    }
    return RC_SUCCESS;
}

ReshapeOp::ReshapeOp(const ir::Node* node) : CudaOptKernel(node) {
    infer_type_func_ = [](InputOutputInfo* info, std::vector<CudaTensorQuant>* quant, datatype_t type) -> RetCode {
        ppl::common::RetCode status;
        if (type == DATATYPE_UNKNOWN) {
            status = InferInheritedType(info);
        } else if (type == DATATYPE_INT8) {
            status = UnifyToOutputQuant(info, quant);
        } else {
            status = InferDefaultType(info, type);
        }
        auto shape = info->GetInput<TensorImpl>(1)->GetShape();
        shape->SetDataType(DATATYPE_INT64);
        return status;
    };

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        if (info->GetInputCount() != 2) {
            LOG(ERROR) << "2 input required.";
            return RC_INVALID_VALUE;
        }

        auto input = info->GetInput<TensorImpl>(1);
        if (!input->GetBufferPtr()) {
            return RC_NOT_FOUND;
        }

        const TensorShape& dst_desc = *input->GetShape();
        vector<int64_t> shape_data(dst_desc.CalcElementsIncludingPadding());
        if (is_constant_shape_) {
            for (size_t i = 0; i < shape_data_.size(); i++) {
                shape_data[i] = shape_data_[i];
            }
        } else {
            auto status = input->CopyToHost(shape_data.data());
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "Copy shape data failed: " << GetRetCodeStr(status);
                return status;
            }
        }
        return onnx::ReshapeReshape(info, &param_, shape_data.data());
    };

    infer_unsafe_dims_func_ = [this](InputOutputInfo* info, std::set<uint32_t>* mask) -> RetCode {
        if (info->GetInputCount() != 2) {
            LOG(ERROR) << "2 input required.";
            return RC_INVALID_VALUE;
        }

        auto input1 = info->GetInput<TensorImpl>(1);
        const TensorShape& shape1 = *input1->GetShape();
        if (shape1.GetRealDimCount() == 1 && shape1.GetDim(0) > 0) {
            ppl::nn::onnx::ReshapeParam param;
            param.allowzero = 0;

            auto input0 = info->GetInput<TensorImpl>(0);
            const TensorShape& shape0 = *input0->GetShape();
            int dim_count0 = shape0.GetDimCount();
            int dst_dim_count = shape1.GetDim(0);

            vector<int64_t> shape_data(dst_dim_count);
            if (!input1->GetBufferPtr()) {
                for (int i = 0; i < dst_dim_count; i++) {
                    if (i == (dst_dim_count - 1)) {
                        shape_data[i] = -1;
                    } else if (i < dim_count0) {
                        int64_t dim = shape0.GetDim(i);
                        shape_data[i] = (dim < 1) ? 1 : dim;
                    } else {
                        shape_data[i] = 1;
                    }
                }
            } else {
                // Should delete
                if (is_constant_shape_) {
                    for (size_t i = 0; i < shape_data_.size(); i++) {
                        shape_data[i] = shape_data_[i];
                    }
                } else {
                    auto status = input1->CopyToHost(shape_data.data());
                    if (status != RC_SUCCESS) {
                        LOG(ERROR) << "Copy shape data failed: " << GetRetCodeStr(status);
                        return status;
                    }
                }
            }

            return onnx::ReshapeReshape(info, &param, shape_data.data());
        } else {
            const TensorShape& in_shape0 = *info->GetInput<TensorImpl>(0)->GetShape();
            for (uint32_t i = 0; i < info->GetOutputCount(); ++i) {
                info->GetOutput<TensorImpl>(i)->GetShape()->Reshape(in_shape0.GetDims(), in_shape0.GetRealDimCount());
            }
            return ppl::common::RC_SUCCESS;
        }
    };
}

RetCode ReshapeOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* ReshapeOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<ReshapeKernel>();
}

}}} // namespace ppl::nn::cuda
