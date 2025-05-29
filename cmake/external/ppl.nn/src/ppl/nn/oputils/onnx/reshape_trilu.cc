// 2024 - Created by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
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

#include "ppl/nn/oputils/onnx/reshape_trilu.h"
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx {

RetCode ReshapeTrilu(InputOutputInfo* info, const ir::Attr* arg) {
    auto input_shape = info->GetInput<TensorImpl>(0)->GetShape();
    info->GetOutput<TensorImpl>(0)->GetShape()->Reshape(input_shape->GetDims(), input_shape->GetDimCount());

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
