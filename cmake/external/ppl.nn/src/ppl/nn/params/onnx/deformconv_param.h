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

#ifndef _ST_HPC_PPL_NN_PARAMS_ONNX_DEFORMCONV_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_ONNX_DEFORMCONV_PARAM_H_

#include "ppl/nn/ir/attr.h"
#include <stdint.h>
#include <vector>

namespace ppl { namespace nn { namespace onnx {

struct DeformConvParam final : public ir::TypedAttr<DeformConvParam> {
    std::vector<int64_t> kernel_shape;
    std::vector<int64_t> dilations;
    std::vector<int64_t> strides;
    std::vector<int64_t> pads;
    int64_t groups;
    int64_t deform_groups;

    bool operator==(const DeformConvParam& p) const {
        return ( groups == p.groups && deform_groups == p.deform_groups && kernel_shape == p.kernel_shape &&
                dilations == p.dilations && strides == p.strides && pads == p.pads);
    }
};

}}} // namespace ppl::nn::onnx

#endif
