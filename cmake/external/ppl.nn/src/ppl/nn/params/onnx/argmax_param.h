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

#ifndef _ST_HPC_PPL_NN_PARAMS_ONNX_ARGMAX_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_ONNX_ARGMAX_PARAM_H_

#include <stdint.h>
#include "ppl/nn/ir/attr.h"

namespace ppl { namespace nn { namespace onnx {

struct ArgMaxParam final : public ir::TypedAttr<ArgMaxParam> {
    int32_t axis;
    int32_t keepdims;
    int32_t select_last_index;

    bool operator==(const ArgMaxParam& p) const {
        return this->axis == p.axis && this->keepdims == p.keepdims && this->select_last_index == p.select_last_index;
    }
};

}}} // namespace ppl::nn::onnx

#endif
