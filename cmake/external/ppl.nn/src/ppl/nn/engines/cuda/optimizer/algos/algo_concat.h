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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_ALGOS_ALGO_CONCAT_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_ALGOS_ALGO_CONCAT_H_

#include <stdlib.h>
#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/optimizer/algos/algorithm.h"
extern bool gSupportNc1hwc0;
using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

class ConcatAlgorithm : public Algorithm {
public:
    ConcatAlgorithm() {
        std::set<dataformat_t> ndarray{DATAFORMAT_NDARRAY};
        concat_formats_.emplace(DATAFORMAT_NDARRAY, ndarray);
    #ifdef PPLNN_USE_DNN
        std::set<dataformat_t> nhwc{DATAFORMAT_NHWC};
        concat_formats_.emplace(DATAFORMAT_NHWC, nhwc);
        std::set<dataformat_t> nchw8{DATAFORMAT_NCHW8};
        concat_formats_.emplace(DATAFORMAT_NCHW8, nchw8);
        std::set<dataformat_t> nchw16{DATAFORMAT_NCHW16};
        concat_formats_.emplace(DATAFORMAT_NCHW16, nchw16);
    #endif
        std::set<dataformat_t> nhwc8{DATAFORMAT_NHWC8};
        concat_formats_.emplace(DATAFORMAT_NHWC8, nhwc8);
        std::set<dataformat_t> nhwc16{DATAFORMAT_NHWC16};
        concat_formats_.emplace(DATAFORMAT_NHWC16, nhwc16);
    }

    void GetAttrParam(void*& param) const override {
        return;
    };
    void DeleteAttrParam(void*& param) override {
        return;
    };

    bool IsSupported(const ir::Node* node, const OptKernelOptions& options, dataformat_t input_format) const override{
        char *env_ptr = getenv("MACART_OP_FORMAT");
        if(env_ptr != nullptr){
            string op_format_str(env_ptr);
            if(op_format_str == "ND" && input_format == DATAFORMAT_NDARRAY){
                return true;
            }else if(op_format_str == "NHWC8" && input_format == DATAFORMAT_NHWC8){
                return true;
            }
            else if(op_format_str == "NHWC16" && input_format == DATAFORMAT_NHWC16){
                return true;
            }
            else if(op_format_str == "NHWC" && input_format == DATAFORMAT_NHWC){
                return true;
            }
            else if(op_format_str == "NCHW8" && input_format == DATAFORMAT_NCHW8){
                return true;
            }
            else if(op_format_str == "NCHW16" && input_format == DATAFORMAT_NCHW16){
                return true;
            }
            if(!isFormatSupport(op_format_str))
                LOG(WARNING) << "MACART_OP_FORMAT only support ND/NHWC/NHWC8/NHWC16/NCHW8/NCHW16, but set:" << env_ptr;
            return false;
        }
        else{
            return true;
        }
    }

    const std::map<dataformat_t, std::set<dataformat_t>> Getformats(const std::string& type_name) const override {
    #ifdef PPLNN_USE_DNN
        if(gSupportNc1hwc0 == false){
            std::map<dataformat_t, std::set<dataformat_t>> concat_formats_tmp = concat_formats_;
            concat_formats_tmp.erase(DATAFORMAT_NCHW8);
            concat_formats_tmp.erase(DATAFORMAT_NCHW16);
            return concat_formats_tmp;
        }else{
            std::map<dataformat_t, std::set<dataformat_t>> concat_formats_tmp = concat_formats_;
            concat_formats_tmp.erase(DATAFORMAT_NHWC8);
            concat_formats_tmp.erase(DATAFORMAT_NHWC16);
            return concat_formats_tmp;
        }
    #endif
        return concat_formats_;
    }

    double ExcuteTimer(const ir::Node* node, OptKernelOptions& options) override;

    RetCode ModifyParam(ir::Node* node, OptKernelOptions& options) override {
        return RC_SUCCESS;
    }

    void ReshapeOnEdges(const ir::Node* node, std::map<edgeid_t, std::unique_ptr<TensorImpl>>* tensors,
                        ppl::common::dataformat_t input_format, ppl::common::dataformat_t output_format) override;

private:
    std::map<dataformat_t, std::set<dataformat_t>> concat_formats_;
};

}}} // namespace ppl::nn::cuda

#endif
