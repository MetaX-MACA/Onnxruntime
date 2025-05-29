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

#include "ppl/nn/engines/cuda/kernels/onnx/bitwise_kernel.h"
#include "cudakernel/arithmetic/bitwise.h"

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode BitwiseAndKernel::DoExecute(KernelExecContext* ctx) {
    auto input0 = ctx->GetInput<TensorImpl>(0);
    auto input_id0 = input0->GetEdge()->GetId();
    auto input_quant0 = GetCommonParam()->cuda_tensor_info->at(input_id0);
    
    auto input1 = ctx->GetInput<TensorImpl>(1);
    auto input_id1 = input1->GetEdge()->GetId();
    auto input_quant1 = GetCommonParam()->cuda_tensor_info->at(input_id1);
    
    auto output = ctx->GetOutput<TensorImpl>(0);

    auto output_id = output->GetEdge()->GetId();
    auto output_quant = GetCommonParam()->cuda_tensor_info->at(output_id);
    ppl::common::RetCode status = 
        PPLCUDABitWiseAndForwardImp(GetStream(),input0->GetShape(),input0->GetBufferPtr(),
            input1->GetShape(),input1->GetBufferPtr(),output->GetShape(),output->GetBufferPtr());
    return status;
}

ppl::common::RetCode BitwiseNotKernel::DoExecute(KernelExecContext* ctx) {
    auto input0 = ctx->GetInput<TensorImpl>(0);
    auto input_id0 = input0->GetEdge()->GetId();
    auto input_quant0 = GetCommonParam()->cuda_tensor_info->at(input_id0);
    
    auto output = ctx->GetOutput<TensorImpl>(0);

    auto output_id = output->GetEdge()->GetId();
    auto output_quant = GetCommonParam()->cuda_tensor_info->at(output_id);
    ppl::common::RetCode status = 
        PPLCUDABitWiseNotForwardImp(GetStream(),input0->GetShape(),input0->GetBufferPtr(),
            output->GetShape(),output->GetBufferPtr());
    return status;
}

ppl::common::RetCode BitwiseOrKernel::DoExecute(KernelExecContext* ctx) {
    auto input0 = ctx->GetInput<TensorImpl>(0);
    auto input_id0 = input0->GetEdge()->GetId();
    auto input_quant0 = GetCommonParam()->cuda_tensor_info->at(input_id0);
    
    auto input1 = ctx->GetInput<TensorImpl>(1);
    auto input_id1 = input1->GetEdge()->GetId();
    auto input_quant1 = GetCommonParam()->cuda_tensor_info->at(input_id1);
    
    auto output = ctx->GetOutput<TensorImpl>(0);

    auto output_id = output->GetEdge()->GetId();
    auto output_quant = GetCommonParam()->cuda_tensor_info->at(output_id);
    ppl::common::RetCode status = 
        PPLCUDABitWiseOrForwardImp(GetStream(),input0->GetShape(),input0->GetBufferPtr(),
            input1->GetShape(),input1->GetBufferPtr(),output->GetShape(),output->GetBufferPtr());
    return status;
}

ppl::common::RetCode BitwiseXorKernel::DoExecute(KernelExecContext* ctx) {
    auto input0 = ctx->GetInput<TensorImpl>(0);
    auto input_id0 = input0->GetEdge()->GetId();
    auto input_quant0 = GetCommonParam()->cuda_tensor_info->at(input_id0);
    
    auto input1 = ctx->GetInput<TensorImpl>(1);
    auto input_id1 = input1->GetEdge()->GetId();
    auto input_quant1 = GetCommonParam()->cuda_tensor_info->at(input_id1);
    
    auto output = ctx->GetOutput<TensorImpl>(0);

    auto output_id = output->GetEdge()->GetId();
    auto output_quant = GetCommonParam()->cuda_tensor_info->at(output_id);
    ppl::common::RetCode status = 
        PPLCUDABitWiseXorForwardImp(GetStream(),input0->GetShape(),input0->GetBufferPtr(),
            input1->GetShape(),input1->GetBufferPtr(),output->GetShape(),output->GetBufferPtr());
    return status;
}

ppl::common::RetCode BitShiftKernel::DoExecute(KernelExecContext* ctx) {
    auto input0 = ctx->GetInput<TensorImpl>(0);
    auto input_id0 = input0->GetEdge()->GetId();
    auto input_quant0 = GetCommonParam()->cuda_tensor_info->at(input_id0);
    
    auto input1 = ctx->GetInput<TensorImpl>(1);
    auto input_id1 = input1->GetEdge()->GetId();
    auto input_quant1 = GetCommonParam()->cuda_tensor_info->at(input_id1);
    
    auto output = ctx->GetOutput<TensorImpl>(0);

    auto output_id = output->GetEdge()->GetId();
    auto output_quant = GetCommonParam()->cuda_tensor_info->at(output_id);
     ppl::common::RetCode status = ppl::common::RC_SUCCESS;
    if(param_->direction == "LEFT"){  
        status = PPLCUDABitShiftForwardImp(GetStream(),input0->GetShape(),input0->GetBufferPtr(),
                input1->GetShape(),input1->GetBufferPtr(),output->GetShape(),output->GetBufferPtr(),0);
    }else if(param_->direction == "RIGHT"){
        status = PPLCUDABitShiftForwardImp(GetStream(),input0->GetShape(),input0->GetBufferPtr(),
                input1->GetShape(),input1->GetBufferPtr(),output->GetShape(),output->GetBufferPtr(),1);
    }else{
        status = ppl::common::RC_UNSUPPORTED;
    }
 
    return status;
}

}
}
}