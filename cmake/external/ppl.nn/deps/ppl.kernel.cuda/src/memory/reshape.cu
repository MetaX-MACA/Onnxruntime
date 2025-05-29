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

#include "cudakernel/memory/reshape.h"
#include "ppl/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
template<typename T>
__global__ void reshape_nhwc(const void* input, void* output,int input_channel,int input_pad_channel, int output_channel, int output_pad_channel, int64_t num_elems_output) {
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= num_elems_output) return;
    int64_t out_c = index % output_pad_channel;
    int64_t out_h = index / output_pad_channel;
    T * ptr_output = (T*)output;
    T* ptr_input = (T*)input;
    if(out_c < output_channel) {
        int64_t in_valid_index = out_h * output_channel + out_c;
        int in_channel = in_valid_index % input_channel;
        int in_h = in_valid_index / input_channel;
        ptr_output[index] = *(ptr_input + in_h * input_pad_channel + in_channel);
        
    } else {
        ptr_output[index] = (T)0;
    }
}

template<typename T>
void check_result(int channel,int pad_channel,int num_elems, void *buffer) {
    T * cpu_buffer = (T*)malloc(sizeof(T)*num_elems);
    cudaMemcpy(cpu_buffer, buffer,sizeof(T)*num_elems, cudaMemcpyDeviceToHost);
    for(int i = 0; i < num_elems; i++) {
        int c = i % pad_channel;
        int h = i / pad_channel;
        if(c < channel) {
            printf("%f,",(float)cpu_buffer[h*pad_channel + c]);
        }
        if((i % pad_channel) == 0) printf("\n");
    }
    free(cpu_buffer);
}

ppl::common::RetCode PPLCUDAReshapeForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output)
{
    if(output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 || output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16 
        || output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC4) {
        if(input_shape->GetDimCount() >= 2 && output_shape->GetDimCount() >= 2 && (input_shape->GetDim(0) != output_shape->GetDim(0) || input_shape->GetDim(1) != output_shape->GetDim(1))) {
            int input_channel = input_shape->GetDim(1);
            int output_channel = output_shape->GetDim(1);
            int input_pad_channel = input_shape->GetDim(1) + input_shape->GetPadding0(1) + input_shape->GetPadding1(1);
            int output_pad_channel = output_shape->GetDim(1) + output_shape->GetPadding0(1) + output_shape->GetPadding1(1);
            int64_t num_elems_output = output_shape->CalcElementsIncludingPadding();
            int blocksize = 512;
            int64_t gridSize = (num_elems_output + blocksize - 1) / blocksize;
            switch(ppl::common::GetSizeOfDataType(input_shape->GetDataType())) {
                case 1:
                    reshape_nhwc<int8_t><<<gridSize, blocksize,0,stream>>>(input,output,input_channel,input_pad_channel,output_channel,output_pad_channel,num_elems_output);
                break;
                case 2:
                    reshape_nhwc<int16_t><<<gridSize, blocksize,0,stream>>>(input,output,input_channel,input_pad_channel,output_channel,output_pad_channel,num_elems_output);
                break;
                case 4:
                    reshape_nhwc<int32_t><<<gridSize, blocksize,0,stream>>>(input,output,input_channel,input_pad_channel,output_channel,output_pad_channel,num_elems_output);
                break;
                case 8:
                    reshape_nhwc<int64_t><<<gridSize, blocksize,0,stream>>>(input,output,input_channel,input_pad_channel,output_channel,output_pad_channel,num_elems_output);
                break;
                default:
                return ppl::common::RC_UNSUPPORTED;
            }
            return ppl::common::RC_SUCCESS;
        }
    }
    int64_t num_elems_output = output_shape->CalcElementsIncludingPadding();
    cudaMemcpyAsync(output, input, ppl::common::GetSizeOfDataType(input_shape->GetDataType()) * num_elems_output, cudaMemcpyDeviceToDevice, stream);
    return ppl::common::RC_SUCCESS;
}
