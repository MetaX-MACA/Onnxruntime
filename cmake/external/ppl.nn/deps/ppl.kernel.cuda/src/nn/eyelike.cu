// 2024 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#include "cudakernel/nn/eyelike.h"
#include "ppl/common/tensor_shape.h"
#include "cudakernel/common/common.h"
#include <stdint.h>
#include <cuda_fp16.h>
template <typename T,int TileM,int TileN>
__global__ void EyeLikeKernel(int k, int row,int col, T* output)
{
    int start_y = blockIdx.y*blockDim.y + threadIdx.y;
    int start_x = blockIdx.x*blockDim.x + threadIdx.x;
    int end_y = min(start_y + TileN,row);
    int end_x = min(start_x + TileM,col);
    for(int y = start_y; y < end_y; y++)
    {
        for(int x = start_x; x < end_x; x++)
        {
            output[y*col + x] = (x - y) == k;
        }
    }
} 

ppl::common::RetCode PPLCUDAEyeLikeForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output,
    int64_t k,
    ppl::common::datatype_t dtype,
    float in_scale,
    float out_scale)
{
    int out_h = input_shape->GetDim(0);
    int out_w  = input_shape->GetDim(1);
    const int tileM = 4;
    const int tileN = 4;
    dim3 threads(16, 16, 1);
    dim3 grid;
    grid.x = DivUp(out_w, 64);
    grid.y = DivUp(out_h, 64);
    grid.z = 1;
    
    if(dtype == ppl::common::DATATYPE_FLOAT32 || (dtype == ppl::common::DATATYPE_UNKNOWN && input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32)){
        EyeLikeKernel<float,tileM,tileN><<<grid, threads, 0, stream>>>(k,out_h,out_w,(float*)output);
    }else if(dtype == ppl::common::DATATYPE_UINT8 || (dtype == ppl::common::DATATYPE_UNKNOWN && input_shape->GetDataType() == ppl::common::DATATYPE_UINT8)){
        EyeLikeKernel<uint8_t,tileM,tileN><<<grid, threads, 0, stream>>>(k,out_h,out_w,(uint8_t*)output);
    }else if(dtype == ppl::common::DATATYPE_INT8 || (dtype == ppl::common::DATATYPE_UNKNOWN && input_shape->GetDataType() == ppl::common::DATATYPE_INT8)){
        EyeLikeKernel<int8_t,tileM,tileN><<<grid, threads, 0, stream>>>(k,out_h,out_w,(int8_t*)output);
    }else if(dtype == ppl::common::DATATYPE_UINT16 || (dtype == ppl::common::DATATYPE_UNKNOWN && input_shape->GetDataType() == ppl::common::DATATYPE_UINT16)){
        EyeLikeKernel<uint16_t,tileM,tileN><<<grid, threads, 0, stream>>>(k,out_h,out_w,(uint16_t*)output);
    }else if(dtype == ppl::common::DATATYPE_INT16 || (dtype == ppl::common::DATATYPE_UNKNOWN && input_shape->GetDataType() == ppl::common::DATATYPE_INT16)){
        EyeLikeKernel<int16_t,tileM,tileN><<<grid, threads, 0, stream>>>(k,out_h,out_w,(int16_t*)output);
    }else if(dtype == ppl::common::DATATYPE_INT32 || (dtype == ppl::common::DATATYPE_UNKNOWN && input_shape->GetDataType() == ppl::common::DATATYPE_INT32)){
        EyeLikeKernel<int32_t,tileM,tileN><<<grid, threads, 0, stream>>>(k,out_h,out_w,(int32_t*)output);
    }else if(dtype == ppl::common::DATATYPE_INT64 || (dtype == ppl::common::DATATYPE_UNKNOWN && input_shape->GetDataType() == ppl::common::DATATYPE_INT64)){
        EyeLikeKernel<long,tileM,tileN><<<grid, threads, 0, stream>>>(k,out_h,out_w,(long*)output);
    }else {
        EyeLikeKernel<float,tileM,tileN><<<grid, threads, 0, stream>>>(k,out_h,out_w,(float*)output);; 
    }
    return ppl::common::RC_SUCCESS;
}