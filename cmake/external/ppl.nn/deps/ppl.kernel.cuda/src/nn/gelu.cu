// 2024 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#include "cudakernel/nn/log.h"
#include "ppl/common/tensor_shape.h"
#include <stdint.h>
#include <cuda_fp16.h>


__global__ void ppl_cukernel_gelu_fp16(const size_t count, const half *input, half *output) {
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= count) return;
    float val = __half2float(input[index]);
    float out_val = val * 0.5 * (1 + erff(val * 0.707106781f));
    output[index] = __float2half_rn(out_val);
#endif
}



__global__ void ppl_cukernel_gelu_fp16_pack(const size_t count, const half2 *input, half2 *output) {
    #if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if(index >= count) return;
        half2 h_val = input[index];
        float2 f_val = __half22float2(h_val);

        half2 t_val;
        t_val.x = erff(f_val.x * 0.707106781f);
        t_val.y = erff(f_val.y * 0.707106781f);
        half2 one_constant = {1,1};
        half2 half_constant = {0.5,0.5};
        t_val = __hmul2(half_constant, __hmul2(h_val, __hadd2(one_constant, t_val)));

        output[index] = t_val;
    #endif
}

#ifdef PPLNN_USE_MACA
//Test of snop N (1000 times of kernel)
//snop 0 : 0.727~0.731ms
//snop 1 : 0.729~0.732ms
//snop 2 : 0.730~0.732ms
//snop 3 : 0.719~0.726ms
//snop 4 : 0.721~0.729ms
//no snop: 0.749~0.751ms
__device__ __forceinline__ void snop() {
    asm ("snop 3");
}

#ifdef MACART_FAST_GELU
__device__ __forceinline__ float gelu_tanh_approximate(float x) {
    //y = sqrt(2/pi) * (x + 0.044715x^3)
    //gelu(x) = 0.5*x*(1+tanh(y))
    const float SQRT_2_D_PI = 0.79788456080286535587989211986876;
    const float KAPPA = 0.044715;
    const float ONE = 1.0f;
    const float HALF = 0.5f;
    float pow3_x = x * x * x;
    float y = SQRT_2_D_PI * (x + KAPPA * pow3_x);
    float tanhy = ::tanh(y);
    return HALF * x * (ONE + tanhy);
}
#endif

__global__ void ppl_cukernel_gelu_fp16_pack_x4(const size_t count, const half2 *input, half2 *output) {
    #if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        index*=4;
        if(index >= count) return;
        const float4 val = (reinterpret_cast<const float4*>(input+index))[0];
        const half2* val_ptr = (const half2*)&val;

        float4 out;
        half2* out_ptr = (half2*)&out;
        #pragma unroll 4
        for (int i = 0; i < 4; i++) {
            half2 h_val = val_ptr[i];
            float2 f_val = __half22float2(h_val);
#ifdef MACART_FAST_GELU
            out_ptr[i].x = gelu_tanh_approximate(f_val.x);
            out_ptr[i].y = gelu_tanh_approximate(f_val.y);
#else
            half2 t_val;
            t_val.x = erff(f_val.x * 0.707106781f);
            t_val.y = erff(f_val.y * 0.707106781f);
            half2 one_constant = {1,1};
            half2 half_constant = {0.5,0.5};
            t_val = __hmul2(half_constant, __hmul2(h_val, __hadd2(one_constant, t_val)));
            out_ptr[i] = t_val;
#endif
            //snop();
        }
        *(reinterpret_cast<float4*>(output+index)) = out;
    #endif
}
#endif


__global__ void ppl_cukernel_gelu_fp32(const size_t count, const float *input, float *output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= count) return;
    float val = input[index];
    float out_val = val * 0.5f * (1.0f + erff(val * 0.707106781f));
    output[index] = out_val;
}

__global__ void ppl_cukernel_gelu_fp32_pack(const size_t count, const float4 *input, float4 *output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= count) return;
    float4 val = input[index];
    float4 out_val;
    out_val.x = val.x * 0.5f * (1.0f + erff(val.x * 0.707106781f));
    out_val.y = val.y * 0.5f * (1.0f + erff(val.y * 0.707106781f));
    out_val.z = val.z * 0.5f * (1.0f + erff(val.z * 0.707106781f));
    out_val.w = val.w * 0.5f * (1.0f + erff(val.w * 0.707106781f));

    output[index] = out_val;
}

__global__ void ppl_cukernel_gelu_int8(const size_t count, const int8_t *input, int8_t *output, float in_scale, float out_scale,int block_dim_x, int gridDim_x) {
    int block_size = block_dim_x << 4;
    int block_offset = blockIdx.x * block_size;
    block_size = min(count - block_offset, block_size);
    const int8_t* ptr_input = input + block_offset;
    int8_t* ptr_output = output + block_offset;
    float div_scale = 1.0 / out_scale;
    if(blockIdx.x != gridDim_x - 1) {
        int thread_offset = threadIdx.x << 4;
        float4 reg_input = *(float4*)(ptr_input + thread_offset);
        int8_t* ptr_reg_input = (int8_t*)&reg_input;
        float4 reg_dst;
        int8_t* ptr_reg_dst = (int8_t*)&reg_dst;
        #pragma unroll 16
        for(int i = 0; i < 16; i++) {
            float val = ptr_reg_input[i] * in_scale;
            float out_val = val * 0.5f * (1.0f + erff(val * 0.707106781f));
            val = out_val * div_scale;
            int dst = round(val);
            dst = max(min(dst,127),-128);
            ptr_reg_dst[i] = dst;
        }
        *(float4*)(ptr_output + thread_offset) = reg_dst;
    } else {
        for(int i = threadIdx.x; i < block_size; i += block_dim_x) {
            float val = ptr_input[i] * in_scale;
            float out_val = val * 0.5f * (1.0f + erff(val * 0.707106781f));
            val = out_val * div_scale;
            int dst = round(val);
            dst = max(min(dst,127),-128);
            ptr_output[i] = dst;
        }
    }
}

ppl::common::RetCode PPLCUDAGeluForwardImp(
    cudaStream_t stream,
    const void* input,
    ppl::common::TensorShape* input_shape,
    void* output,
    float in_scale,
    float out_scale){

        int BS = 256;
        uint64_t elemCount = input_shape->CalcElementsIncludingPadding();
        uint64_t GS = (elemCount + BS - 1) / BS;
        if (input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
            if (elemCount % 4 == 0) {
                GS = ((elemCount >> 2) + BS - 1) / BS;
                ppl_cukernel_gelu_fp32_pack<<<GS, BS, 0, stream>>>(
                    elemCount >> 2, (const float4*)input, (float4*)output);
            } else {
            ppl_cukernel_gelu_fp32<<<GS, BS, 0, stream>>>(
                elemCount, (const float*)input, (float*)output);
            }
        
        } else if (input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
            if (elemCount % 2 == 0) {
                GS = ((elemCount >> 1) + BS - 1) / BS;
#ifdef PPLNN_USE_MACA
                bool use_4x = (elemCount >> 1) % (4 * BS) == 0;
                if (use_4x) {
                    ppl_cukernel_gelu_fp16_pack_x4<<<GS/4, BS, 0, stream>>>(
                        elemCount >> 1, (const half2*)input, (half2*)output);
                } else {
#endif
                ppl_cukernel_gelu_fp16_pack<<<GS, BS, 0, stream>>>(
                    elemCount >> 1, (const half2*)input, (half2*)output);
#ifdef PPLNN_USE_MACA
                }
#endif
            } else {
            ppl_cukernel_gelu_fp16<<<GS, BS, 0, stream>>>(
                elemCount, (const half*)input, (half*)output);
            }

        } else if(input_shape->GetDataType() == ppl::common::DATATYPE_INT8) {
            int blockSize = 512;
            int gridSize = (elemCount + blockSize * 16 - 1) / (blockSize*16);
            ppl_cukernel_gelu_int8<<<gridSize, blockSize, 0, stream>>>(elemCount, (const int8_t*)input, (int8_t*)output, in_scale, out_scale, blockSize,gridSize);
        } else {
            return ppl::common::RC_UNSUPPORTED;  //TODO
        }

    return ppl::common::RC_SUCCESS;
}