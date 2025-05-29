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

#include "cudakernel/reduce/reduce.h"
#include "cudakernel/reduce/reduce_kernel.h"
#include "../reformat/cvt_int8_float.cuh"
#include "cudakernel/nn/global_pooling_ave.h"

// void printtoHost(half *dev,int batchsize,int channel,int height,int width, int format);
template <int VPT>
struct BytesToType;

template <>
struct BytesToType<2>
{
    using type = uint16_t;
};
template <>
struct BytesToType<4>
{
    using type = uint32_t;
};
template <>
struct BytesToType<8>
{
    using type = uint64_t;
};
template <>
struct BytesToType<16>
{
    using type = float4;
};

template <int Bytes>
__device__ inline void copy(const void* local, void* data)
{
    using T = typename BytesToType<Bytes>::type;

    const T* in = static_cast<const T*>(local);
    T* out = static_cast<T*>(data);
    *out = *in;
}

template<>
__device__ inline void copy<32>(const void* local, void* data)
{
    const float4* in = static_cast<const float4*>(local);
    float4* out = static_cast<float4*>(data);
    out[0] = in[0];
    out[1] = in[1];
}

template<>
__device__ inline void copy<64>(const void* local, void* data)
{
    const float4* in = static_cast<const float4*>(local);
    float4* out = static_cast<float4*>(data);
    out[0] = in[0];
    out[1] = in[1];
    out[2] = in[2];
    out[3] = in[3];
}

__global__ void CudaSetInitVal(
    half* output,
    half initval,
    int64_t size)
{
    int tid     = blockIdx.x * gridDim.x + threadIdx.y * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    output[tid] = initval;
}
__global__ void CudaSetInitVal(
    float* output,
    float initval,
    int64_t size)
{
    int tid     = blockIdx.x * gridDim.x + threadIdx.y * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    output[tid] = initval;
}
__global__ void CudaSetInitVal(
    int64_t* output,
    int64_t initval,
    int64_t size)
{
    int tid     = blockIdx.x * gridDim.x + threadIdx.y * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    output[tid] = initval;
}
__global__ void CudaSetInitVal(
    int32_t* output,
    int32_t initval,
    int32_t size)
{
    int tid     = blockIdx.x * gridDim.x + threadIdx.y * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    output[tid] = initval;
}
__global__ void CudaSetInitVal(
    int8_t* output,
    int8_t initval,
    int8_t size)
{
    int tid     = blockIdx.x * gridDim.x + threadIdx.y * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    output[tid] = initval;
}
template <typename T>
void SetInitVal(
    void* output,
    T initval,
    int64_t size)
{
    dim3 blockDim(32, 32);
    dim3 gridDim(DivUpPassive(32 * 32, size));
    CudaSetInitVal<<<gridDim, blockDim>>>((T *)output, initval, size);
    cudaDeviceSynchronize();
}

template <typename srcT>
__global__ void rescale_from_int8_to_int8(int32_t num_elems,
    const srcT* input, int8_t* output, QuantKernelParamCuda qparam) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= num_elems) return;
    float inter_val = (float)(input[tid] - qparam.i_zero_point) * qparam.i_step;
    output[tid] = _float2int8(inter_val, qparam.o_step, qparam.o_zero_point);
}

template <typename srcT>
ppl::common::RetCode PPLRescaleInt82Int8(cudaStream_t stream, const ppl::common::TensorShape* output_shape,
    const srcT* input, int8_t* output, const QuantKernelParamCuda* qparam) {
    int num_elems = output_shape->CalcElementsExcludingPadding();
    int block_size = 256;
    int grid_size = (num_elems + block_size - 1) / block_size;
    rescale_from_int8_to_int8<srcT><<<grid_size, block_size, 0, stream>>>(num_elems,
    input, output, *qparam);
    return ppl::common::RC_SUCCESS;
}

std::pair<dim3, dim3> ComputeKernelConfigure(
    ReduceParam param,
    ReduceMode mode,
    int64_t& num_elements,
    bool& multi_block,
    PPLReduceDimDes& des)
{
    switch (mode) {
        case ReduceAll:
            return ComputeReduceAllConfigure(param, mode, num_elements, multi_block, des);
        case ReduceRow:
            return ComputeReduceRowConfigure(param, mode, num_elements, multi_block, des);
        case ReduceCol:
            return ComputeReduceColConfigure(param, mode, num_elements, multi_block, des);
        default:
            return ComputeReduceAllConfigure(param, mode, num_elements, multi_block, des);
    }
}

template <typename T, class Operator>
__global__ void ppl_cukernel_ReduceL2_case256512_fp32(
    SumOp<float,float,float> op
)
{
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ T temp[512];

    temp[threadIdx.x] = (T)op.InitVal();

    temp[threadIdx.x] = (T)op.fetch(globalIdx);
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            temp[threadIdx.x] += temp[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        op.out(blockIdx.x, temp[0]);
    }
}
//ReduceL2 for case datatype fp32 tensor[256,512]---->[256,1]
template <class Operator>
ppl::common::RetCode PPLCUDAReduceL2256512fp32OPImp(
    cudaStream_t stream,
    ReduceParam param,
    PPLReduceDimDes des,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output)
{
    typedef typename Operator::acctype acc_type;
    typedef typename Operator::srctype in_type;
    typedef typename Operator::dsttype dst_type;
    Operator op((in_type *)input, (dst_type *)output);
    dst_type initval     = static_cast<dst_type>(op.InitVal());
    dim3 gridDim(256,1);
    dim3 blockDim(512, 1);
    ppl_cukernel_ReduceL2_case256512_fp32<acc_type,Operator><<<gridDim, blockDim, 0, stream>>>(op);
    return ppl::common::RC_SUCCESS;
}

template <class Operator>
ppl::common::RetCode PPLCUDAReduceOPImp(
    cudaStream_t stream,
    ReduceParam param,
    PPLReduceDimDes des,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output)
{
    typedef typename Operator::acctype acc_type;
    typedef typename Operator::srctype in_type;
    typedef typename Operator::dsttype dst_type;
    Operator op((in_type *)input, (dst_type *)output);
    ReduceMode mode      = pplGetReduceMode(des);
    bool multi_block     = false;
    int64_t num_elements = 1;
    auto configure       = ComputeKernelConfigure(param, mode, num_elements, multi_block, des);
    dst_type initval     = static_cast<dst_type>(op.InitVal());
#ifdef PPLNN_USE_MACA
    //As ppl_reduce is a small kernel which runs in microseconds, each if / else clause
    //will make obvious negative effects on it. Fortunately, templated conditions will
    //remove these annoying if clauses by compiler at compile time.
    //Of course, we have to compile more specialized kernel functions
    int reduce_version = 0;
    if (des.n_reduce < 32 && des.split_k_num == 1) reduce_version = ReduceVersionSI;
    else if (des.n_reduce < 1024 && des.split_k_num == 1) reduce_version = ReduceVersionMI;
    else reduce_version = ReduceVersionLI;
    //Can process 4 rows per thread in ppl_reduce_row_mi
    bool can_use_4x = (des.n_outer % (configure.first.y * 4)) == 0;
    //@param param is not needed by reduce mean, but kept for compability
#define PPL_REDUCE_WITH_TYPE(MULTI_BLOCK, MODE, REDUCE_VERSION, REDUCE_TYPE)                                                                                                        \
    switch (REDUCE_TYPE) {                                                                                                                                                          \
        case REDUCE_MEAN:                                                                                                                                                           \
            if (can_use_4x && MODE == ReduceRow && REDUCE_VERSION == ReduceVersionMI) {                                                                                             \
                dim3 grid = configure.second;                                                                                                                                       \
                grid.x /= 4;                                                                                                                                                        \
                ppl_reduce<acc_type, Operator, BLOCKSIZE, MULTI_BLOCK, MODE, REDUCE_VERSION, REDUCE_MEAN, true><<<grid, configure.first, 0, stream>>>(op, des, param);              \
            } else {                                                                                                                                                                \
                ppl_reduce<acc_type, Operator, BLOCKSIZE, MULTI_BLOCK, MODE, REDUCE_VERSION, REDUCE_MEAN, false><<<configure.second, configure.first, 0, stream>>>(op, des, param); \
            }                                                                                                                                                                       \
            break;                                                                                                                                                                  \
        default:                                                                                                                                                                    \
            if (can_use_4x && MODE == ReduceRow && REDUCE_VERSION == ReduceVersionMI) {                                                                                             \
                dim3 grid = configure.second;                                                                                                                                       \
                grid.x /= 4;                                                                                                                                                        \
                ppl_reduce<acc_type, Operator, BLOCKSIZE, MULTI_BLOCK, MODE, REDUCE_VERSION, REDUCE_DEFAULT, true><<<grid, configure.first, 0, stream>>>(op, des, param);           \
            } else {                                                                                                                                                                \
                ppl_reduce<acc_type, Operator, BLOCKSIZE, MULTI_BLOCK, MODE, REDUCE_VERSION, REDUCE_DEFAULT, false><<<configure.second, configure.first, 0, stream>>>(op, des, param); \
            }                                                                                                                                                                       \
    }

#define PPL_REDUCE_WITH_VERSION(MULTI_BLOCK, MODE) \
    if (reduce_version == ReduceVersionSI) {\
        PPL_REDUCE_WITH_TYPE(MULTI_BLOCK, MODE, ReduceVersionSI, param) \
    } else if (reduce_version == ReduceVersionMI) {\
        PPL_REDUCE_WITH_TYPE(MULTI_BLOCK, MODE, ReduceVersionMI, param) \
    } else {\
        PPL_REDUCE_WITH_TYPE(MULTI_BLOCK, MODE, ReduceVersionLI, param) \
    }

#define CASE(Mode)                                                                                                                          \
    case Mode:                                                                                                                              \
        if (multi_block) {                                                                                                                  \
            SetInitVal(output, initval, des.n_inner *des.n_outer);                                                                          \
            PPL_REDUCE_WITH_VERSION(true, (int)Mode)                                                                                        \
        } else {                                                                                                                            \
            PPL_REDUCE_WITH_VERSION(false, (int)Mode)                                                                                       \
        }                                                                                                                                   \
        break;

    switch (mode) {
        CASE(ReduceAll)
        CASE(ReduceRow)
        CASE(ReduceCol)
        default:
            return ppl::common::RC_SUCCESS;
    }
#undef PPL_REDUCE_WITH_TYPE
#undef PPL_REDUCE_WITH_VERSION
#else
#define CASE(Mode)                                                                                                                         \
    case Mode:                                                                                                                             \
        if (multi_block) {                                                                                                                 \
            SetInitVal(output, initval, des.n_inner *des.n_outer);                                                                         \
            ppl_reduce<acc_type, Operator, BLOCKSIZE, true, (int)Mode><<<configure.second, configure.first, 0, stream>>>(op, des, param);  \
        } else                                                                                                                             \
            ppl_reduce<acc_type, Operator, BLOCKSIZE, false, (int)Mode><<<configure.second, configure.first, 0, stream>>>(op, des, param); \
        break;

    switch (mode) {
        CASE(ReduceAll)
        CASE(ReduceRow)
        CASE(ReduceCol)
        default:
            return ppl::common::RC_SUCCESS;
    }
#endif
#undef CASE
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode PPLCUDAReduceForwardImp_NHWC(
    cudaStream_t stream,
    ReduceParam param,
    PPLReduceDimDes des,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output,
    void* tmp_buffer,
    const QuantKernelParamCuda* qparam);

ppl::common::RetCode PPLCUDAReduceForwardImp(
    cudaStream_t stream,
    ReduceParam param,
    PPLReduceDimDes des,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output,
    void* tmp_buffer,
    const QuantKernelParamCuda* qparam)
{
#define CASEFP16(Mode, OP, Tin, Tout, Tacc) \
    case Mode:                              \
        return PPLCUDAReduceOPImp<OP<Tin, Tout, Tacc>>(stream, param, des, input_shape, input, output_shape, output);
#define CASEFP32(Mode, OP, Tin, Tout, Tacc) \
    case Mode:                              \
        return PPLCUDAReduceOPImp<OP<Tin, Tout, Tacc>>(stream, param, des, input_shape, input, output_shape, output);
#define CASEINT64(Mode, OP, Tin, Tout, Tacc) \
    case Mode:                              \
        return PPLCUDAReduceOPImp<OP<Tin, Tout, Tacc>>(stream, param, des, input_shape, input, output_shape, output);
#define CASEINT32(Mode, OP, Tin, Tout, Tacc) \
    case Mode:                              \
        return PPLCUDAReduceOPImp<OP<Tin, Tout, Tacc>>(stream, param, des, input_shape, input, output_shape, output);
#define CASEINT8(Mode, OP, Tin, Tout, Tacc) \
    case Mode:                              \
        status = PPLCUDAReduceOPImp<OP<Tin, Tout, Tacc>>(stream, param, des, input_shape, input, output_shape, tmp_buffer);\
        break;

    if(output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16 || output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 ||
        output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC || input_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16 ||
        input_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 ||
        input_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC ){
        return PPLCUDAReduceForwardImp_NHWC(stream, param, des, input_shape, input, output_shape, output, tmp_buffer,qparam);
    }

    if (input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        switch (param) {
            CASEFP16(ReduceSum, SumOp, half, half, float)
            CASEFP16(ReduceProd, ProdOp, half, half, float)
            CASEFP16(ReduceMean, SumOp, half, half, float)
            CASEFP16(ReduceMax, MaxOp, half, half, half)
            CASEFP16(ReduceMin, MinOp, half, half, half)
            default:
                return ppl::common::RC_UNSUPPORTED;
        }
    } else if (input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
        if(des.n_inner == 1 && des.n_outer == 256 && des.n_reduce == 512 && des.split_k_num == 1 && param == ReduceSum && pplGetReduceMode(des) == ReduceRow)
            {
                return PPLCUDAReduceL2256512fp32OPImp<SumOp<float,float,float>>(stream, param, des, input_shape, input, output_shape, output);
            }
        else{
            switch (param) {
                CASEFP32(ReduceSum, SumOp, float, float, float)
                CASEFP32(ReduceProd, ProdOp, float, float, float)
                CASEFP32(ReduceMean, SumOp, float, float, float)
                CASEFP32(ReduceMax, MaxOp, float, float, float)
                CASEFP32(ReduceMin, MinOp, float, float, float)
                default:
                    return ppl::common::RC_UNSUPPORTED;
            }
        }
    } else if (input_shape->GetDataType() == ppl::common::DATATYPE_INT64) {
        switch (param) {
            CASEINT64(ReduceSum, SumOp, int64_t, int64_t, int64_t)
            CASEINT64(ReduceProd, ProdOp, int64_t, int64_t, int64_t)
            CASEINT64(ReduceMean, SumOp, int64_t, int64_t, int64_t)
            CASEINT64(ReduceMax, MaxOp, int64_t, int64_t, int64_t)
            CASEINT64(ReduceMin, MinOp, int64_t, int64_t, int64_t)
            default:
                return ppl::common::RC_UNSUPPORTED;
        }
    } else if (input_shape->GetDataType() == ppl::common::DATATYPE_INT32) {
        switch (param) {
            CASEINT32(ReduceSum, SumOp, int32_t, int32_t, int32_t)
            CASEINT32(ReduceProd, ProdOp, int32_t, int32_t, int32_t)
            CASEINT32(ReduceMean, SumOp, int32_t, int32_t, int32_t)
            CASEINT32(ReduceMax, MaxOp, int32_t, int32_t, int32_t)
            CASEINT32(ReduceMin, MinOp, int32_t, int32_t, int32_t)
            default:
                return ppl::common::RC_UNSUPPORTED;
        }
    } else if (input_shape->GetDataType() == ppl::common::DATATYPE_INT8) {
        ppl::common::RetCode status = ppl::common::RC_SUCCESS;
        switch (param) {
            CASEINT8(ReduceSum, SumOp, int8_t, float, float)
            CASEINT8(ReduceProd, ProdOp, int8_t, float, float)
            CASEINT8(ReduceMean, SumOp, int8_t, float, float)
            CASEINT8(ReduceMax, MaxOp, int8_t, int8_t, int8_t)
            CASEINT8(ReduceMin, MinOp, int8_t, int8_t, int8_t)
            default:
                status =  ppl::common::RC_UNSUPPORTED;
        }
        // rescale from int8_t to int8_t
        if (param == ReduceSum || param == ReduceProd || param == ReduceMean) {
            status = PPLRescaleInt82Int8<float>(stream, output_shape, (const float*)tmp_buffer,
                (int8_t*)output, qparam);
        } else {
            status = PPLRescaleInt82Int8<int8_t>(stream, output_shape, (const int8_t*)tmp_buffer,
                (int8_t*)output, qparam);
        }
        return status;
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
    // return ppl::common::RC_SUCCESS;
#undef CASE
#undef CASEPROMOTION
}

static int condition_nhwc(const ppl::common::TensorShape * input_shape, const ppl::common::TensorShape * output_shape,
            PPLReduceDimDes des,ReduceMode mode){
    int status = 0;
    if(mode == 2)
    {
        if(input_shape->GetDimCount() == 4 && des.n_reduce == input_shape->GetDim(2) * input_shape->GetDim(3))
        {
            return 3;
        }
        else if(input_shape->GetDimCount() == 3 && des.n_reduce == input_shape->GetDim(2))
        {
            return 3;
        }
        else if(input_shape->GetDimCount() == 5 && des.n_reduce == input_shape->GetDim(2) * input_shape->GetDim(3) * input_shape->GetDim(4))
        {
            return 3;
        }
        else
        {
            return 0;
        }
    }
    else if(mode == 3)
    {
        if(input_shape->GetDimCount() == 3)
        {
            if(des.n_reduce == input_shape->GetDim(1)&& des.n_inner == input_shape->GetDim(2) && des.n_outer == input_shape->GetDim(0)){
                return 1;
            } else if (des.n_reduce == input_shape->GetDim(0) && des.n_inner == input_shape->GetDim(1)*input_shape->GetDim(2)) {
                return 2;
            }
        }
        else if(input_shape->GetDimCount() == 4)
        {
            if(des.n_reduce == input_shape->GetDim(1)&& des.n_inner == input_shape->GetDim(2)*input_shape->GetDim(3) && des.n_outer == input_shape->GetDim(0))
            {
                return 1;
            }
        }
        else if(input_shape->GetDimCount() == 5)
        {
            if(des.n_reduce == input_shape->GetDim(1)&& des.n_inner == input_shape->GetDim(2) * input_shape->GetDim(3) * input_shape->GetDim(4) && des.n_outer == input_shape->GetDim(0))
            {
                return 1;
            }
            if(des.n_reduce == input_shape->GetDim(0) && des.n_inner == input_shape->GetDim(2) * input_shape->GetDim(3) * input_shape->GetDim(4) * input_shape->GetDim(1))
            {
                return 2;
            }
            return 0;
        }
        else
        {
            return 0;
        }
    }
    return 0;
}

template<typename T, int branch>
__global__ void ppl_cukernel_reduce_m3_nhwc(
    const T* input,
    T* output,
    int channels,
    int pad_channels,
    int HW,
    int i_stride,
    int o_stride,
    ReduceParam param)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int c  = (blockIdx.y * blockDim.y + threadIdx.y);
    int bc = (blockIdx.z * pad_channels + c);
    if (c >= pad_channels)
        return;
    if(branch != 1){
        if(c >= channels){
            output[blockIdx.z*o_stride + c] = (T)0;
            return;
        }
    }

    float res = 0.f;
    for (int i = 0; i < HW; i += 64) {
        bool pred0 = i + threadIdx.x * 2 + 0 < HW;
        bool pred1 = i + threadIdx.x * 2 + 1 < HW;
        T ival0 = pred0 ? input[bc * i_stride + 2 * threadIdx.x + i + 0] : T(0);
        T ival1 = pred1 ? input[bc * i_stride + 2 * threadIdx.x + i + 1] : T(0);
        float val  = (float)ival0 + (float)ival1;
        res        += val;
    }

    for (int offset = 16; offset > 0; offset /= 2) {
#if __CUDACC_VER_MAJOR__ >= 9
        float val = __shfl_down_sync(0xffffffff, res, offset);
#else
        float val = __shfl_down(res, offset);
#endif
        res +=  val;
    }

    // store output
    if (threadIdx.x == 0){
        T dst ;
        if(param == ReduceMean) {
            dst = (T)(res / HW);
        } else {
            dst = res;
        }

        if(branch == 1){
            output[bc*o_stride] = dst;
            for(int i = 1; i < o_stride; i++){
                output[bc*o_stride + i] = (T)0;
            }
        }else{
            output[blockIdx.z*o_stride + c] = dst;
        }
    }
#endif
}

//input outer x reduce x hw x i_stride
//output outer x hw x pad_channels
template<typename T, typename acc_T>
__global__ void ppl_cukernel_reduce_col_nhwc(
    const T* input,
    T* output,
    int outer,          // 1
    int reduce,         // 2
    int hw,             // 1
    int i_stride,       // 32
    int channels,       // 32
    int pad_channels,   // 48
    ReduceParam param)
{
    const T* ptr_block_input = input + blockIdx.y * reduce * hw * i_stride;
    T* ptr_block_output = output + blockIdx.y * hw * pad_channels;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index >= hw * channels) return;
    int c = index % channels;
    int hw_id = index / channels;
    acc_T sum = (acc_T)0;

    for(int i = 0; i < reduce; i++) {
        sum += (acc_T)ptr_block_input[i * hw * i_stride + hw_id * i_stride + c];
    }
    T dst;
    if(param == ReduceMean) {
        dst = (T)(sum / reduce);
    } else {
        dst = sum;
    }
    ptr_block_output[hw_id * pad_channels + c] = dst;
}

template<typename T, typename acc_T>
__global__ void ppl_cukernel_reduce_col_nhwc_5D(
    const T* input,
    T* output,
    int reduce,
    int DHW,
    int channels,
    int pad_channels,
    int o_stride,
    ReduceParam param)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= DHW * channels) return;

    int c = index % channels;
    int hw_id = index / channels;
    acc_T sum = (acc_T)0;

    for(int i = 0; i < reduce; i++) {
        sum += (acc_T)input[i * o_stride + hw_id * channels + c];
    }
    T dst;
    if(param == ReduceMean) {
        dst = (T)(sum / reduce);
    } else {
        dst = sum;
    }
    output[hw_id * pad_channels + c] = dst;
}

template<typename T, typename acc_T, int32_t VPT>
__global__ void ppl_cukernel_reduce_col_nhwc_5D_opt(
    const T* input,
    T* output,
    int reduce,
    int DHW,
    int channels,
    int pad_channels,
    int o_stride,
    ReduceParam param)
{
    T sum[VPT];
    T tmp_data[VPT];
#pragma unroll
    for(int i = 0; i < VPT; i++){
        sum[i] = (acc_T)0;
    }
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= DHW * channels) return;

    int c = index % channels;
    int hw_id = index / channels;

    for(int i = 0; i < reduce; i++) {
        int idx = i * o_stride + hw_id * channels * 8 + c * 8;
        copy<sizeof(half) * VPT>(&input[idx], tmp_data);
#pragma unroll
        for(int j = 0; j < VPT; j++){
            sum[j] = (T)((acc_T)sum[j] + (acc_T)tmp_data[j]);
        }
    }
    if(param == ReduceMean) {
#pragma unroll
        for(int j = 0; j < VPT; j++){
            sum[j] = (T)((acc_T)sum[j] / (acc_T)reduce);
        }
    }
    int idx_out = hw_id * channels * 8 + c * 8;
    copy<sizeof(half) * VPT>(sum, &output[idx_out]);
}

template<typename T>
__global__ void ppl_reduce_all_nhwc(
    const int num_elems,
    const T* input,
    T* output,
    int channels,
    int pad_channels,
    ReduceParam param)
{
    float res = 0;
    for(int i = threadIdx.x; i < num_elems; i += blockDim.x){
        int h,w;
        h = i / channels;
        w = i % channels;
        res += (float)input[h * pad_channels + w];
    }
    __shared__ float sum_buffer[1024];
    sum_buffer[threadIdx.x] = res;
    __syncthreads();
    res = 0;
    if(threadIdx.x == 0){
        for(int i = 0; i < blockDim.x; i++){
            res += sum_buffer[i];
        }
        T dst;
        if(param == ReduceMean){
          dst = res / num_elems;
        } else {
          dst = res;
        }
        output[0] = dst;
    }
}

template<int branch>
__global__ void ppl_cukernel_reduce_m3_nhwc_int8(
    const int8_t* input,
    int8_t* output,
    int channels,
    int pad_channels,
    int HW,
    int i_stride,
    int o_stride,
    float in_scale,
    float out_scale,
    ReduceParam param)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int c  = (blockIdx.y * blockDim.y + threadIdx.y);
    int bc = (blockIdx.z * pad_channels + c);
    if (c >= pad_channels)
        return;
    if(branch != 1){
        if(c >= channels){
            output[blockIdx.z * o_stride + c] = 0;
            return;
        }
    }

    int32_t res = 0.f;
    for (int i = 0; i < HW; i += 64) {
        bool pred0 = i + threadIdx.x * 2 + 0 < HW;
        bool pred1 = i + threadIdx.x * 2 + 1 < HW;
        int8_t ival0 = pred0 ? input[bc * i_stride + 2 * threadIdx.x + i + 0] : (0);
        int8_t ival1 = pred1 ? input[bc * i_stride + 2 * threadIdx.x + i + 1] : (0);
        int32_t val  = ival0 + ival1;
        res += val;
    }

    for (int offset = 16; offset > 0; offset /= 2) {
#if __CUDACC_VER_MAJOR__ >= 9
        float val = __shfl_down_sync(0xffffffff, res, offset);
#else
        float val = __shfl_down(res, offset);
#endif
        res += val;
    }

    // store output
    if (threadIdx.x == 0){
        int32_t temp;
        if(param == ReduceMean){
            temp = round(float(res) / HW * in_scale / out_scale);
        } else {
            temp = round(float(res) * in_scale / out_scale);
        }
        temp = min(temp,127);
        temp = max(temp,-128);
        if(branch == 1){
            output[bc*o_stride] = temp;
            for(int i = 1; i < o_stride; i++){
                output[bc*o_stride + i] = 0;
            }
        }else{
            output[blockIdx.z*o_stride + c] = temp;
        }
    }
#endif
}

__global__ void ppl_reduce_all_nhwc_int8(
    const int num_elems,
    const int8_t* input,
    int8_t* output,
    int channels,
    int pad_channels,
    float in_scale,
    float out_scale,
    ReduceParam param)
{
    int32_t res = 0;
    for(int i = threadIdx.x; i < num_elems; i += blockDim.x){
        int h, w;
        h = i / channels;
        w = i % channels;
        res += input[h * pad_channels + w];
    }
    __shared__ int sum_buffer[1024];
    sum_buffer[threadIdx.x] = res;
    __syncthreads();
    res = 0;
    if(threadIdx.x == 0){
        for(int i = 0; i < blockDim.x; i++){
            res += sum_buffer[i];
        }
        int64_t temp;
        if (param == ReduceMean) {
            temp= round((float(res) / num_elems * in_scale) / out_scale);
        } else {
            temp= round((float(res) * in_scale) / out_scale);
        }
        if (temp > 127)         temp = 127;
        else if (temp < -128)   temp = -128;
        output[0] = temp;
    }
}

template<int PAD_CHANNELS>
__launch_bounds__(1024)
__global__ void ppl_cukernel_reducesum_Opt_C2N(
    const int8_t* input,
    int8_t* output,
    int batch,
    int pad_channels,
    int HW,
    float scale
) {
    const int8_t* ptr_block_input = input + blockIdx.x * PAD_CHANNELS * HW;
    int8_t* ptr_block_output = output + blockIdx.x * PAD_CHANNELS;
    const float4 *ptr_input = (const float4*)(ptr_block_input);

    //Every thread will read 4 bytes per loop, aka process 16 images
    //A block have 512 threads, so, each loop, a block can handle
    //512 * 16 pixels = 4 channels of 16 images
    __shared__ int32_t results[PAD_CHANNELS]; //store final result

    union {
        float4 packed;
        int8_t unpacked[16];
    } mediate;
    int32_t temp_result[16] = {0};

    for (int i = threadIdx.x; i < HW * PAD_CHANNELS / sizeof(float4); i+=blockDim.x) {
        mediate.packed = ptr_input[i];
        for (int j = 0; j < 16; j++) temp_result[j] += mediate.unpacked[j];
    }

    for (int i = threadIdx.x; i < PAD_CHANNELS; i+= blockDim.x) {
        results[i] = 0;
    }

    __syncthreads();

    //Add thread 0, thread 128, thread 256 and thread 384
    //first loop, thread 0 write 0, thread 128 write 1, thread 256 write 2, thread 384 write 3
    //second loop, thread 0 write 1, thread 128 write 2, thread 256 write 3, thread 384 write 4
    //...
    //sixteenth loop, thread 0 write 15, thread 128 write 0, thread 256 write 1, thread 384 write 2

    //Add thread 1, thread 129, thread 257 and thread 385
    int bid = threadIdx.x / (PAD_CHANNELS / 16);
    int b_start = (threadIdx.x - (bid * (PAD_CHANNELS / 16))) << 4;
    for (int i = 0; i < 16; i++) {
        int write_idx = b_start + (i + bid) % 16;
        results[write_idx] += temp_result[(i + bid) % 16];
        __syncthreads();
    }
    float4* op = (float4*)ptr_block_output;

    for (int i = threadIdx.x; i < (PAD_CHANNELS / 16); i+=blockDim.x) {
        for (int j = 0; j < 16; j++) {
            int32_t temp = round(float(results[(i<<4) + j]) * scale);
            temp = min(temp,127);
            temp = max(temp,-128);
            mediate.unpacked[j] = temp;
        }
        op[i] = mediate.packed;
    }
}

template<typename T>
__global__ void ppl_cukernel_reducesum_NHWC_Opt_line(
    const int8_t* input,
    int8_t* output,
    int batch,
    int pad_channels,
    int HW,
    int line_pb,
    int channelCount,
    float scale
)
{
    constexpr int times = sizeof(T);
    constexpr int SM_SIZE = 8192;
    constexpr int block_size = SM_SIZE / times;
    int inputCount = channelCount * HW;
    int offset = line_pb * channelCount;
    int tid = threadIdx.x;
    int height = HW - line_pb + 1;
    int h = line_pb;
    int stride = pad_channels << 1;
    __shared__ union{
        int8_t m1[SM_SIZE];
        T m2[block_size];
    } sm_input;

    __shared__ int sum_buffer[2048];

    const int8_t* ptr_block_input = input + blockIdx.x * pad_channels * HW;
    int8_t* ptr_block_output = output + blockIdx.x * pad_channels;
    const T* ptr_input = (const T*)ptr_block_input;

    for(int i = tid; i < block_size; i += blockDim.x){
        if(i < inputCount){
            sm_input.m2[i] = *(ptr_input + i);
        }
    }
    __syncthreads();
    ptr_input += offset;

    for(int i = tid; i < pad_channels; i += blockDim.x){
        sum_buffer[i] = sm_input.m1[i] + sm_input.m1[pad_channels + i];
    }
    int offset1 = stride;
    for(int j = 2; j < line_pb; j += 2){
        for(int i = tid; i < pad_channels; i += blockDim.x){
            sum_buffer[i] += sm_input.m1[offset1 + i] + sm_input.m1[offset1 + pad_channels + i];
        }
        offset1 += stride;
    }
    __syncthreads();
    for(; h < height; h += line_pb){
        for(int i = tid; i < block_size; i += blockDim.x){
            if(i < inputCount){
                sm_input.m2[i] = *(ptr_input + i);
            }
        }
        __syncthreads();
        ptr_input += offset;
        int n = 0;
        for(int j = 0; j < line_pb; j+=2){
            for(int i = tid; i < pad_channels; i += blockDim.x){
                sum_buffer[i] += sm_input.m1[n + pad_channels + i] + sm_input.m1[n + i];
            }
            n += stride;
        }
        __syncthreads();
    }

    for(; h < HW; h += line_pb){
        for(int i = tid; i < block_size; i += blockDim.x){
            if(i < inputCount){
                sm_input.m2[i] = *(ptr_input + i);
            }
        }
        __syncthreads();
        ptr_input += offset;
        for(int j = 0; j < line_pb; j++){
            if(h + j < HW){
                for(int i = tid; i < pad_channels; i += blockDim.x){
                    sum_buffer[i] += sm_input.m1[j*pad_channels + i];
                }
            }
        }
        __syncthreads();
    }

    for(int i = tid; i < pad_channels; i += blockDim.x){
        int32_t temp = round(float(sum_buffer[i]) * scale);
        temp = min(temp,127);
        temp = max(temp,-128);
        sm_input.m1[i] = temp;
    }
    __syncthreads();
    for(int i = tid; i < channelCount; i += blockDim.x){
        *((T*)ptr_block_output + i) = *((T*)sm_input.m1 + i);
    }
}

__global__ void ppl_cukernel_redcuesum_NHWC_Opt_dbuf(
    const int8_t* input,
    int8_t* output,
    int batch,
    int pad_channels,
    int HW,
    int lines_per_loop,
    float scale) {
    using T = float4;
    using MT = uint64_t;
    constexpr int IBUF_SZ = 8192;
    constexpr int SZ = IBUF_SZ/2;

    const T* block_input = (T*)(input + pad_channels * HW * blockIdx.x);
    __shared__ union {
        int8_t unpacked[IBUF_SZ];
        T packed[IBUF_SZ/sizeof(T)];
        MT mediate[IBUF_SZ/sizeof(MT)];
    } ibuf; //8K
    __shared__ int sum_buffer[SZ];  //16K

    for (int i = threadIdx.x; i < SZ; i += blockDim.x) {
        sum_buffer[i] = 0;
    }
    int line = 0;
    union {
        uint64_t packed;
        int8_t unpacked[8];
    } i0, i1;
    const int stride_lines_first_loop = lines_per_loop * 2;
    const int stride_first_loop = stride_lines_first_loop * pad_channels;
    const int read_pack_size = stride_first_loop / sizeof(T);
    for (int l = 0 ; l + stride_lines_first_loop <= HW; l += stride_lines_first_loop) {
        if (threadIdx.x < read_pack_size)  {
            //VERY IMPORTANT:
            //A const qualifier is needed for tmp
            //or the compiler will use stp to store tmp
            const float4 tmp = block_input[threadIdx.x];
            block_input += read_pack_size;
            line += stride_lines_first_loop;
            //Before arrive of tmp, do some calculations
            int i0_start = threadIdx.x;
            int i1_start = threadIdx.x + read_pack_size;
            int res[8];
            int res2[8];

            ibuf.packed[threadIdx.x] = tmp;

            __syncthreads();
            //Do a reduce in ibuf.mediate, which means split the read data into two parts
            //and add them
            i0.packed = ibuf.mediate[i0_start];
            i1.packed = ibuf.mediate[i1_start];

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                res[i] = sum_buffer[i + threadIdx.x * 8];
                res2[i] = i0.unpacked[i] + i1.unpacked[i];
                res[i] += res2[i];
            }
            //Following shared buffer store will translate into a sts_b128
            #pragma unroll
            for (int i = 0; i < 8; i++) sum_buffer[threadIdx.x*8 + i] = res[i];
        } else {
            block_input += read_pack_size;
            line += stride_lines_first_loop;
        }
    }

    //if there are more lines than lines_per_loop, we should process it
    if (HW-line >= lines_per_loop) {
        if (threadIdx.x < lines_per_loop * pad_channels / sizeof(T))
            ibuf.packed[threadIdx.x] = block_input[threadIdx.x];
        __syncthreads();
        for (int i = threadIdx.x; i < lines_per_loop * pad_channels; i+=blockDim.x)
            sum_buffer[i] += ibuf.unpacked[i];
        line += lines_per_loop;
        block_input += lines_per_loop * pad_channels / sizeof(T);
    }

    const int rest_data_size = HW * pad_channels - line * pad_channels;
    for (int i = threadIdx.x; i < rest_data_size / sizeof(T); i+=blockDim.x) {
        ibuf.packed[i] = block_input[i];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < rest_data_size; i+=blockDim.x) {
        sum_buffer[i] += ibuf.unpacked[i];
    }
    __syncthreads();

    //Reduce on all channels into the first channel
    for (int i = lines_per_loop; i > 1; ) {
        if ( i % 2 == 1) {
            for (int j = threadIdx.x; j < pad_channels; j+=blockDim.x) {
                sum_buffer[j] += sum_buffer[j+(i-1)*pad_channels];
            }
            __syncthreads();
            i -= 1;
        }
        int stride = i / 2 * pad_channels;
        for (int j = threadIdx.x; j < stride; j+=blockDim.x)
            sum_buffer[j] += sum_buffer[j+stride];
        __syncthreads();
        i /= 2;
    }

    for (int tid = threadIdx.x; tid < pad_channels; tid+=blockDim.x) {
        int32_t temp = round(float(sum_buffer[tid]) * scale);
        temp = min(temp, 127);
        temp = max(temp,-128);
        output[pad_channels  * blockIdx.x + tid] = temp;
    }
}

template <uint32_t block_size, typename T>
__device__ __forceinline__ void WarpReduceAdd(volatile T *sdata, uint32_t tid) {
    if (block_size >= 128) sdata[tid] += sdata[tid + 64];
    if (block_size >= 64) sdata[tid] += sdata[tid + 32];
    if (block_size >= 32) sdata[tid] += sdata[tid + 16];
    if (block_size >= 16) sdata[tid] += sdata[tid + 8];
    if (block_size >= 8) sdata[tid] += sdata[tid + 4];
    if (block_size >= 4) sdata[tid] += sdata[tid + 2];
    if (block_size >= 2) sdata[tid] += sdata[tid + 1];
}

__global__ void ppl_cukernel_reducesum_int8_NHWC_Opt(
    const int64_t* input,
    int8_t* output,
    int batch,
    int pad_channels,
    int HW,
    float scale)
{
    int c        = blockIdx.y * blockDim.y + threadIdx.y;
    if(c >= pad_channels)
        return;
    int b_offset = blockIdx.z * pad_channels;
    const int64_t* ptr_input = input + b_offset * HW + c;
    int8_t* ptr_output = output + ((b_offset + c)<<3);
    int32_t res[8];
    #pragma unroll 8
    for(int i = 0; i < 8; i++)
    {
        res[i] = 0;
    }

    int64_t ival = 0;
    int8_t *ptr_ival = (int8_t*)&ival;
    for(int i = threadIdx.x; i < HW; i += blockDim.x){
        ival = *(ptr_input + i * pad_channels);
        #pragma unroll 8
        for(int j = 0; j < 8; j++)
        {
            res[j] += ptr_ival[j];
        }
    }
    //hw x channel
    __shared__ int32_t sum_buffer[64][33];
    #pragma unroll 8
    for(int i = 0; i < 8; i++)
    {
        sum_buffer[(threadIdx.y<<3) + i][threadIdx.x] = res[i];
    }
    __syncthreads();

    if(threadIdx.x < 16) WarpReduceAdd<32,int32_t>(sum_buffer[(threadIdx.y<<3)],threadIdx.x);
    if(threadIdx.x < 16) WarpReduceAdd<32,int32_t>(sum_buffer[(threadIdx.y<<3) + 1],threadIdx.x);
    if(threadIdx.x < 16) WarpReduceAdd<32,int32_t>(sum_buffer[(threadIdx.y<<3) + 2],threadIdx.x);
    if(threadIdx.x < 16) WarpReduceAdd<32,int32_t>(sum_buffer[(threadIdx.y<<3) + 3],threadIdx.x);
    if(threadIdx.x < 16) WarpReduceAdd<32,int32_t>(sum_buffer[(threadIdx.y<<3) + 4],threadIdx.x);
    if(threadIdx.x < 16) WarpReduceAdd<32,int32_t>(sum_buffer[(threadIdx.y<<3) + 5],threadIdx.x);
    if(threadIdx.x < 16) WarpReduceAdd<32,int32_t>(sum_buffer[(threadIdx.y<<3) + 6],threadIdx.x);
    if(threadIdx.x < 16) WarpReduceAdd<32,int32_t>(sum_buffer[(threadIdx.y<<3) + 7],threadIdx.x);
    __syncthreads();

    if(threadIdx.x < 8)
    {
        int32_t temp = round(float(sum_buffer[(threadIdx.y<<3) + threadIdx.x][0]) * scale);
        temp = min(temp, 127);
        temp = max(temp,-128);
        ptr_output[threadIdx.x] = temp;
    }
}

static __device__ float2 __f2add(float2 val0, float2 val1) {
    float2 res{0.f, 0.f};
    res.x = val0.x + val1.x;
    res.y = val0.y + val1.y;
    return res;
}

template<int TILE_C, int TILE_HW>
__global__ void ppl_cukernel_reduceSum_nhwc(
    const half2* input,
    half2* output,
    int batch,
    int pad_channels,
    int HW)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int c        = blockIdx.x * blockDim.x + threadIdx.x;
    int b_offset = blockIdx.z * pad_channels;
    if (c >= pad_channels)
        return;

    float2 res = float2{0.f, 0.f};
    // main loop
    for (int i = threadIdx.y; i < HW; i += blockDim.y) {
        half2 ival = input[b_offset * HW + i * pad_channels + c];
        res        = __f2add(res, __half22float2(ival));
    }
    __shared__ float2 sum_buffer[TILE_HW][TILE_C];
    sum_buffer[threadIdx.y][threadIdx.x] = res;
    __syncthreads();

    for (int i = (blockDim.y >> 1); i > 0; i = (i >> 1)) {
        if (threadIdx.y < i) {
            float2 res                           = sum_buffer[threadIdx.y + i][threadIdx.x];
            res                                  = __f2add(res, sum_buffer[threadIdx.y][threadIdx.x]);
            sum_buffer[threadIdx.y][threadIdx.x] = res;
            __syncthreads();
        }
    }
    // store output
    if (threadIdx.y == 0) {
        float2 res = sum_buffer[threadIdx.y][threadIdx.x];
        res.x = res.x;
        res.y = res.y;
        output[b_offset + c] = __float22half2_rn(res);
    }
#endif
}

template<int TILE_C, int TILE_HW>
__global__ void ppl_cukernel_reduceSum_nhwc_fp32(
    const float* input,
    float* output,
    int batch,
    int pad_channels,
    int HW)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int c        = blockIdx.x * blockDim.x + threadIdx.x;
    int b_offset = blockIdx.z * pad_channels;
    if (c >= pad_channels)
        return;

    float res = 0.0f;
    // main loop
    for (int i = threadIdx.y; i < HW; i += blockDim.y) {
        float ival = input[b_offset * HW + i * pad_channels + c];
        res        =  res + ival;
    }
    __shared__ float sum_buffer[TILE_HW][TILE_C];
    sum_buffer[threadIdx.y][threadIdx.x] = res;
    __syncthreads();

    for (int i = (blockDim.y >> 1); i > 0; i = (i >> 1)) {
        if (threadIdx.y < i) {
            float res                           = sum_buffer[threadIdx.y + i][threadIdx.x];
            res                                  = res + sum_buffer[threadIdx.y][threadIdx.x];
            sum_buffer[threadIdx.y][threadIdx.x] = res;
            __syncthreads();
        }
    }
    // store output
    if (threadIdx.y == 0) {
        float res = sum_buffer[threadIdx.y][threadIdx.x];
        output[b_offset + c] = res;
    }
#endif
}

template<typename T,typename acc_T>
ppl::common::RetCode PPLCUDAReduceSumImp(
    cudaStream_t stream,
    ppl::common::TensorShape* input_shape, const T* input,
    ppl::common::TensorShape* output_shape, T* output, float in_scale, float out_scale, bool isQuantize)
{
    int in_width;
    if(input_shape->GetDimCount() == 4){
        in_width     = input_shape->GetDim(3);
    } else if(input_shape->GetDimCount() == 3) {
        in_width = 1;
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
    int batch = output_shape->GetDim(0);
    int pad_channels = output_shape->GetDim(1) + output_shape->GetPadding0(1) + output_shape->GetPadding1(1);
    int in_height = input_shape->GetDim(2);
    int HW = in_height * in_width;
    float scale = in_scale / out_scale;
    if(pad_channels == 512 || pad_channels == 1024 || pad_channels == 2048 || pad_channels == 4096){
        dim3 dim_block(1024, 1, 1);
        dim3 dim_grid(batch, 1, 1);
        if (HW < 256) dim_block.x /= 2;
#define CALL_C2N(PAD_CHANNELS) \
        ppl_cukernel_reducesum_Opt_C2N<PAD_CHANNELS><<<dim_grid, \
                                                       dim_block, \
                                                        0, \
                                                        stream>>>((const int8_t*)input, (int8_t*)output, batch, pad_channels, HW, scale)
        if (pad_channels == 512) {
            CALL_C2N(512);
        } else if (pad_channels == 1024) {
            CALL_C2N(1024);
        } else if (pad_channels == 2048) {
            CALL_C2N(2048);
        } else if (pad_channels == 4096) {
            CALL_C2N(4096);
        }
    } else if(pad_channels < 2048 && pad_channels >= 1024 && (pad_channels & 7) == 0){
        dim3 dim_block(512, 1, 1);
        dim3 dim_grid(batch, 1, 1);
        int line_pb = 8192 / pad_channels;
        line_pb = line_pb >> 1 << 1;
        if((pad_channels & 15) == 0){
            int channelCount = pad_channels / 16;
            ppl_cukernel_reducesum_NHWC_Opt_line<float4><<<dim_grid,
                                                            dim_block,
                                                            0,
                                                            stream>>>((const int8_t*)input, (int8_t*)output, batch, pad_channels, HW, line_pb, channelCount, scale);
        }else if((pad_channels & 7) == 0){
            int channelCount = pad_channels / 8;
            ppl_cukernel_reducesum_NHWC_Opt_line<int64_t><<<dim_grid,
                                                            dim_block,
                                                            0,
                                                            stream>>>((const int8_t*)input, (int8_t*)output, batch, pad_channels, HW, line_pb, channelCount, scale);
        }
    } else if ((pad_channels & 15) == 0 && pad_channels < 1024 && HW > 4096 / pad_channels * 2) {
        dim3 dim_block(512, 1, 1);
        dim3 dim_grid(batch, 1, 1);
        int lines_per_loop = 4096 / pad_channels;
        ppl_cukernel_redcuesum_NHWC_Opt_dbuf<<<dim_grid, dim_block, 0, stream>>>(
            (const int8_t*)input, (int8_t*)output, batch, pad_channels, HW, lines_per_loop, scale);
    } else {
        dim3 dim_block(32,8,1);
        int pad_channels1 = pad_channels / 8;
        int channel_blocks = (pad_channels1 + dim_block.y - 1) / (dim_block.y);
        dim3 dim_grid(1, channel_blocks, batch);
        ppl_cukernel_reducesum_int8_NHWC_Opt<<<dim_grid, dim_block, 0, stream>>>((const int64_t*)input, (int8_t*)output, batch, pad_channels1, HW, scale);
    }
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode PPLCUDAReduceSumImp_NHWC(
    cudaStream_t stream,
    ppl::common::TensorShape* input_shape,
    const void* input,
    ppl::common::TensorShape* output_shape,
    void* output, float in_scale, float out_scale)
{
    if (output_shape->GetDataType() == ppl::common::DATATYPE_INT8) {
        return PPLCUDAReduceSumImp<int8_t,int32_t>(
            stream, input_shape, (const int8_t*)input, output_shape, (int8_t*)output, in_scale, out_scale,true);
    } else if(output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        int in_width;
        if(input_shape->GetDimCount() == 4){
            in_width     = input_shape->GetDim(3);
        } else if(input_shape->GetDimCount() == 3) {
            in_width = 1;
        } else {
            return ppl::common::RC_UNSUPPORTED;
        }
        int batch = output_shape->GetDim(0);
        int pad_channels = output_shape->GetDim(1) + output_shape->GetPadding0(1) + output_shape->GetPadding1(1);
        int in_height = input_shape->GetDim(2);
        int in_hw = in_height * in_width;
        dim3 dim_block(32, 8, 1); // (c, hw, 1)
        int padChannelsDivide = (pad_channels >> 1); // half2
        int channel_blocks    = (padChannelsDivide + dim_block.x - 1) / dim_block.x;
        constexpr int block_threshold = 64;
        constexpr int hw_threshold = 128;
        if (channel_blocks * batch < block_threshold && in_hw > hw_threshold) {
            dim3 dim_block(8, 32, 1); // (c, hw, 1)
            int channel_blocks    = (padChannelsDivide + dim_block.x - 1) / dim_block.x;
            dim3 dim_grid(channel_blocks, 1, batch);
            ppl_cukernel_reduceSum_nhwc<8, 32><<<dim_grid,
                                                dim_block,
                                                0,
                                                stream>>>((const half2*)input, (half2*)output, batch, padChannelsDivide, in_hw);
        } else {
            dim3 dim_grid(channel_blocks, 1, batch);
            ppl_cukernel_reduceSum_nhwc<32, 8><<<dim_grid,
                                                dim_block,
                                                0,
                                                stream>>>((const half2*)input, (half2*)output, batch, padChannelsDivide, in_hw);
        }
        return ppl::common::RC_SUCCESS;
    } else if(output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
        int in_width;
        if(input_shape->GetDimCount() == 4){
            in_width     = input_shape->GetDim(3);
        } else if(input_shape->GetDimCount() == 3) {
            in_width = 1;
        } else {
            return ppl::common::RC_UNSUPPORTED;
        }
        int batch = output_shape->GetDim(0);
        int pad_channels = output_shape->GetDim(1) + output_shape->GetPadding0(1) + output_shape->GetPadding1(1);
        int in_height = input_shape->GetDim(2);
        int in_hw = in_height * in_width;
        dim3 dim_block(32, 8, 1); // (c, hw, 1)
        int padChannelsDivide = pad_channels;
        int channel_blocks    = (padChannelsDivide + dim_block.x - 1) / dim_block.x;
        constexpr int block_threshold = 64;
        constexpr int hw_threshold = 128;
        if (channel_blocks * batch < block_threshold && in_hw > hw_threshold) {
            dim3 dim_block(8, 32, 1); // (c, hw, 1)
            int channel_blocks    = (padChannelsDivide + dim_block.x - 1) / dim_block.x;
            dim3 dim_grid(channel_blocks, 1, batch);
            ppl_cukernel_reduceSum_nhwc_fp32<8, 32><<<dim_grid,
                                                dim_block,
                                                0,
                                                stream>>>((const float*)input, (float*)output, batch, padChannelsDivide, in_hw);
        } else {
            dim3 dim_grid(channel_blocks, 1, batch);
            ppl_cukernel_reduceSum_nhwc_fp32<32, 8><<<dim_grid,
                                                dim_block,
                                                0,
                                                stream>>>((const float*)input, (float*)output, batch, padChannelsDivide, in_hw);
        }
        return ppl::common::RC_SUCCESS;
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
}

ppl::common::RetCode PPLCUDAReduceOPImp_NHWC(
    cudaStream_t stream,
    ReduceParam param,
    PPLReduceDimDes des,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output,
    const QuantKernelParamCuda* qparam)
{
    //need use nhwc mode
    ReduceMode mode = pplGetReduceMode(des);
    if(0 == qparam->i_zero_point && 0 == qparam->o_zero_point) {
        if(3 == condition_nhwc(input_shape, output_shape,des,mode)) {
            switch(param){
                case ReduceMean:
                    return PPLCUDAGlobalAvePoolingForwardImp(stream, (ppl::common::TensorShape*)input_shape, input, (ppl::common::TensorShape*)output_shape, output, qparam->i_step, qparam->o_step);
                    break;
                case ReduceSum:
                    return PPLCUDAReduceSumImp_NHWC(stream, (ppl::common::TensorShape*)input_shape, input, (ppl::common::TensorShape*)output_shape, output, qparam->i_step, qparam->o_step);
                    break;
                default:
                    return ppl::common::RC_UNSUPPORTED;
            }
        }
        else if(1 == condition_nhwc(input_shape, output_shape, des, mode))
        {
            dim3 dim_grid(1, 1, input_shape->GetDim(0));
            dim3 dim_block(32, 4, 1);
            int i_stride = 0, o_stride = 0;
            int channels = 0, pad_channels = 0, HW = 0;
            int branch = 1;
            if(input_shape->GetDimCount() == 3)
            {
                if(output_shape->GetDimCount() == 3)
                {
                    i_stride = input_shape->GetDim(1) + input_shape->GetPadding0(1) + input_shape->GetPadding1(1);
                    o_stride = output_shape->GetDim(1) + output_shape->GetPadding0(1) + output_shape->GetPadding1(1);
                    pad_channels = input_shape->GetDim(2);
                    channels = pad_channels;
                    HW = input_shape->GetDim(1);
                    dim_grid.y = (input_shape->GetDim(2) + dim_block.y - 1) / dim_block.y;
                }
                else if(output_shape->GetDimCount() == 2)
                {
                    i_stride = input_shape->GetDim(1) + input_shape->GetPadding0(1) + input_shape->GetPadding1(1);
                    o_stride = output_shape->GetDim(1) + output_shape->GetPadding0(1) + output_shape->GetPadding1(1);
                    channels = input_shape->GetDim(2);
                    pad_channels = input_shape->GetDim(2) + input_shape->GetPadding0(2) + input_shape->GetPadding1(2);
                    dim_grid.y = (pad_channels + dim_block.y - 1) / dim_block.y;
                    HW = input_shape->GetDim(1);
                    branch = 0;
                }
                else
                {
                    return ppl::common::RC_UNSUPPORTED;
                }
            }
            else if(output_shape->GetDimCount() == 4 && input_shape->GetDimCount() == 4)
            {
                i_stride = input_shape->GetDim(1) + input_shape->GetPadding0(1) + input_shape->GetPadding1(1);
                o_stride = output_shape->GetDim(1) + output_shape->GetPadding0(1) + output_shape->GetPadding1(1);
                pad_channels = input_shape->GetDim(2) * input_shape->GetDim(3);
                channels = pad_channels;
                HW = input_shape->GetDim(1);
                dim_grid.y = (pad_channels + dim_block.y - 1) / dim_block.y;
            }
            else if(input_shape->GetDimCount() == 5)
            {
                i_stride = input_shape->GetDim(1) + input_shape->GetPadding0(1) + input_shape->GetPadding1(1);
                o_stride = output_shape->GetDim(1) + output_shape->GetPadding0(1) + output_shape->GetPadding1(1);
                pad_channels = input_shape->GetDim(2) * input_shape->GetDim(3) * input_shape->GetDim(4);
                channels = pad_channels;
                HW = input_shape->GetDim(1);        //HW means DHW when input_dimcout = 5
                dim_grid.y = (pad_channels + dim_block.y - 1) / dim_block.y;
            }
            else
            {
                return ppl::common::RC_UNSUPPORTED;
            }
            if(output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16)
            {
                if(branch==1)
                {
                    ppl_cukernel_reduce_m3_nhwc<half, 1><<<dim_grid, dim_block, 0, stream>>>((const half*)input, (half*)output, channels,
                        pad_channels, HW, i_stride, o_stride, param);
                }
                else
                {
                    ppl_cukernel_reduce_m3_nhwc<half, 0><<<dim_grid, dim_block, 0, stream>>>((const half*)input, (half*)output, channels,
                        pad_channels, HW, i_stride, o_stride, param);
                }
                return ppl::common::RC_SUCCESS;
            }
            else if(output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32)
            {
                if(branch == 1)
                {
                    ppl_cukernel_reduce_m3_nhwc<float, 1><<<dim_grid, dim_block, 0, stream>>>((const float*)input, (float*)output, channels,
                        pad_channels, HW,i_stride,o_stride, param);
                } else {
                    ppl_cukernel_reduce_m3_nhwc<float, 0><<<dim_grid, dim_block, 0, stream>>>((const float*)input, (float*)output, channels,
                        pad_channels, HW,i_stride,o_stride, param);
                }
                return ppl::common::RC_SUCCESS;
            } else if(output_shape->GetDataType() == ppl::common::DATATYPE_INT8)
            {
                if(branch == 1)
                {
                    ppl_cukernel_reduce_m3_nhwc_int8<1><<<dim_grid, dim_block, 0, stream>>>((const int8_t*)input, (int8_t*)output, channels,
                        pad_channels, HW,i_stride,o_stride,
                        qparam->i_step, qparam->o_step, param);
                }
                else
                {
                    ppl_cukernel_reduce_m3_nhwc_int8<0><<<dim_grid, dim_block, 0, stream>>>((const int8_t*)input, (int8_t*)output, channels,
                        pad_channels, HW,i_stride,o_stride,
                        qparam->i_step, qparam->o_step,param);
                }
                return ppl::common::RC_SUCCESS;
            }
        }
        else if(2 == condition_nhwc(input_shape, output_shape, des, mode))
        {
            if(input_shape->GetDimCount() == 5){
                int i_stride = input_shape->GetDim(1) + input_shape->GetPadding0(1) + input_shape->GetPadding1(1);
                int DHW = i_stride;
                int channels = input_shape->GetDim(2) * input_shape->GetDim(3) * input_shape->GetDim(4);
                int pad_channels = channels;

                int o_stride = DHW * channels;
                dim3 dim_Block(256);
                dim3 dim_Grid((o_stride + 255) / 256);

                if(output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16)
                {
                    channels = (channels + 7) / 8;
                    pad_channels = (pad_channels + 7) / 8;
                    if(DHW % 8 == 0){
                        ppl_cukernel_reduce_col_nhwc_5D_opt<half, float, 8><<<dim_Grid, dim_Block, 0, stream>>>(
                        (const half*)input, (half*)output, des.n_reduce, DHW, channels, pad_channels, o_stride, param);
                        return ppl::common::RC_SUCCESS;
                    }
                    ppl_cukernel_reduce_col_nhwc_5D<half, float><<<dim_Grid, dim_Block, 0, stream>>>(
                        (const half*)input, (half*)output, des.n_reduce, DHW, channels, pad_channels, o_stride, param);
                }
                else if(output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32){
                    ppl_cukernel_reduce_col_nhwc_5D<float, float><<<dim_Grid, dim_Block, 0, stream>>>(
                        (const float*)input, (float*)output, des.n_reduce, DHW, channels, pad_channels, o_stride, param);
                }
                else{
                    return ppl::common::RC_UNSUPPORTED;
                }
                return ppl::common::RC_SUCCESS;
            }
            if(output_shape->GetDimCount() != 2) return ppl::common::RC_UNSUPPORTED;
            int i_stride = input_shape->GetDim(1) + input_shape->GetPadding0(1) + input_shape->GetPadding1(1);
            int o_stride = output_shape->GetDim(1) + output_shape->GetPadding0(1) + output_shape->GetPadding1(1);

            dim3 dimGrid((des.n_inner + 255) / 256, des.n_outer,1);
            dim3 dimBlock(256);
            int pad_channels = input_shape->GetDim(2) + input_shape->GetPadding0(2) + input_shape->GetPadding1(2);

            if(output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16)
            {
                ppl_cukernel_reduce_col_nhwc<half,float><<<dimGrid, dimBlock, 0, stream>>>(
                    (const half*)input, (half*)output, des.n_outer, des.n_reduce, 1, output_shape->GetDim(0), i_stride, o_stride, param);
            }
            else if(output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32)
            {
                ppl_cukernel_reduce_col_nhwc<float,float><<<dimGrid, dimBlock, 0, stream>>>(
                    (const float*)input, (float*)output, des.n_outer, des.n_reduce, 1, output_shape->GetDim(0), i_stride, o_stride, param);
            }
            else
            {
                return ppl::common::RC_UNSUPPORTED;
            }
            return ppl::common::RC_SUCCESS;
        }
        else if(mode == 1)
        {
            int block_size = 512;
            int num_elements = 0, channels, pad_channels;
            if(input_shape->GetDimCount() == 3)
            {
                num_elements = input_shape->GetDim(0) * input_shape->GetDim(1) * input_shape->GetDim(2);
                channels = input_shape->GetDim(1);
                pad_channels = input_shape->GetDim(1) + input_shape->GetPadding0(1) + input_shape->GetPadding1(1);
            }
            else if(input_shape->GetDimCount() == 4)
            {
                num_elements = input_shape->GetDim(0) * input_shape->GetDim(1) * input_shape->GetDim(2) * input_shape->GetDim(3);
                channels = input_shape->GetDim(1);
                pad_channels = input_shape->GetDim(1) + input_shape->GetPadding0(1) + input_shape->GetPadding1(1);
            }
            else
            {
                return ppl::common::RC_UNSUPPORTED;
            }
            int grid_size = 1;
            if(output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16)
            {
                ppl_reduce_all_nhwc<half><<<grid_size, block_size,0,stream>>>(num_elements,(const half *)input, (half*)output,channels, pad_channels,param);
                return ppl::common::RC_SUCCESS;
            }
            else if(output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32)
            {
                ppl_reduce_all_nhwc<float><<<grid_size, block_size,0,stream>>>(num_elements,(const float *)input, (float*)output,channels, pad_channels,param);
                return ppl::common::RC_SUCCESS;
            }
            else if(output_shape->GetDataType() == ppl::common::DATATYPE_INT8)
            {
                ppl_reduce_all_nhwc_int8<<<grid_size, block_size,0,stream>>>(num_elements,(const int8_t*)input, (int8_t*)output,channels, pad_channels, qparam->i_step, qparam->o_step,param);
                return ppl::common::RC_SUCCESS;
            }
        }
    }
    return ppl::common::RC_UNSUPPORTED;
}

ppl::common::RetCode PPLCUDAReduceForwardImp_NHWC(
    cudaStream_t stream,
    ReduceParam param,
    PPLReduceDimDes des,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const ppl::common::TensorShape* output_shape,
    void* output,
    void* tmp_buffer,
    const QuantKernelParamCuda* qparam)
{
    ppl::common::RetCode status = ppl::common::RC_SUCCESS;
    switch (param) {
        case ReduceMean:
        case ReduceSum:
            status = PPLCUDAReduceOPImp_NHWC(stream, param, des, input_shape, input, output_shape, output, qparam);
            break;
        default:
            status = ppl::common::RC_UNSUPPORTED;
        }
    return status;
}

// void printtoHost(half *dev,int batchsize,int channel,int height,int width, int format)
// {
//     half *cpu = nullptr;
//     cudaMallocHost((void**)&cpu,batchsize*channel*width*height*sizeof(half));
//     cudaMemcpy(cpu, dev,sizeof(half)*batchsize*channel*width*height, cudaMemcpyDeviceToHost);
//     if(format == 1){
//         for(int b = 0; b < batchsize; b++){
//             for(int c = 0; c < channel; c++){
//                 for(int h = 0; h < height; h++){
//                     for(int w = 0; w < width; w++){
//                         printf("%f,",(float)cpu[b*channel*height*width + c*height*width + h*width + w]);
//                     }
//                     printf("\n");
//                 }
//                 printf("=======next channel=====\n");
//             }
//             printf("=======next batch=====\n");
//         }
//     }
//     else{
//         for(int b = 0; b < batchsize; b++){
//             for(int c = 0; c < channel; c++){
//                 for(int h = 0; h < height; h++){
//                     for(int w = 0; w < width; w++){
//                         printf("%f,",(float)cpu[b*channel*height*width + (h*width + w)*channel + c]);
//                     }
//                     printf("\n");
//                 }
//                 printf("=======next channel=====\n");
//             }
//             printf("=======next batch=====\n");
//         }
//     }
//     cudaFreeHost((void*)cpu);
// }
