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

#include "cudakernel/arithmetic/fused_arithmetic.h"
#include "cudakernel/arithmetic/arithmetic.h"
#include "cudakernel/common/divmod_fast.h"
#include "ppl/common/types.h"
#include "ppl/common/log.h"
#include <cuda_fp16.h>
#include <algorithm>

#define DivUp(x,y) (((x) + (y) - 1) / (y))

enum FusedArithmeticOpType {
    FusedArithmetic_Unknown = 0,
    FusedArithmetic_MulAdd,
    FusedArithmetic_ForceWord = INT_MAX,
};

template<FusedArithmeticOpType op_type, typename T>
__device__ inline T ppl_fused_arithmetic_scalar(T a, T b, T c, int with_act);

#define MulAddScalar(T) \
template<> __device__ inline T ppl_fused_arithmetic_scalar<FusedArithmetic_MulAdd, T>(T a, T b, T c, int with_act) { \
    if (with_act == 1){   \
        T out_val = a * b + c;   \
        T res = (out_val > 0) ? out_val : 0;    \
        return res; \
    } else \
        return a * b + c;\
}
MulAddScalar(double);
MulAddScalar(float);
MulAddScalar(int64_t);
MulAddScalar(int32_t);
MulAddScalar(int16_t);
MulAddScalar(uint64_t);
MulAddScalar(uint32_t);
MulAddScalar(uint16_t);
template<> __device__ inline half ppl_fused_arithmetic_scalar<FusedArithmetic_MulAdd, half>(half a, half b, half c, int with_act) {
    #if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
        if (with_act == 1){
            half out_val = __hadd(__hmul(a, b), c);
            half res = __hgt(out_val, 0) ? out_val : half(0);
            return res;
        }
        else
            return __hadd(__hmul(a, b), c);
        // return __hfma(a, b, c);
    #else
        return 0;
    #endif
}
template<> __device__ inline int8_t ppl_fused_arithmetic_scalar<FusedArithmetic_MulAdd, int8_t>(int8_t a, int8_t b, int8_t c, int with_act) {
    // ppl_fused_arithmetic_scalar_int8_no_scale
    int res = a * b + c;
    if (with_act == 1){
        if (res > 127) res = 127;
        else if(res < 0) res = 0;
        return res;
    } else {
        if(res > 127) res = 127;
        else if(res < -128) res = -128;
        return res;
    }
}
template<> __device__ inline uint8_t ppl_fused_arithmetic_scalar<FusedArithmetic_MulAdd, uint8_t>(uint8_t a, uint8_t b, uint8_t c, int with_act) {
    // ppl_fused_arithmetic_scalar_uint8_no_scale
    int res = a * b + c;
    if(res > 255) res = 255;
    else if(res < 0) res = 0;
    return res;
}

template<FusedArithmeticOpType op_type, typename T>
__device__ inline T ppl_fused_arithmetic_scalar_int8(T a, T b, T c, float mul_scale, float add_scale, int with_act);
template<> __device__ inline int8_t ppl_fused_arithmetic_scalar_int8<FusedArithmetic_MulAdd, int8_t>(int8_t a, int8_t b, int8_t c, float mul_scale, float add_scale, int with_act) {
    int res = round((a * b * mul_scale) + c * add_scale);
    if (with_act == 1){
        if (res > 127) res = 127;
        else if(res < 0) res = 0;
        return res;
    } else {
        if(res > 127) res = 127;
        else if(res < -128) res = -128;
        return res;
    }
}

#define GET_SHAPE_PARAM    \
    int input_dims0 = input_shape0->GetDimCount();  \
    int input_dims1 = input_shape1->GetDimCount();  \
    int input_dims2 = input_shape2->GetDimCount();

int fused_opt_4x4(const ppl::common::TensorShape *input_shape0, 
                    const ppl::common::TensorShape *input_shape1, 
                    const ppl::common::TensorShape *input_shape2){
    GET_SHAPE_PARAM
    if (input_dims0 == 4 && input_dims2 == 4 && input_dims1 > 2){
        if (input_shape2->GetDim(3) == input_shape0->GetDim(3) && input_shape1->GetDim(input_dims1 - 1) == 1)
            if (input_shape2->GetDim(2) == input_shape0->GetDim(2) && input_shape1->GetDim(input_dims1 - 2) == 1)
                if (input_shape2->GetDim(1) == input_shape0->GetDim(1) && input_shape1->GetDim(input_dims1 - 3) == input_shape0->GetDim(1)) {
                    if (input_dims1 == 4 && input_shape2->GetDim(0) == input_shape0->GetDim(0) && input_shape1->GetDim(0) == input_shape0->GetDim(0))
                        return 1;
                    if (input_shape2->GetDim(0) == 1 && (input_dims1 == 3 || (input_dims1 == 4 && input_shape1->GetDim(0) == 1)))
                        return 2;
                }
    }
    return 0;
}

bool fused_opt_4x1(const ppl::common::TensorShape *input_shape0, 
                    const ppl::common::TensorShape *input_shape1, 
                    const ppl::common::TensorShape *input_shape2){
    GET_SHAPE_PARAM
    if (input_dims0 == 4 && input_dims1 == 1 && input_dims2 == 4)
        if (input_shape0->GetDim(0) == input_shape2->GetDim(0))
            if (input_shape0->GetDim(1) == input_shape2->GetDim(1))
                if (input_shape0->GetDim(2) == input_shape2->GetDim(2))
                    if (input_shape0->GetDim(3) == input_shape2->GetDim(3))
                        return true;
    return false;
}

bool fused_opt_common1(const ppl::common::TensorShape *input_shape0, 
                    const ppl::common::TensorShape *input_shape1, 
                    const ppl::common::TensorShape *input_shape2){
    GET_SHAPE_PARAM
    if (input_dims0 == input_dims1 && input_dims1 == input_dims2)
        if (input_shape1->GetDim(0) == input_shape2->GetDim(0) && input_shape1->GetDim(0) == 1){
            int truedim = 1;
            for (int i = 1; i < input_dims0; i++) 
                if (input_shape0->GetDim(i) == input_shape1->GetDim(i) && input_shape0->GetDim(i) == input_shape2->GetDim(i))
                    truedim++;
            if (truedim == input_dims0)
                return true;
        }
    return false;
}

bool fused_opt_common2(const ppl::common::TensorShape *input_shape0, 
                    const ppl::common::TensorShape *input_shape1, 
                    const ppl::common::TensorShape *input_shape2){
    GET_SHAPE_PARAM
    if (input_dims0 == input_dims2 && input_dims1 == 1)
        if (input_shape0->GetDim(0) == input_shape2->GetDim(0) && input_shape1->GetDim(0) == 1){
            int truedim = 1;
            for (int i = 1; i < input_dims0; i++) {
                if (input_shape0->GetDim(i) == input_shape2->GetDim(i)){
                    truedim++;
                }
            }   
            if (truedim == input_dims0)
                return true;
        }
    return false;
}

#define GetMulAdd(Doutput, Dinput0, Dinput1, Dinput2)   \
    if (same_scale){    \
        Doutput = ppl_fused_arithmetic_scalar<op_type, T>(Dinput0, Dinput1, Dinput2, with_act); \
    }   \
    else{   \
        Doutput = ppl_fused_arithmetic_scalar_int8<op_type, int8_t>(Dinput0, Dinput1, Dinput2, mul_scale, add_scale, with_act); \
    }

template<FusedArithmeticOpType op_type, typename T, typename PackT, int PackSize>
__global__ void ppl_cukernel_fused_arithmetic_4x4_aba_nhwc_nopadding(
    const int hw, const int hw_inter, const int hw_outer, const int channel, DivModFast groupfast, int product_b,
    const T *input0, const T* input1, T *output, bool same_scale, float mul_scale, float add_scale, int with_act) {

    int tid = threadIdx.x;
    int batch_idx = blockIdx.x;

    __shared__ T sharedmul[PackSize * 256];
    int c_len = blockDim.x * PackSize;
    int c_start = blockIdx.z * c_len;
    int b_idx = 0;
    if (product_b == 1)  b_idx = batch_idx;
    else if (product_b == 2)  b_idx = 0;
    auto input1_start = input1 + b_idx * channel + c_start;
    for (int c_inner_idx = tid; c_inner_idx < c_len && c_start + c_inner_idx < channel; c_inner_idx += blockDim.x)
        sharedmul[c_inner_idx] = input1_start[c_inner_idx]; 
    __syncthreads();
    
    union Package {
        PackT packed;
        T     unpacked[PackSize];
    };
    int g_idx, hw_inidx;
    groupfast.divmod(c_start + tid, hw_inidx, g_idx);
    int hw_idx = blockIdx.y * hw_inter + hw_inidx;
    if (hw_idx < hw){
        int c_idx = g_idx * PackSize;
        int hwc_idx = hw_idx * channel + c_idx;
        int index = batch_idx * hw * channel + hwc_idx;
        auto cur_in = reinterpret_cast<const PackT*>(input0 + index);
        auto cur_out = reinterpret_cast<PackT*>(output + index);
        Package p, o;
        p.packed = __ldg(cur_in);
        if (c_idx >= c_len) c_idx -= c_len;
        for (int i = 0; i < PackSize; i++){
            T mul_val = sharedmul[c_idx + i];
            GetMulAdd(o.unpacked[i], p.unpacked[i], mul_val, p.unpacked[i])
        }
        cur_out[0] = o.packed;
    }
}

template<FusedArithmeticOpType op_type, typename T, typename PackT, int PackSize>
__global__ void ppl_cukernel_fused_arithmetic_4x4_aba_nhwcp(
    const int hw, const int hw_inter, const int hw_outer, const int channel, 
    const int cp0, const int cp1, const int ocp, DivModFast groupfast, int product_b,
    const T *input0, const T* input1, T *output, bool same_scale, float mul_scale, float add_scale, int with_act) {

    int tid = threadIdx.x;
    int batch_idx = blockIdx.x;

    __shared__ T sharedmul[PackSize * 256];

    int c_len = blockDim.x * PackSize;
    int c_start = blockIdx.z * c_len;
    int b_idx = 0;
    if (product_b == 1) b_idx = batch_idx;
    else if (product_b == 2) b_idx = 0;
    auto input1_start = input1 + b_idx * cp1 + c_start;
    for (int c_inner_idx = tid; c_inner_idx < c_len && c_start + c_inner_idx < channel; c_inner_idx += blockDim.x)
        sharedmul[c_inner_idx] = input1_start[c_inner_idx]; 
    __syncthreads();
    
    union Package {
        PackT packed;
        T     unpacked[PackSize];
    };
    int g_idx, hw_inidx;
    groupfast.divmod(c_start + tid, hw_inidx, g_idx);
    int hw_idx = blockIdx.y * hw_inter + hw_inidx;
    if (hw_idx < hw){
        int cp_idx = g_idx * PackSize;
        int hwc_idx = hw_idx * cp0 + cp_idx;
        int index = batch_idx * hw * cp0 + hwc_idx;
        auto cur_in = reinterpret_cast<const PackT*>(input0 + index);
        if (cp_idx < ocp){
            int out_idx = batch_idx * hw * ocp + hw_idx * ocp + cp_idx;
            auto cur_out = reinterpret_cast<PackT*>(output + out_idx);
            Package p, o;
            p.packed = __ldg(cur_in);
            if (cp_idx >= c_len) cp_idx -= c_len;
            for (int i = 0; i < PackSize; i++){
                int c_idx = cp_idx + i;
                if (c_idx < ocp){
                    T mul_val = sharedmul[c_idx];
                    GetMulAdd(o.unpacked[i], p.unpacked[i], mul_val, p.unpacked[i])
                } else break;
            }
            cur_out[0] = o.packed;
        }
    }
}

template<FusedArithmeticOpType op_type, typename T, typename PackT, int PackSize>
__global__ void ppl_cukernel_fused_arithmetic_4x1_aba_nhwc_nopadding(
    const int hw, const int hw_inter, const int hw_outer, const int channel, DivModFast groupfast,
    const T *input0, const T* input1, T *output, bool same_scale, float mul_scale, float add_scale, int with_act) {

    int tid = threadIdx.x;
    int batch_idx = blockIdx.x;
    T mul_val = __ldg(input1);
    
    union Package {
        PackT packed;
        T     unpacked[PackSize];
    };
    int g_idx, hw_inidx;
    groupfast.divmod(tid, hw_inidx, g_idx);
    int hw_idx = blockIdx.y * hw_inter + hw_inidx;
    if (hw_inidx < hw_inter && hw_idx < hw){
        int c_idx = g_idx * PackSize;
        int hwc_idx = hw_idx * channel + c_idx;
        int index = batch_idx * hw * channel + hwc_idx;
        auto cur_in = reinterpret_cast<const PackT*>(input0 + index);
        auto cur_out = reinterpret_cast<PackT*>(output + index);
        Package p, o;
        p.packed = __ldg(cur_in);
        for (int i = 0; i < PackSize; i++){
            GetMulAdd(o.unpacked[i], p.unpacked[i], mul_val, p.unpacked[i])
        }
        cur_out[0] = o.packed;
    }
}

template<FusedArithmeticOpType op_type, typename T, typename PackT, int PackSize>
__global__ void ppl_cukernel_fused_arithmetic_4x1_aba_nhwcp(
    const int hw, const int hw_inter, const int hw_outer, const int channel,
    const int cp0, const int ocp, DivModFast groupfast,
    const T *input0, const T* input1, T *output, bool same_scale, float mul_scale, float add_scale, int with_act) {

    int tid = threadIdx.x;
    int batch_idx = blockIdx.x;
    T mul_val = __ldg(input1);
    
    union Package {
        PackT packed;
        T     unpacked[PackSize];
    };
    int g_idx, hw_inidx;
    groupfast.divmod(tid, hw_inidx, g_idx);
    int hw_idx = blockIdx.y * hw_inter + hw_inidx;
    if (hw_inidx < hw_inter && hw_idx < hw){
        int cp_idx = g_idx * PackSize;
        int hwc_idx = hw_idx * cp0 + cp_idx;
        int index = batch_idx * hw * cp0 + hwc_idx;
        auto cur_in = reinterpret_cast<const PackT*>(input0 + index);
        if (cp_idx < ocp){
            int out_idx = batch_idx * hw * ocp + hw_idx * ocp + cp_idx;
            auto cur_out = reinterpret_cast<PackT*>(output + out_idx);
            Package p, o;
            p.packed = __ldg(cur_in);
            for (int i = 0; i < PackSize; i++){
                int c_idx = cp_idx + i;
                if (c_idx < ocp){
                    GetMulAdd(o.unpacked[i], p.unpacked[i], mul_val, p.unpacked[i])
                } else break;
            }
            cur_out[0] = o.packed;
        }
    }
}

template<FusedArithmeticOpType op_type, typename T, typename PackT, int PackSize>
__global__ void ppl_cukernel_fused_arithmetic_4x4_nhwc_nopadding(
    const int hw, const int hw_inter, const int hw_outer, const int channel, DivModFast groupfast, int product_b,
    const T *input0, const T* input1, const T* input2, T *output, bool same_scale, float mul_scale, float add_scale, int with_act) {

    int tid = threadIdx.x;
    int batch_idx = blockIdx.x;
    
    __shared__ T sharedmul[PackSize * 256];
    
    int c_len = blockDim.x * PackSize;
    int c_start = blockIdx.z * c_len;
    int b_idx = 0;
    if (product_b == 1) b_idx = batch_idx;
    else if (product_b == 2) b_idx = 0;
    auto input1_start = input1 + b_idx * channel + c_start;

    for (int c_inner_idx = tid; c_inner_idx < c_len && c_start + c_inner_idx < channel; c_inner_idx += blockDim.x)
        sharedmul[c_inner_idx] = input1_start[c_inner_idx]; 
    __syncthreads();
    
    union Package {
        PackT packed;
        T     unpacked[PackSize];
    };
    int g_idx, hw_inidx;
    groupfast.divmod(c_start + tid, hw_inidx, g_idx);
    int hw_idx = blockIdx.y * hw_inter + hw_inidx;
    if (hw_idx < hw){
        int c_idx = g_idx * PackSize;
        int hwc_idx = hw_idx * channel + c_idx;
        int index = batch_idx * hw * channel + hwc_idx;
        auto cur_in1 = reinterpret_cast<const PackT*>(input0 + index);
        auto cur_in2 = reinterpret_cast<const PackT*>(input2 + index);
        auto cur_out = reinterpret_cast<PackT*>(output + index);
        Package p1, p2, o;
        p1.packed = __ldg(cur_in1);
        p2.packed = __ldg(cur_in2);
        if (c_idx >= c_len) c_idx -= c_len;
        for (int i = 0; i < PackSize; i++){
            T mul_val = sharedmul[c_idx + i];
            GetMulAdd(o.unpacked[i], p1.unpacked[i], mul_val, p2.unpacked[i])
        }
        cur_out[0] = o.packed;
    }
}

template<FusedArithmeticOpType op_type, typename T, typename PackT, int PackSize>
__global__ void ppl_cukernel_fused_arithmetic_4x1_nhwc_nopadding(
    const int hw, const int hw_inter, const int hw_outer, const int channel, DivModFast groupfast,
    const T *input0, const T* input1, const T* input2, T *output, bool same_scale, float mul_scale, float add_scale, int with_act) {
    int tid = threadIdx.x;
    int batch_idx = blockIdx.x;
    T mul_val = __ldg(input1);
    union Package {
        PackT packed;
        T     unpacked[PackSize];
    };
    int g_idx, hw_inidx;
    groupfast.divmod(tid, hw_inidx, g_idx);
    int hw_idx = blockIdx.y * hw_inter + hw_inidx;
    if (hw_inidx < hw_inter && hw_idx < hw){
        int c_idx = g_idx * PackSize;
        int hwc_idx = hw_idx * channel + c_idx;
        int index = batch_idx * hw * channel + hwc_idx;
        auto cur_in1 = reinterpret_cast<const PackT*>(input0 + index);
        auto cur_in2 = reinterpret_cast<const PackT*>(input2 + index);
        auto cur_out = reinterpret_cast<PackT*>(output + index);
        Package p1, p2, o;
        p1.packed = __ldg(cur_in1);
        p2.packed = __ldg(cur_in2);
        for (int i = 0; i < PackSize; i++){
            GetMulAdd(o.unpacked[i], p1.unpacked[i], mul_val, p2.unpacked[i])
        }
        cur_out[0] = o.packed;
    }
}

template<FusedArithmeticOpType op_type, typename T, typename PackT, int PackSize>
__global__ void ppl_cukernel_fused_arithmetic_common_opt(
    const uint64_t num_elems, const int dim_count, const int batchsize,
    const T *input0, const T* input1, const T* input2, T *output, bool same_scale, float mul_scale, float add_scale, int with_act) {

    int inner_idx = (blockIdx.y * blockDim.x + threadIdx.x) * PackSize;
    if (inner_idx >= batchsize) return;
    int batch_idx = blockIdx.x;
    if (batchsize - inner_idx >= PackSize){
        int index = batch_idx * batchsize + inner_idx;
        auto cur_in0 = reinterpret_cast<const PackT*>(input0 + index);
        auto cur_in1 = reinterpret_cast<const PackT*>(input1 + inner_idx);
        auto cur_in2 = reinterpret_cast<const PackT*>(input2 + inner_idx);
        auto cur_out = reinterpret_cast<PackT*>(output + index);
        union Package {
            PackT packed;
            T     unpacked[PackSize];
        };
        Package p0, p1, p2, o;
        p0.packed = __ldg(cur_in0);
        p1.packed = __ldg(cur_in1);
        p2.packed = __ldg(cur_in2);
        for (int i = 0; i < PackSize; i++){
            GetMulAdd(o.unpacked[i], p0.unpacked[i], p1.unpacked[i], p2.unpacked[i])
        }
        cur_out[0] = o.packed;
    } else {
        int index = batch_idx * batchsize + inner_idx;
        GetMulAdd(output[index], input0[index], input1[inner_idx], input2[inner_idx])
    }
}

template<FusedArithmeticOpType op_type, typename T, typename PackT, int PackSize>
__global__ void ppl_cukernel_fused_arithmetic_common_opt2(
    const uint64_t num_elems, const int dim_count, const int batchsize,
    const T *input0, const T* input1, const T* input2, T *output, bool same_scale, float mul_scale, float add_scale, int with_act) {

    int inner_idx = (blockIdx.y * blockDim.x + threadIdx.x) * PackSize;
    int batch_idx = blockIdx.x;
    int index = batch_idx * batchsize + inner_idx;
    if (index >= num_elems) return;
    T mul_val = __ldg(input1);
    if (num_elems - index >= PackSize){
        auto cur_in0 = reinterpret_cast<const PackT*>(input0 + index);
        auto cur_in2 = reinterpret_cast<const PackT*>(input2 + index);
        auto cur_out = reinterpret_cast<PackT*>(output + index);
        union Package {
            PackT packed;
            T     unpacked[PackSize];
        };
        Package p0, p2, o;
        p0.packed = __ldg(cur_in0);
        p2.packed = __ldg(cur_in2);
        for (int i = 0; i < PackSize; i++){
            GetMulAdd(o.unpacked[i], p0.unpacked[i], mul_val, p2.unpacked[i])
        }
        cur_out[0] = o.packed;
    } else {
        GetMulAdd(output[index], input0[index], mul_val, input2[index])
    }
}

template<FusedArithmeticOpType op_type, typename T, typename PackT, int PackSize>
__global__ void ppl_cukernel_fused_arithmetic_common_opt3(
    const uint64_t num_elems, const int dim_count, const int channel, const int hw_count, int product_b,
    const T *input0, const T* input1, const T* input2, T *output, bool same_scale, float mul_scale, float add_scale, int with_act) {

    int inner_idx = (blockIdx.y * blockDim.x + threadIdx.x) * PackSize; 
    if (inner_idx >= hw_count) return;

    int c_idx = blockIdx.z;
    int bc_start = blockIdx.x * channel * hw_count;
    int c_start = c_idx * hw_count;
    int input0_idx = 0;
    int input1_idx = 0;
    int input2_idx = 0;
    
    if (product_b == 1) {
        input0_idx = bc_start + c_start + inner_idx;
        input1_idx = blockIdx.x * channel + c_idx;
        input2_idx = input0_idx;
    } else if (product_b == 2) {
        input1_idx = c_idx;
        input2_idx = c_start + inner_idx;
        input0_idx = bc_start + input2_idx;
    }
    T mul_val = __ldg(input1 + input1_idx);
    if (hw_count - inner_idx >= PackSize){
        auto cur_in0 = reinterpret_cast<const PackT*>(input0 + input0_idx);
        auto cur_in2 = reinterpret_cast<const PackT*>(input2 + input2_idx);
        auto cur_out = reinterpret_cast<PackT*>(output + input0_idx);
        union Package {
            PackT packed;
            T     unpacked[PackSize];
        };
        Package p0, p2, o;
        p0.packed = __ldg(cur_in0);
        p2.packed = __ldg(cur_in2);
        for (int i = 0; i < PackSize; i++){
            GetMulAdd(o.unpacked[i], p0.unpacked[i], mul_val, p2.unpacked[i])
        }
        cur_out[0] = o.packed;
    } else {
        GetMulAdd(output[input0_idx], input0[input0_idx], mul_val, input2[input2_idx])
    }
}

#define USE_COMMON_OPT(T, PackT, PackSize)    \
    int batch_num = 1;      \
    for (int i = 1; i < dim_count; i++) batch_num *= input_shape0->GetDim(i);   \
    int blockY = DivUp(batch_num, blocksize * PackSize);    \
    gridsize.y = blockY;    \
    ppl_cukernel_fused_arithmetic_common_opt<FusedArithmetic_MulAdd, T, PackT, PackSize><<<gridsize, blocksize>>>(  \
        num_elems, dim_count, batch_num, (const T*) input0, (const T*) input1, (const T*) input2, (T*) output, true, 1.f, 1.f, act_type);

#define USE_COMMON_INT8_OPT(T, PackT, PackSize)    \
    int batch_num = 1;      \
    for (int i = 1; i < dim_count; i++) batch_num *= input_shape0->GetDim(i);   \
    float mul_scale = in_scale0 * in_scale1 / out_scale;    \
    float add_scale = in_scale2 / out_scale;                \
    int blockY = DivUp(batch_num, blocksize * PackSize);    \
    gridsize.y = blockY;    \
    ppl_cukernel_fused_arithmetic_common_opt<FusedArithmetic_MulAdd, int8_t, PackT, PackSize><<<gridsize, blocksize>>>(  \
        num_elems, dim_count, batch_num, (const int8_t*) input0, (const int8_t*) input1, \
        (const int8_t*) input2, (int8_t*) output, false, mul_scale, add_scale, act_type);

#define USE_COMMON_OPT2(T, PackT, PackSize)    \
    int batch_num = 1;      \
    for (int i = 1; i < dim_count; i++) batch_num *= input_shape0->GetDim(i);   \
    gridsize.y = DivUp(batch_num, blocksize * PackSize);    \
    ppl_cukernel_fused_arithmetic_common_opt2<FusedArithmetic_MulAdd, T, PackT, PackSize><<<gridsize, blocksize>>>(  \
        num_elems, dim_count, batch_num, (const T*) input0, (const T*) input1, (const T*) input2, (T*) output, true, 1.f, 1.f, act_type);

#define USE_COMMON_INT8_OPT2(T, PackT, PackSize)    \
    int batch_num = 1;      \
    for (int i = 1; i < dim_count; i++) batch_num *= input_shape0->GetDim(i);   \
    float mul_scale = in_scale0 * in_scale1 / out_scale;    \
    float add_scale = in_scale2 / out_scale;                \
    gridsize.y = DivUp(batch_num, blocksize * PackSize);    \
    ppl_cukernel_fused_arithmetic_common_opt2<FusedArithmetic_MulAdd, int8_t, PackT, PackSize><<<gridsize, blocksize>>>(  \
        num_elems, dim_count, batch_num, (const int8_t*) input0, (const int8_t*) input1, \
        (const int8_t*) input2, (int8_t*) output, false, mul_scale, add_scale, act_type);

#define USE_COMMON_OPT3(T, PackT, PackSize)    \
    int inner_num = 1;  \
    for (int i = 2; i < dim_count; i++) inner_num *= input_shape0->GetDim(i);   \
    gridsize.y = DivUp(inner_num, blocksize * PackSize);   \
    gridsize.z = channel;   \
    ppl_cukernel_fused_arithmetic_common_opt3<FusedArithmetic_MulAdd, T, PackT, PackSize><<<gridsize, blocksize>>>( \
        num_elems, dim_count, channel, inner_num, fuseopt4x4,   \
        (const T*) input0, (const T*) input1, (const T*) input2, (T*) output, true, 1.f, 1.f, act_type  \
    );

#define USE_COMMON_INT8_OPT3(T, PackT, PackSize)    \
    float mul_scale = in_scale0 * in_scale1 / out_scale;    \
    float add_scale = in_scale2 / out_scale;                \
    int inner_num = 1;  \
    for (int i = 2; i < dim_count; i++) inner_num *= input_shape0->GetDim(i);   \
    gridsize.y = DivUp(inner_num, blocksize * PackSize);   \
    gridsize.z = channel;   \
    ppl_cukernel_fused_arithmetic_common_opt3<FusedArithmetic_MulAdd, int8_t, PackT, PackSize><<<gridsize, blocksize>>>( \
        num_elems, dim_count, channel, inner_num, fuseopt4x4,   \
        (const int8_t*) input0, (const int8_t*) input1, (const int8_t*) input2, (int8_t*) output, false, mul_scale, add_scale, act_type);

#define NO_USE_OPT(T, NT, NPS)  \
    uint64_t mul_outelems = input_shape0->CalcElementsIncludingPadding();    \
    T* mul_out = nullptr;  \
    cudaMalloc((void**)&mul_out, (mul_outelems)*sizeof(T)); \
    auto status = PPLCUDAArithMeticMulForwardImp(stream, input_shape0, input0, input_shape1, input1, input_shape0, mul_out, in_scale0, in_scale1, in_scale2);  \
    status = PPLCUDAArithMeticAddForwardImp(stream, input_shape0, mul_out, input_shape2, input2, output_shape, output, in_scale2, in_scale2, out_scale);   \
    cudaFree(mul_out);  \
    if (act_type == 1)  \
        LOG(WARNING) << "Need implement PPLCUDAFueseArithMeticMulAddForwardImp Relu QuantKernelParamCuda";  \
    return status;

ppl::common::RetCode PPLCUDAFusedArithMeticMulAddForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape0_ref,
    const void *input0,
    const ppl::common::TensorShape* input_shape1_ref,
    const void *input1,
    const ppl::common::TensorShape* input_shape2_ref,
    const void *input2,
    const ppl::common::TensorShape* output_shape_ref,
    void *output,
    float in_scale0,
    float in_scale1,
    float in_scale2,
    float out_scale,
    int act_type) {
    ppl::common::TensorShape input_shape0_obj = *input_shape0_ref;
    ppl::common::TensorShape input_shape1_obj = *input_shape1_ref;
    ppl::common::TensorShape input_shape2_obj = *input_shape2_ref;
    ppl::common::TensorShape output_shape_obj = *output_shape_ref;
    ppl::common::TensorShape* input_shape0 = &input_shape0_obj;
    ppl::common::TensorShape* input_shape1 = &input_shape1_obj;
    ppl::common::TensorShape* input_shape2 = &input_shape2_obj;
    ppl::common::TensorShape* output_shape = &output_shape_obj;
    
    uint64_t num_elems = output_shape->CalcElementsIncludingPadding();
    int dim_count = output_shape->GetDimCount();
    int batch = input_shape0->GetDim(0);
    int channel = input_shape0->GetDim(1);
    int height = input_shape0->GetDim(2);
    int width = input_shape0->GetDim(3);
    int hw = height * width;
    int blocksize = 256;
    dim3 gridsize(batch, 1, 1);
    int fuseopt4x4 = fused_opt_4x4(input_shape0, input_shape1, input_shape2);
    bool fuseopt4x1 = fused_opt_4x1(input_shape0, input_shape1, input_shape2);
    bool fuseoptcommon1 = fused_opt_common1(input_shape0, input_shape1, input_shape2);
    bool fuseoptcommon2 = fused_opt_common2(input_shape0, input_shape1, input_shape2);
    if (input_shape0->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8) {
        int padding0 = input_shape0->GetPadding0(1) + input_shape0->GetPadding1(1);
        if (padding0 == 0){
            int groups = channel / 8;
            int hw_inter = 256 / groups;
            int hw_outer = hw;
            if (hw_inter > 0){
                hw_outer = DivUp(hw, hw_inter);
            } else {
                gridsize.z = DivUp(channel, 2048);
            }
            gridsize.y = hw_outer;
            if(input0 == input2){
                // aba
                if(fuseopt4x4 > 0) {
                    ppl_cukernel_fused_arithmetic_4x4_aba_nhwc_nopadding<FusedArithmetic_MulAdd, half, float4, 8><<<gridsize, blocksize>>>(
                        hw, hw_inter, hw_outer, channel, DivModFast(groups), fuseopt4x4, (const half*) input0, (const half*) input1, (half*) output, true, 1.f, 1.f, act_type);
                } else if(fuseopt4x1)
                    ppl_cukernel_fused_arithmetic_4x1_aba_nhwc_nopadding<FusedArithmetic_MulAdd, half, float4, 8><<<gridsize, blocksize>>>(
                        hw, hw_inter, hw_outer, channel, DivModFast(groups), (const half*) input0, (const half*) input1, (half*) output, true, 1.f, 1.f, act_type);
                else if(fuseoptcommon1){
                    USE_COMMON_OPT(half, float4, 8)
                } else{
                    NO_USE_OPT(half, half, 1)
                }
            } else {
                // abc
                if(fuseopt4x4 > 0)
                    ppl_cukernel_fused_arithmetic_4x4_nhwc_nopadding<FusedArithmetic_MulAdd, half, float4, 8><<<gridsize, blocksize>>>(
                        hw, hw_inter, hw_outer, channel, DivModFast(groups), fuseopt4x4,
                        (const half*) input0, (const half*) input1, (const half*) input2, (half*) output, true, 1.f, 1.f, act_type);
                else if(fuseopt4x1)
                    ppl_cukernel_fused_arithmetic_4x1_nhwc_nopadding<FusedArithmetic_MulAdd, half, float4, 8><<<gridsize, blocksize>>>(
                        hw, hw_inter, hw_outer, channel, DivModFast(groups), 
                        (const half*) input0, (const half*) input1, (const half*) input2, (half*) output, true, 1.f, 1.f, act_type);
                else if(fuseoptcommon1){
                    USE_COMMON_OPT(half, float4, 8)
                } else{
                    NO_USE_OPT(half, half, 1)
                }
            }
        } else { //nhwc8 have padding
            // can usen *_hwcp function, no case test precision
            NO_USE_OPT(half, half, 1)
            // return ppl::common::RC_UNSUPPORTED;
        }
    } else if (input_shape0->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16) {
        float mul_scale = in_scale0 * in_scale1 / (out_scale);
        float add_scale = in_scale2 / out_scale;
        int padding0 = input_shape0->GetPadding0(1) + input_shape0->GetPadding1(1);
        if (padding0 == 0){
            int groups = channel / 16;
            int hw_inter = 256 / groups;
            int hw_outer = hw;
            if (hw_inter > 0){
                hw_outer = DivUp(hw, hw_inter);
            } else {
                gridsize.z = DivUp(channel, 4096);
            }
            gridsize.y = hw_outer;
            if(input0 == input2){
                if(fuseopt4x4 > 0)
                    ppl_cukernel_fused_arithmetic_4x4_aba_nhwc_nopadding<FusedArithmetic_MulAdd, int8_t, float4, 16><<<gridsize, blocksize>>>(
                        hw, hw_inter, hw_outer, channel, DivModFast(groups), fuseopt4x4, (const int8_t*) input0, (const int8_t*) input1, (int8_t*) output, false, mul_scale, add_scale, act_type);
                else if(fuseopt4x1)
                    ppl_cukernel_fused_arithmetic_4x1_aba_nhwc_nopadding<FusedArithmetic_MulAdd, int8_t, float4, 16><<<gridsize, blocksize>>>(
                        hw, hw_inter, hw_outer, channel, DivModFast(groups), (const int8_t*) input0, (const int8_t*) input1, (int8_t*) output, false, mul_scale, add_scale, act_type);
                else if(fuseoptcommon1){
                    USE_COMMON_INT8_OPT(int8_t, float4, 16)
                } else{
                    NO_USE_OPT(int8_t, int8_t, 1)
                }
            } else {
                if(fuseopt4x4 > 0)
                    ppl_cukernel_fused_arithmetic_4x4_nhwc_nopadding<FusedArithmetic_MulAdd, int8_t, float4, 16><<<gridsize, blocksize>>>(
                        hw, hw_inter, hw_outer, channel, DivModFast(groups), fuseopt4x4, 
                        (const int8_t*) input0, (const int8_t*) input1, (const int8_t*) input2, (int8_t*) output, false, mul_scale, add_scale, act_type);
                else if(fuseopt4x1)
                    ppl_cukernel_fused_arithmetic_4x1_nhwc_nopadding<FusedArithmetic_MulAdd, int8_t, float4, 16><<<gridsize, blocksize>>>(
                        hw, hw_inter, hw_outer, channel, DivModFast(groups), 
                        (const int8_t*) input0, (const int8_t*) input1, (const int8_t*) input2, (int8_t*) output, false, mul_scale, add_scale, act_type);
                else if(fuseoptcommon1){
                    USE_COMMON_INT8_OPT(int8_t, float4, 16)
                } else{
                    NO_USE_OPT(int8_t, int8_t, 1)
                }
            }
        } else {
            int padding1 = input_shape1->GetPadding0(1) + input_shape1->GetPadding1(1);
            int outpadding = output_shape->GetPadding0(1) + output_shape->GetPadding1(1);
            int groups = (channel + outpadding) / 16;
            int hw_inter = 256 / groups;
            int hw_outer = hw;
            if (hw_inter > 0){
                hw_outer = DivUp(hw, hw_inter);
            } else {
                gridsize.z = DivUp(channel, 4096);
            }
            gridsize.y = hw_outer;
            if(input0 == input2){
                // aba
                if(fuseopt4x4 > 0)
                    ppl_cukernel_fused_arithmetic_4x4_aba_nhwcp<FusedArithmetic_MulAdd, int8_t, float4, 16><<<gridsize, blocksize>>>(
                        hw, hw_inter, hw_outer, channel, channel + padding0, channel + padding1, channel + outpadding, DivModFast(groups), fuseopt4x4, 
                        (const int8_t*) input0, (const int8_t*) input1, (int8_t*) output, false, mul_scale, add_scale, act_type);
                else if(fuseopt4x1)
                    ppl_cukernel_fused_arithmetic_4x1_aba_nhwcp<FusedArithmetic_MulAdd, int8_t, float4, 16><<<gridsize, blocksize>>>(
                        hw, hw_inter, hw_outer, channel, channel + padding0, channel + outpadding, DivModFast(groups), 
                        (const int8_t*) input0, (const int8_t*) input1, (int8_t*) output, false, mul_scale, add_scale, act_type);
                else {
                    NO_USE_OPT(int8_t, int8_t, 1)
                }
            } else {
                // abc
                // can usen *_hwcp function, no case test precision
                NO_USE_OPT(int8_t, int8_t, 1)
            }
        }
    } else if (input_shape0->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
#define USEFUNCTION(FUNC, FUNC_INT8)                                            \
    if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {         \
        FUNC(half, float4, 8)                                                   \
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {  \
        FUNC(float, float4, 4)                                                  \
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT64) {  \
        FUNC(double, float4, 2)                                                 \
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_INT64) {    \
        FUNC(int64_t, float4, 2)                                                \
    } else if(output_shape->GetDataType() == ppl::common::DATATYPE_INT32) {     \
        FUNC(int32_t, float4, 4)                                                \
    } else if(output_shape->GetDataType() == ppl::common::DATATYPE_INT16) {     \
        FUNC(int16_t, float4, 8)                                                \
    } else if(output_shape->GetDataType() == ppl::common::DATATYPE_INT8) {      \
        FUNC_INT8(int8_t, float4, 16)                                           \
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_UINT64) {   \
        FUNC(uint64_t, float4, 2)                                               \
    } else if(output_shape->GetDataType() == ppl::common::DATATYPE_UINT32) {    \
        FUNC(uint32_t, float4, 4)                                               \
    } else if(output_shape->GetDataType() == ppl::common::DATATYPE_UINT16) {    \
        FUNC(uint16_t, float4, 8)                                               \
    } else if(output_shape->GetDataType() == ppl::common::DATATYPE_UINT8) {     \
        FUNC(uint8_t, float4, 16)                                               \
    } else return ppl::common::RC_UNSUPPORTED;
        if(fuseoptcommon1){
            USEFUNCTION(USE_COMMON_OPT, USE_COMMON_INT8_OPT)
        } else if(fuseoptcommon2) {
            USEFUNCTION(USE_COMMON_OPT2, USE_COMMON_INT8_OPT2)
        } else if(fuseopt4x4 > 0) {
            USEFUNCTION(USE_COMMON_OPT3, USE_COMMON_INT8_OPT3)
        } else{
            USEFUNCTION(NO_USE_OPT, NO_USE_OPT)
        }
    } else {
        // ppl::common::DATAFORMAT == NHWC or OTHER
        USEFUNCTION(NO_USE_OPT, NO_USE_OPT)
    }
    // LOG(WARNING) << "Need implement PPLCUDAFueseArithMeticMulAddForwardImp";
    return ppl::common::RC_SUCCESS;
}

