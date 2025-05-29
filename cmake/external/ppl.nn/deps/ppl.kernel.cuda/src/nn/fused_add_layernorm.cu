#include "cudakernel/nn/fused_add_layernorm.h"
#include "ppl/common/tensor_shape.h"
#include <cuda_fp16.h>
#include "cudakernel/common/common.cuh"

struct broadcast_param
{
    int32_t div_input[8];
    int32_t input_shape_0_broad[8];
    int32_t input_shape_1_broad[8];
    int32_t mul_vec_input0[8];
    int32_t mul_vec_input1[8];
    int32_t mod_vec_input[8];
};

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

template <int VPT, typename S, typename T>
__device__ inline void copy_convert(const void* local, void* data, T in_scale)
{
    const S* in = static_cast<const S*>(local);
    T* out = static_cast<T*>(data);
#pragma unroll
    for(int i = 0; i < VPT; i++){
        out[i] = (T)in[i] * in_scale;
    }
}

template<int32_t VPT, int32_t TPB>
__global__ void ppl_cukernel_fused_add_layernorm_fp16(
    const half* input_0,
    const half* input_1,
    const half* weight,
    const half* bias,
    const float eps,
    const int32_t normalize_shape,
    half* output
){
    const int32_t idx = normalize_shape * blockIdx.x + threadIdx.x * VPT;
    half sum[VPT]; half inputLocal[VPT]; half inputLocal_1[VPT]; half weightLocal[VPT]; half biasLocal[VPT];

    copy<sizeof(half) * VPT>(&input_0[idx], inputLocal);
    copy<sizeof(half) * VPT>(&input_1[idx], inputLocal_1);
    copy<sizeof(half) * VPT>(&input_1[idx], sum);
#pragma unroll
    for (int i = 0; i < VPT; i++){
        sum[i] = __hadd(sum[i], inputLocal[i]);
    }

    float2 loc = make_float2(0.f, 0.f); // accumulator
    float r_normalize_shape = 1 / (float)(normalize_shape);

#pragma unroll
    for (int32_t it = 0; it < VPT; it++)
    {
        float v = __half2float(sum[it]);
        loc.x += v;
        loc.y += v * v;
    }

    copy<sizeof(half) * VPT>(&bias[threadIdx.x * VPT], biasLocal);
    copy<sizeof(half) * VPT>(&weight[threadIdx.x * VPT], weightLocal);
    __shared__ float mu;//__shared__ half mu;     // mean
    __shared__ float rsigma;//__shared__ half rsigma; // 1 / std.dev.
    #if (__CUDACC_VER_MAJOR__ >= 11)
        // workaround cub error(C500-20246)
        // const half2 reduced = BlockAllReduce<SumOp, half2, TPB>(loc);
        const float reduced_x =  BlockAllReduce<SumOp, float, TPB>(loc.x) * r_normalize_shape;//const half reduced_x =  __float2half(BlockAllReduce<SumOp, float, TPB>(loc.x) * r_normalize_shape);
        const float reduced_y =  BlockAllReduce<SumOp, float, TPB>(loc.y) * r_normalize_shape;//const half reduced_y =  __float2half(BlockAllReduce<SumOp, float, TPB>(loc.y) * r_normalize_shape);
    #else
        // workaround cub error(C500-20246)
        // const half2 reduced = blockReduceSum<half2>(loc);
        const float reduced_x =  blockReduceSum<float>(loc.x) * r_normalize_shape;//const half reduced_x =  __float2half(blockReduceSum<float>(loc.x) * r_normalize_shape);
        const float reduced_y =  blockReduceSum<float>(loc.y) * r_normalize_shape;//const half reduced_y =  __float2half(blockReduceSum<float>(loc.y) * r_normalize_shape);
    #endif

    if (threadIdx.x == 0)
    {
        // workaround cub error(C500-20246)
        // mu = __low2half(reduced);
        // rsigma = rsqrt(__high2half(reduced) - mu * mu + __float2half(eps));
        mu = reduced_x;
        rsigma = rsqrt(reduced_y - mu * mu + eps);//rsigma = rsqrt(reduced_y - mu * mu + __float2half(eps));
    }
    __syncthreads();

    half outLocal[VPT];
#pragma unroll
    for (int32_t it = 0; it < VPT; it++)
    {
        outLocal[it] = __float2half((__half2float(sum[it]) - mu) * rsigma * __half2float(weightLocal[it]) + __half2float(biasLocal[it]));//outLocal[it] = (sum[it] - mu) * rsigma * weightLocal[it] + biasLocal[it];
    }
    copy<sizeof(half) * VPT>(outLocal, &output[idx]);
}

template<int32_t VPT>
__global__ void ppl_cukernel_fused_add_layernorm_fp16_multiple_VPT(
    const half* input_0,
    const half* input_1,
    const half* weight,
    const half* bias,
    const float eps,
    const int32_t normalize_shape,
    half* output
){
    const int32_t idx = normalize_shape * blockIdx.x + threadIdx.x * VPT;
    half sum[VPT]; half inputLocal[VPT]; half inputLocal_1[VPT]; half weightLocal[VPT]; half biasLocal[VPT];

    copy<sizeof(half) * VPT>(&input_0[idx], inputLocal);
    copy<sizeof(half) * VPT>(&input_1[idx], inputLocal_1);
    copy<sizeof(half) * VPT>(&input_1[idx], sum);
#pragma unroll
    for (int i = 0; i < VPT; i++){
        sum[i] = __hadd(sum[i], inputLocal[i]);
    }

    float2 loc = make_float2(0.f, 0.f); // accumulator
    float r_normalize_shape = 1 / (float)(normalize_shape);

#pragma unroll
    for (int32_t it = 0; it < VPT; it++)
    {
        float v = __half2float(sum[it]);
        loc.x += v;
        loc.y += v * v;
    }

    copy<sizeof(half) * VPT>(&bias[threadIdx.x * VPT], biasLocal);
    copy<sizeof(half) * VPT>(&weight[threadIdx.x * VPT], weightLocal);
    __shared__ float mu;//__shared__ half mu;     // mean
    __shared__ float rsigma;//__shared__ half rsigma; // 1 / std.dev.

    const float reduced_x =  blockReduceSum<float>(loc.x) * r_normalize_shape;//const half reduced_x =  __float2half(blockReduceSum<float>(loc.x) * r_normalize_shape);
    const float reduced_y =  blockReduceSum<float>(loc.y) * r_normalize_shape;//const half reduced_y =  __float2half(blockReduceSum<float>(loc.y) * r_normalize_shape);


    if (threadIdx.x == 0)
    {
        // workaround cub error(C500-20246)
        // mu = __low2half(reduced);
        // rsigma = rsqrt(__high2half(reduced) - mu * mu + __float2half(eps));
        mu = reduced_x;
        rsigma = rsqrt(reduced_y - mu * mu + eps);//rsigma = rsqrt(reduced_y - mu * mu + __float2half(eps));
    }
    __syncthreads();

    half outLocal[VPT];
#pragma unroll
    for (int32_t it = 0; it < VPT; it++)
    {
        outLocal[it] = __float2half((__half2float(sum[it]) - mu) * rsigma * __half2float(weightLocal[it]) + __half2float(biasLocal[it]));//outLocal[it] = (sum[it] - mu) * rsigma * weightLocal[it] + biasLocal[it];
    }
    copy<sizeof(half) * VPT>(outLocal, &output[idx]);
}

__global__ __launch_bounds__(256) void ppl_cukernel_fused_add_layernorm_fp16_default(
    const half* input_0,
    const half* input_1,
    const half* scale,
    const half* shift,
    float eps,
    int32_t N,
    half* output,
    bool has_affine
){
    auto cur_in_0 = input_0 + blockIdx.x * N;
    auto cur_in_1 = input_1 + blockIdx.x * N;
    auto cur_out = output + blockIdx.x * N;
    float2 loc = make_float2(0.f, 0.f);

    for (auto tid = threadIdx.x; tid < N; tid += blockDim.x) {
        half sum = cur_in_0[tid] + cur_in_1[tid];
        float v = __half2float(sum);
        loc.x += v;
        loc.y += v * v;
    }
    #if (__CUDACC_VER_MAJOR__ >= 11)
        loc.x =  BlockAllReduce<SumOp, float, 256>(loc.x);
        loc.y =  BlockAllReduce<SumOp, float, 256>(loc.y);
    #else
        BlockDoubleReduceSum(loc.x, loc.y);
    #endif

    float mean = loc.x / N;
    float rstd = rsqrtf(loc.y / N - mean * mean + eps);

    half mean_h = __float2half(mean);
    half rstd_h = __float2half(rstd);

    for(auto tid = threadIdx.x; tid < N; tid += blockDim.x) {
        half sum = __hadd(cur_in_0[tid], cur_in_1[tid]);
        half val = (sum - mean_h) * rstd_h;
        if(has_affine)
            val = val * scale[tid] + shift[tid];
        cur_out[tid] = val;
    }
}

template<int32_t VPT, int32_t TPB>
__global__ void ppl_cukernel_fused_add_layernorm_fp16_double_output(
    const half* input_0,
    const half* input_1,
    const half* weight,
    const half* bias,
    const float eps,
    const int32_t normalize_shape,
    half* output_0,
    half* output_1
){
    const int32_t idx = normalize_shape * blockIdx.x + threadIdx.x * VPT;
    half sum[VPT]; half inputLocal[VPT]; half weightLocal[VPT]; half biasLocal[VPT];

    copy<sizeof(half) * VPT>(&input_0[idx], inputLocal);
    copy<sizeof(half) * VPT>(&input_1[idx], sum);
#pragma unroll
    for (int i = 0; i < VPT; i++){
        sum[i] = __hadd(sum[i], inputLocal[i]);
    }

    float2 loc = make_float2(0.f, 0.f); // accumulator
    float r_normalize_shape = 1 / (float)(normalize_shape);

#pragma unroll
    for (int32_t it = 0; it < VPT; it++)
    {
        float v = __half2float(sum[it]);
        loc.x += v;
        loc.y += v * v;
    }

    copy<sizeof(half) * VPT>(&bias[threadIdx.x * VPT], biasLocal);
    copy<sizeof(half) * VPT>(&weight[threadIdx.x * VPT], weightLocal);
    __shared__ float mu;//__shared__ half mu;     // mean
    __shared__ float rsigma;//__shared__ half rsigma; // 1 / std.dev.
    #if (__CUDACC_VER_MAJOR__ >= 11)
        // workaround cub error(C500-20246)
        // const half2 reduced = BlockAllReduce<SumOp, half2, TPB>(loc);
        const float reduced_x =  BlockAllReduce<SumOp, float, TPB>(loc.x) * r_normalize_shape;//const half reduced_x =  __float2half(BlockAllReduce<SumOp, float, TPB>(loc.x) * r_normalize_shape);
        const float reduced_y =  BlockAllReduce<SumOp, float, TPB>(loc.y) * r_normalize_shape;//const half reduced_y =  __float2half(BlockAllReduce<SumOp, float, TPB>(loc.y) * r_normalize_shape);
    #else
        // workaround cub error(C500-20246)
        // const half2 reduced = blockReduceSum<half2>(loc);
        const float reduced_x =  blockReduceSum<float>(loc.x) * r_normalize_shape;//const half reduced_x =  __float2half(blockReduceSum<float>(loc.x) * r_normalize_shape);
        const float reduced_y =  blockReduceSum<float>(loc.y) * r_normalize_shape;//const half reduced_y =  __float2half(blockReduceSum<float>(loc.y) * r_normalize_shape);
    #endif

    if (threadIdx.x == 0)
    {
        // workaround cub error(C500-20246)
        // mu = __low2half(reduced);
        // rsigma = rsqrt(__high2half(reduced) - mu * mu + __float2half(eps));
        mu = reduced_x;
        rsigma = rsqrt(reduced_y - mu * mu + eps);//rsigma = rsqrt(reduced_y - mu * mu + __float2half(eps));
    }
    __syncthreads();

    half outLocal[VPT];
#pragma unroll
    for (int32_t it = 0; it < VPT; it++)
    {
        outLocal[it] = __float2half((__half2float(sum[it]) - mu) * rsigma * __half2float(weightLocal[it]) + __half2float(biasLocal[it]));//outLocal[it] = (sum[it] - mu) * rsigma * weightLocal[it] + biasLocal[it];
    }
    copy<sizeof(half) * VPT>(outLocal, &output_0[idx]);
    copy<sizeof(half) * VPT>(sum, &output_1[idx]);
}

template<int32_t VPT>
__global__ void ppl_cukernel_fused_add_layernorm_fp16_double_output_multiple_VPT(
    const half* input_0,
    const half* input_1,
    const half* weight,
    const half* bias,
    const float eps,
    const int32_t normalize_shape,
    half* output_0,
    half* output_1
){
    const int32_t idx = normalize_shape * blockIdx.x + threadIdx.x * VPT;
    half sum[VPT]; half inputLocal[VPT]; half weightLocal[VPT]; half biasLocal[VPT];

    copy<sizeof(half) * VPT>(&input_0[idx], inputLocal);
    copy<sizeof(half) * VPT>(&input_1[idx], sum);
#pragma unroll
    for (int i = 0; i < VPT; i++){
        sum[i] = __hadd(sum[i], inputLocal[i]);
    }

    float2 loc = make_float2(0.f, 0.f); // accumulator
    float r_normalize_shape = 1 / (float)(normalize_shape);

#pragma unroll
    for (int32_t it = 0; it < VPT; it++)
    {
        float v = __half2float(sum[it]);
        loc.x += v;
        loc.y += v * v;
    }

    copy<sizeof(half) * VPT>(&bias[threadIdx.x * VPT], biasLocal);
    copy<sizeof(half) * VPT>(&weight[threadIdx.x * VPT], weightLocal);
    __shared__ float mu;//__shared__ half mu;     // mean
    __shared__ float rsigma;//__shared__ half rsigma; // 1 / std.dev.

    const float reduced_x =  blockReduceSum<float>(loc.x) * r_normalize_shape;//const half reduced_x =  __float2half(blockReduceSum<float>(loc.x) * r_normalize_shape);
    const float reduced_y =  blockReduceSum<float>(loc.y) * r_normalize_shape;//const half reduced_y =  __float2half(blockReduceSum<float>(loc.y) * r_normalize_shape);


    if (threadIdx.x == 0)
    {
        mu = reduced_x;
        rsigma = rsqrt(reduced_y - mu * mu + eps);//rsigma = rsqrt(reduced_y - mu * mu + __float2half(eps));
    }
    __syncthreads();

    half outLocal[VPT];
#pragma unroll
    for (int32_t it = 0; it < VPT; it++)
    {
        outLocal[it] = __float2half((__half2float(sum[it]) - mu) * rsigma * __half2float(weightLocal[it]) + __half2float(biasLocal[it]));//outLocal[it] = (sum[it] - mu) * rsigma * weightLocal[it] + biasLocal[it];
    }
    copy<sizeof(half) * VPT>(outLocal, &output_0[idx]);
    copy<sizeof(half) * VPT>(sum, &output_1[idx]);
}

template<int32_t VPT, int32_t TPB>
__global__ void ppl_cukernel_fused_add_layernorm_fp16_double_output_broadcast(
    const half* input_0,
    const half* input_1,
    ppl::common::TensorShape* input_shape_0,
    ppl::common::TensorShape* input_shape_1,
    const half* weight,
    const half* bias,
    const float eps,
    const int32_t normalize_shape,
    half* output_0,
    half* output_1,
    const broadcast_param param,
    int max_dim
){
    const int32_t idx = normalize_shape * blockIdx.x + threadIdx.x * VPT;
    int idx_vec_0 = 0;
    int idx_vec_1 = 0;
    int idx_input_0 = 0, idx_input_1 = 0;
    for (int i = 0; i < max_dim; i++){
        idx_vec_0 = param.input_shape_0_broad[i] == 1 ? 0 : (idx % param.mod_vec_input[i]) / param.div_input[i];
        idx_vec_1 = param.input_shape_1_broad[i] == 1 ? 0 : (idx % param.mod_vec_input[i]) / param.div_input[i];
        idx_input_0 += idx_vec_0 * param.mul_vec_input0[i];
        idx_input_1 += idx_vec_1 * param.mul_vec_input1[i];
    }

    half sum[VPT]; half inputLocal[VPT]; half weightLocal[VPT]; half biasLocal[VPT];

    copy<sizeof(half) * VPT>(&input_0[idx_input_0], inputLocal);
    copy<sizeof(half) * VPT>(&input_1[idx_input_1], sum);
#pragma unroll
    for (int i = 0; i < VPT; i++){
        sum[i] = __hadd(sum[i], inputLocal[i]);
    }

    float2 loc = make_float2(0.f, 0.f); // accumulator
    float r_normalize_shape = 1 / (float)(normalize_shape);

#pragma unroll
    for (int32_t it = 0; it < VPT; it++)
    {
        float v = __half2float(sum[it]);
        loc.x += v;
        loc.y += v * v;
    }

    copy<sizeof(half) * VPT>(&bias[threadIdx.x * VPT], biasLocal);
    copy<sizeof(half) * VPT>(&weight[threadIdx.x * VPT], weightLocal);
    __shared__ float mu;//__shared__ half mu;     // mean
    __shared__ float rsigma;//__shared__ half rsigma; // 1 / std.dev.
    #if (__CUDACC_VER_MAJOR__ >= 11)
        // workaround cub error(C500-20246)
        // const half2 reduced = BlockAllReduce<SumOp, half2, TPB>(loc);
        const float reduced_x =  BlockAllReduce<SumOp, float, TPB>(loc.x) * r_normalize_shape;//const half reduced_x =  __float2half(BlockAllReduce<SumOp, float, TPB>(loc.x) * r_normalize_shape);
        const float reduced_y =  BlockAllReduce<SumOp, float, TPB>(loc.y) * r_normalize_shape;//const half reduced_y =  __float2half(BlockAllReduce<SumOp, float, TPB>(loc.y) * r_normalize_shape);
    #else
        // workaround cub error(C500-20246)
        // const half2 reduced = blockReduceSum<half2>(loc);
        const float reduced_x =  blockReduceSum<float>(loc.x) * r_normalize_shape;//const half reduced_x =  __float2half(blockReduceSum<float>(loc.x) * r_normalize_shape);
        const float reduced_y =  blockReduceSum<float>(loc.y) * r_normalize_shape;//const half reduced_y =  __float2half(blockReduceSum<float>(loc.y) * r_normalize_shape);
    #endif

    if (threadIdx.x == 0)
    {
        // workaround cub error(C500-20246)
        // mu = __low2half(reduced);
        // rsigma = rsqrt(__high2half(reduced) - mu * mu + __float2half(eps));
        mu = reduced_x;
        rsigma = rsqrt(reduced_y - mu * mu + eps);//rsigma = rsqrt(reduced_y - mu * mu + __float2half(eps));
    }
    __syncthreads();

    half outLocal[VPT];
#pragma unroll
    for (int32_t it = 0; it < VPT; it++)
    {
        outLocal[it] = __float2half((__half2float(sum[it]) - mu) * rsigma * __half2float(weightLocal[it]) + __half2float(biasLocal[it]));//outLocal[it] = (sum[it] - mu) * rsigma * weightLocal[it] + biasLocal[it];
    }
    copy<sizeof(half) * VPT>(outLocal, &output_0[idx]);
    copy<sizeof(half) * VPT>(sum, &output_1[idx]);
}

template<int32_t VPT>
__global__ void ppl_cukernel_fused_add_layernorm_fp16_double_output_broadcast_multiple_VPT(
    const half* input_0,
    const half* input_1,
    ppl::common::TensorShape* input_shape_0,
    ppl::common::TensorShape* input_shape_1,
    const half* weight,
    const half* bias,
    const float eps,
    const int32_t normalize_shape,
    half* output_0,
    half* output_1,
    const broadcast_param param,
    int max_dim
){
    const int32_t idx = normalize_shape * blockIdx.x + threadIdx.x * VPT;
    int idx_vec_0 = 0;
    int idx_vec_1 = 0;
    int idx_input_0 = 0, idx_input_1 = 0;
    for (int i = 0; i < max_dim; i++){
        idx_vec_0 = param.input_shape_0_broad[i] == 1 ? 0 : (idx % param.mod_vec_input[i]) / param.div_input[i];
        idx_vec_1 = param.input_shape_1_broad[i] == 1 ? 0 : (idx % param.mod_vec_input[i]) / param.div_input[i];
        idx_input_0 += idx_vec_0 * param.mul_vec_input0[i];
        idx_input_1 += idx_vec_1 * param.mul_vec_input1[i];
    }

    half sum[VPT]; half inputLocal[VPT]; half weightLocal[VPT]; half biasLocal[VPT];

    copy<sizeof(half) * VPT>(&input_0[idx_input_0], inputLocal);
    copy<sizeof(half) * VPT>(&input_1[idx_input_1], sum);
#pragma unroll
    for (int i = 0; i < VPT; i++){
        sum[i] = __hadd(sum[i], inputLocal[i]);
    }

    float2 loc = make_float2(0.f, 0.f); // accumulator
    float r_normalize_shape = 1 / (float)(normalize_shape);

#pragma unroll
    for (int32_t it = 0; it < VPT; it++)
    {
        float v = __half2float(sum[it]);
        loc.x += v;
        loc.y += v * v;
    }

    copy<sizeof(half) * VPT>(&bias[threadIdx.x * VPT], biasLocal);
    copy<sizeof(half) * VPT>(&weight[threadIdx.x * VPT], weightLocal);
    __shared__ float mu;//__shared__ half mu;     // mean
    __shared__ float rsigma;//__shared__ half rsigma; // 1 / std.dev.

    const float reduced_x =  blockReduceSum<float>(loc.x) * r_normalize_shape;//const half reduced_x =  __float2half(blockReduceSum<float>(loc.x) * r_normalize_shape);
    const float reduced_y =  blockReduceSum<float>(loc.y) * r_normalize_shape;//const half reduced_y =  __float2half(blockReduceSum<float>(loc.y) * r_normalize_shape);

    if (threadIdx.x == 0)
    {
        // workaround cub error(C500-20246)
        // mu = __low2half(reduced);
        // rsigma = rsqrt(__high2half(reduced) - mu * mu + __float2half(eps));
        mu = reduced_x;
        rsigma = rsqrt(reduced_y - mu * mu + eps);//rsigma = rsqrt(reduced_y - mu * mu + __float2half(eps));
    }
    __syncthreads();

    half outLocal[VPT];
#pragma unroll
    for (int32_t it = 0; it < VPT; it++)
    {
        outLocal[it] = __float2half((__half2float(sum[it]) - mu) * rsigma * __half2float(weightLocal[it]) + __half2float(biasLocal[it]));//outLocal[it] = (sum[it] - mu) * rsigma * weightLocal[it] + biasLocal[it];
    }
    copy<sizeof(half) * VPT>(outLocal, &output_0[idx]);
    copy<sizeof(half) * VPT>(sum, &output_1[idx]);
}

template<int32_t VPT, int32_t TPB>
__global__ void ppl_cukernel_fused_add_layernorm_fp16_double_output_easy_broadcast(
    const half* input_0,
    const half* input_1,
    const half* weight,
    const half* bias,
    const float eps,
    const int32_t normalize_shape,
    half* output_0,
    half* output_1,
    int div_num,
    bool input0_is_greater
){
    const int32_t idx = normalize_shape * blockIdx.x + threadIdx.x * VPT;
    half sum[VPT]; half inputLocal[VPT]; half weightLocal[VPT]; half biasLocal[VPT];
    if(input0_is_greater) {
        copy<sizeof(half) * VPT>(&input_0[idx], inputLocal);
        copy<sizeof(half) * VPT>(&input_1[idx % div_num], sum);
    }else{
        copy<sizeof(half) * VPT>(&input_0[idx % div_num], inputLocal);
        copy<sizeof(half) * VPT>(&input_1[idx], sum);
    }
#pragma unroll
    for (int i = 0; i < VPT; i++){
        sum[i] = __hadd(sum[i], inputLocal[i]);
    }

    float2 loc = make_float2(0.f, 0.f); // accumulator
    float r_normalize_shape = 1 / (float)(normalize_shape);

#pragma unroll
    for (int32_t it = 0; it < VPT; it++)
    {
        float v = __half2float(sum[it]);
        loc.x += v;
        loc.y += v * v;
    }

    copy<sizeof(half) * VPT>(&bias[threadIdx.x * VPT], biasLocal);
    copy<sizeof(half) * VPT>(&weight[threadIdx.x * VPT], weightLocal);
    __shared__ float mu;//__shared__ half mu;     // mean
    __shared__ float rsigma;//__shared__ half rsigma; // 1 / std.dev.
    #if (__CUDACC_VER_MAJOR__ >= 11)
        // workaround cub error(C500-20246)
        // const half2 reduced = BlockAllReduce<SumOp, half2, TPB>(loc);
        const float reduced_x =  BlockAllReduce<SumOp, float, TPB>(loc.x) * r_normalize_shape;//const half reduced_x =  __float2half(BlockAllReduce<SumOp, float, TPB>(loc.x) * r_normalize_shape);
        const float reduced_y =  BlockAllReduce<SumOp, float, TPB>(loc.y) * r_normalize_shape;//const half reduced_y =  __float2half(BlockAllReduce<SumOp, float, TPB>(loc.y) * r_normalize_shape);
    #else
        // workaround cub error(C500-20246)
        // const half2 reduced = blockReduceSum<half2>(loc);
        const float reduced_x =  blockReduceSum<float>(loc.x) * r_normalize_shape;//const half reduced_x =  __float2half(blockReduceSum<float>(loc.x) * r_normalize_shape);
        const float reduced_y =  blockReduceSum<float>(loc.y) * r_normalize_shape;//const half reduced_y =  __float2half(blockReduceSum<float>(loc.y) * r_normalize_shape);
    #endif

    if (threadIdx.x == 0)
    {
        // workaround cub error(C500-20246)
        // mu = __low2half(reduced);
        // rsigma = rsqrt(__high2half(reduced) - mu * mu + __float2half(eps));
        mu = reduced_x;
        rsigma = rsqrt(reduced_y - mu * mu + eps);//rsigma = rsqrt(reduced_y - mu * mu + __float2half(eps));
    }
    __syncthreads();

    half outLocal[VPT];
#pragma unroll
    for (int32_t it = 0; it < VPT; it++)
    {
        outLocal[it] = __float2half((__half2float(sum[it]) - mu) * rsigma * __half2float(weightLocal[it]) + __half2float(biasLocal[it]));//outLocal[it] = (sum[it] - mu) * rsigma * weightLocal[it] + biasLocal[it];
    }
    copy<sizeof(half) * VPT>(outLocal, &output_0[idx]);
    copy<sizeof(half) * VPT>(sum, &output_1[idx]);
}

template<int32_t VPT>
__global__ void ppl_cukernel_fused_add_layernorm_fp16_double_output_easy_broadcast_multiple_VPT(
    const half* input_0,
    const half* input_1,
    const half* weight,
    const half* bias,
    const float eps,
    const int32_t normalize_shape,
    half* output_0,
    half* output_1,
    int div_num,
    bool input0_is_greater
){
    const int32_t idx = normalize_shape * blockIdx.x + threadIdx.x * VPT;
    half sum[VPT]; half inputLocal[VPT]; half weightLocal[VPT]; half biasLocal[VPT];
    if(input0_is_greater) {
        copy<sizeof(half) * VPT>(&input_0[idx], inputLocal);
        copy<sizeof(half) * VPT>(&input_1[idx % div_num], sum);
    }else{
        copy<sizeof(half) * VPT>(&input_0[idx % div_num], inputLocal);
        copy<sizeof(half) * VPT>(&input_1[idx], sum);
    }
#pragma unroll
    for (int i = 0; i < VPT; i++){
        sum[i] = __hadd(sum[i], inputLocal[i]);
    }

    float2 loc = make_float2(0.f, 0.f); // accumulator
    float r_normalize_shape = 1 / (float)(normalize_shape);

#pragma unroll
    for (int32_t it = 0; it < VPT; it++)
    {
        float v = __half2float(sum[it]);
        loc.x += v;
        loc.y += v * v;
    }

    copy<sizeof(half) * VPT>(&bias[threadIdx.x * VPT], biasLocal);
    copy<sizeof(half) * VPT>(&weight[threadIdx.x * VPT], weightLocal);
    __shared__ float mu;//__shared__ half mu;     // mean
    __shared__ float rsigma;//__shared__ half rsigma; // 1 / std.dev.

    const float reduced_x =  blockReduceSum<float>(loc.x) * r_normalize_shape;//const half reduced_x =  __float2half(blockReduceSum<float>(loc.x) * r_normalize_shape);
    const float reduced_y =  blockReduceSum<float>(loc.y) * r_normalize_shape;//const half reduced_y =  __float2half(blockReduceSum<float>(loc.y) * r_normalize_shape);


    if (threadIdx.x == 0)
    {
        // workaround cub error(C500-20246)
        // mu = __low2half(reduced);
        // rsigma = rsqrt(__high2half(reduced) - mu * mu + __float2half(eps));
        mu = reduced_x;
        rsigma = rsqrt(reduced_y - mu * mu + eps);//rsigma = rsqrt(reduced_y - mu * mu + __float2half(eps));
    }
    __syncthreads();

    half outLocal[VPT];
#pragma unroll
    for (int32_t it = 0; it < VPT; it++)
    {
        outLocal[it] = __float2half((__half2float(sum[it]) - mu) * rsigma * __half2float(weightLocal[it]) + __half2float(biasLocal[it]));//outLocal[it] = (sum[it] - mu) * rsigma * weightLocal[it] + biasLocal[it];
    }
    copy<sizeof(half) * VPT>(outLocal, &output_0[idx]);
    copy<sizeof(half) * VPT>(sum, &output_1[idx]);
}

template<int32_t VPT, int32_t TPB>
__global__ void ppl_cukernel_fused_add_layernorm_fp16_single_output_easy_broadcast(
    const half* input_0,
    const half* input_1,
    const half* weight,
    const half* bias,
    const float eps,
    const int32_t normalize_shape,
    half* output_0,
    int div_num,
    bool input0_is_greater
){
    const int32_t idx = normalize_shape * blockIdx.x + threadIdx.x * VPT;
    half sum[VPT]; half inputLocal[VPT]; half weightLocal[VPT]; half biasLocal[VPT];
    if(input0_is_greater) {
        copy<sizeof(half) * VPT>(&input_0[idx], inputLocal);
        copy<sizeof(half) * VPT>(&input_1[idx % div_num], sum);
    }else{
        copy<sizeof(half) * VPT>(&input_0[idx % div_num], inputLocal);
        copy<sizeof(half) * VPT>(&input_1[idx], sum);
    }
#pragma unroll
    for (int i = 0; i < VPT; i++){
        sum[i] = __hadd(sum[i], inputLocal[i]);
    }

    float2 loc = make_float2(0.f, 0.f); // accumulator
    float r_normalize_shape = 1 / (float)(normalize_shape);

#pragma unroll
    for (int32_t it = 0; it < VPT; it++)
    {
        float v = __half2float(sum[it]);
        loc.x += v;
        loc.y += v * v;
    }

    copy<sizeof(half) * VPT>(&bias[threadIdx.x * VPT], biasLocal);
    copy<sizeof(half) * VPT>(&weight[threadIdx.x * VPT], weightLocal);
    __shared__ float mu;//__shared__ half mu;     // mean
    __shared__ float rsigma;//__shared__ half rsigma; // 1 / std.dev.
    #if (__CUDACC_VER_MAJOR__ >= 11)
        // workaround cub error(C500-20246)
        // const half2 reduced = BlockAllReduce<SumOp, half2, TPB>(loc);
        const float reduced_x =  BlockAllReduce<SumOp, float, TPB>(loc.x) * r_normalize_shape;//const half reduced_x =  __float2half(BlockAllReduce<SumOp, float, TPB>(loc.x) * r_normalize_shape);
        const float reduced_y =  BlockAllReduce<SumOp, float, TPB>(loc.y) * r_normalize_shape;//const half reduced_y =  __float2half(BlockAllReduce<SumOp, float, TPB>(loc.y) * r_normalize_shape);
    #else
        // workaround cub error(C500-20246)
        // const half2 reduced = blockReduceSum<half2>(loc);
        const float reduced_x =  blockReduceSum<float>(loc.x) * r_normalize_shape;//const half reduced_x =  __float2half(blockReduceSum<float>(loc.x) * r_normalize_shape);
        const float reduced_y =  blockReduceSum<float>(loc.y) * r_normalize_shape;//const half reduced_y =  __float2half(blockReduceSum<float>(loc.y) * r_normalize_shape);
    #endif

    if (threadIdx.x == 0)
    {
        // workaround cub error(C500-20246)
        // mu = __low2half(reduced);
        // rsigma = rsqrt(__high2half(reduced) - mu * mu + __float2half(eps));
        mu = reduced_x;
        rsigma = rsqrt(reduced_y - mu * mu + eps);//rsigma = rsqrt(reduced_y - mu * mu + __float2half(eps));
    }
    __syncthreads();

    half outLocal[VPT];
#pragma unroll
    for (int32_t it = 0; it < VPT; it++)
    {
        outLocal[it] = __float2half((__half2float(sum[it]) - mu) * rsigma * __half2float(weightLocal[it]) + __half2float(biasLocal[it]));//outLocal[it] = (sum[it] - mu) * rsigma * weightLocal[it] + biasLocal[it];
    }
    copy<sizeof(half) * VPT>(outLocal, &output_0[idx]);
}

template<int32_t VPT>
__global__ void ppl_cukernel_fused_add_layernorm_fp16_single_output_easy_broadcast_multiple_VPT(
    const half* input_0,
    const half* input_1,
    const half* weight,
    const half* bias,
    const float eps,
    const int32_t normalize_shape,
    half* output_0,
    int div_num,
    bool input0_is_greater
){
    const int32_t idx = normalize_shape * blockIdx.x + threadIdx.x * VPT;
    half sum[VPT]; half inputLocal[VPT]; half weightLocal[VPT]; half biasLocal[VPT];
    if(input0_is_greater) {
        copy<sizeof(half) * VPT>(&input_0[idx], inputLocal);
        copy<sizeof(half) * VPT>(&input_1[idx % div_num], sum);
    }else{
        copy<sizeof(half) * VPT>(&input_0[idx % div_num], inputLocal);
        copy<sizeof(half) * VPT>(&input_1[idx], sum);
    }
#pragma unroll
    for (int i = 0; i < VPT; i++){
        sum[i] = __hadd(sum[i], inputLocal[i]);
    }

    float2 loc = make_float2(0.f, 0.f); // accumulator
    float r_normalize_shape = 1 / (float)(normalize_shape);

#pragma unroll
    for (int32_t it = 0; it < VPT; it++)
    {
        float v = __half2float(sum[it]);
        loc.x += v;
        loc.y += v * v;
    }

    copy<sizeof(half) * VPT>(&bias[threadIdx.x * VPT], biasLocal);
    copy<sizeof(half) * VPT>(&weight[threadIdx.x * VPT], weightLocal);
    __shared__ float mu;//__shared__ half mu;     // mean
    __shared__ float rsigma;//__shared__ half rsigma; // 1 / std.dev.

    const float reduced_x =  blockReduceSum<float>(loc.x) * r_normalize_shape;//const half reduced_x =  __float2half(blockReduceSum<float>(loc.x) * r_normalize_shape);
    const float reduced_y =  blockReduceSum<float>(loc.y) * r_normalize_shape;//const half reduced_y =  __float2half(blockReduceSum<float>(loc.y) * r_normalize_shape);

    if (threadIdx.x == 0)
    {
        // workaround cub error(C500-20246)
        // mu = __low2half(reduced);
        // rsigma = rsqrt(__high2half(reduced) - mu * mu + __float2half(eps));
        mu = reduced_x;
        rsigma = rsqrt(reduced_y - mu * mu + eps);//rsigma = rsqrt(reduced_y - mu * mu + __float2half(eps));
    }
    __syncthreads();

    half outLocal[VPT];
#pragma unroll
    for (int32_t it = 0; it < VPT; it++)
    {
        outLocal[it] = __float2half((__half2float(sum[it]) - mu) * rsigma * __half2float(weightLocal[it]) + __half2float(biasLocal[it]));//outLocal[it] = (sum[it] - mu) * rsigma * weightLocal[it] + biasLocal[it];
    }
    copy<sizeof(half) * VPT>(outLocal, &output_0[idx]);
}

__global__ __launch_bounds__(256) void ppl_cukernel_fused_add_layernorm_fp16_default(
    const half* input_0,
    const half* input_1,
    const half* scale,
    const half* shift,
    float eps,
    int32_t N,
    half* output_0,
    half* output_1,
    bool has_affine
){
    auto cur_in_0 = input_0 + blockIdx.x * N;
    auto cur_in_1 = input_1 + blockIdx.x * N;
    auto cur_out_0 = output_0 + blockIdx.x * N;
    auto cur_out_1 = output_1 + blockIdx.x * N;
    float2 loc = make_float2(0.f, 0.f);

    for (auto tid = threadIdx.x; tid < N; tid += blockDim.x) {
        half sum = cur_in_0[tid] + cur_in_1[tid];
        cur_out_1[tid] = sum;
        float v = __half2float(sum);
        loc.x += v;
        loc.y += v * v;
    }
    #if (__CUDACC_VER_MAJOR__ >= 11)
        loc.x =  BlockAllReduce<SumOp, float, 256>(loc.x);
        loc.y =  BlockAllReduce<SumOp, float, 256>(loc.y);
    #else
        BlockDoubleReduceSum(loc.x, loc.y);
    #endif

    float mean = loc.x / N;
    float rstd = rsqrtf(loc.y / N - mean * mean + eps);

    half mean_h = __float2half(mean);
    half rstd_h = __float2half(rstd);

    for(auto tid = threadIdx.x; tid < N; tid += blockDim.x) {
        half sum = __hadd(cur_in_0[tid], cur_in_1[tid]);
        half val = (sum - mean_h) * rstd_h;
        if(has_affine)
            val = val * scale[tid] + shift[tid];
        cur_out_0[tid] = val;
    }
}

__global__ __launch_bounds__(256) void ppl_cukernel_fused_add_layernorm_fp16_easybroadcast_default(
    const half* input_0,
    const half* input_1,
    const half* scale,
    const half* shift,
    float eps,
    int32_t N,
    half* output_0,
    half* output_1,
    bool has_affine,
    int div_num_0,
    int div_num_1
){
    auto cur_in = blockIdx.x * N;
    auto cur_out_0 = output_0 + blockIdx.x * N;
    auto cur_out_1 = output_1 + blockIdx.x * N;
    float2 loc = make_float2(0.f, 0.f);

    for (auto tid = threadIdx.x; tid < N; tid += blockDim.x) {
        half sum = input_0[(cur_in + tid) % div_num_0] + input_1[(cur_in + tid) % div_num_1];
        cur_out_1[tid] = sum;
        float v = __half2float(sum);
        loc.x += v;
        loc.y += v * v;
    }
    #if (__CUDACC_VER_MAJOR__ >= 11)
        loc.x =  BlockAllReduce<SumOp, float, 256>(loc.x);
        loc.y =  BlockAllReduce<SumOp, float, 256>(loc.y);
    #else
        BlockDoubleReduceSum(loc.x, loc.y);
    #endif

    float mean = loc.x / N;
    float rstd = rsqrtf(loc.y / N - mean * mean + eps);

    half mean_h = __float2half(mean);
    half rstd_h = __float2half(rstd);

    for(auto tid = threadIdx.x; tid < N; tid += blockDim.x) {
        half sum = input_0[(cur_in + tid) % div_num_0] + input_1[(cur_in + tid) % div_num_1];;
        half val = (sum - mean_h) * rstd_h;
        if(has_affine)
            val = val * scale[tid] + shift[tid];
        cur_out_0[tid] = val;
    }
}

__global__ __launch_bounds__(256) void ppl_cukernel_fused_add_layernorm_fp16_easybroadcast_default(
    const half* input_0,
    const half* input_1,
    const half* scale,
    const half* shift,
    float eps,
    int32_t N,
    half* output_0,
    bool has_affine,
    int div_num_0,
    int div_num_1
){
    auto cur_in = blockIdx.x * N;
    auto cur_out_0 = output_0 + blockIdx.x * N;
    float2 loc = make_float2(0.f, 0.f);

    for (auto tid = threadIdx.x; tid < N; tid += blockDim.x) {
        half sum = input_0[(cur_in + tid) % div_num_0] + input_1[(cur_in + tid) % div_num_1];
        float v = __half2float(sum);
        loc.x += v;
        loc.y += v * v;
    }
    #if (__CUDACC_VER_MAJOR__ >= 11)
        loc.x =  BlockAllReduce<SumOp, float, 256>(loc.x);
        loc.y =  BlockAllReduce<SumOp, float, 256>(loc.y);
    #else
        BlockDoubleReduceSum(loc.x, loc.y);
    #endif

    float mean = loc.x / N;
    float rstd = rsqrtf(loc.y / N - mean * mean + eps);

    half mean_h = __float2half(mean);
    half rstd_h = __float2half(rstd);

    for(auto tid = threadIdx.x; tid < N; tid += blockDim.x) {
        half sum = input_0[(cur_in + tid) % div_num_0] + input_1[(cur_in + tid) % div_num_1];;
        half val = (sum - mean_h) * rstd_h;
        if(has_affine)
            val = val * scale[tid] + shift[tid];
        cur_out_0[tid] = val;
    }
}

__global__ __launch_bounds__(256) void ppl_cukernel_fused_add_layernorm_fp16_broadcast_default(
    const half* input_0,
    const half* input_1,
    const half* scale,
    const half* shift,
    float eps,
    half* output_0,
    half* output_1,
    bool has_affine,
    int max_dim,
    int N,
    broadcast_param param
){
    auto cur_out_0 = output_0 + blockIdx.x * N;
    auto cur_out_1 = output_1 + blockIdx.x * N;
    float2 loc = make_float2(0.f, 0.f);

    for (auto tid = threadIdx.x; tid < N; tid += blockDim.x) {
        int32_t idx = N * blockIdx.x + tid;
        int idx_vec_0 = 0;
        int idx_vec_1 = 0;
        int idx_input_0 = 0, idx_input_1 = 0;
        for (int i = 0; i < max_dim; i++){
            idx_vec_0 = param.input_shape_0_broad[i] == 1 ? 0 : (idx % param.mod_vec_input[i]) / param.div_input[i];
            idx_vec_1 = param.input_shape_1_broad[i] == 1 ? 0 : (idx % param.mod_vec_input[i]) / param.div_input[i];
            idx_input_0 += idx_vec_0 * param.mul_vec_input0[i];
            idx_input_1 += idx_vec_1 * param.mul_vec_input1[i];
        }
        half sum = input_0[idx_input_0] + input_1[idx_input_1];

        cur_out_1[tid] = sum;
        float v = __half2float(sum);
        loc.x += v;
        loc.y += v * v;
    }
    #if (__CUDACC_VER_MAJOR__ >= 11)
        loc.x =  BlockAllReduce<SumOp, float, 256>(loc.x);
        loc.y =  BlockAllReduce<SumOp, float, 256>(loc.y);
    #else
        BlockDoubleReduceSum(loc.x, loc.y);
    #endif

    float mean = loc.x / N;
    float rstd = rsqrtf(loc.y / N - mean * mean + eps);

    half mean_h = __float2half(mean);
    half rstd_h = __float2half(rstd);

    for(auto tid = threadIdx.x; tid < N; tid += blockDim.x) {
        int32_t idx = N * blockIdx.x + tid;
        int idx_vec_0 = 0;
        int idx_vec_1 = 0;
        int idx_input_0 = 0, idx_input_1 = 0;
        for (int i = 0; i < max_dim; i++){
            idx_vec_0 = param.input_shape_0_broad[i] == 1 ? 0 : (idx % param.mod_vec_input[i]) / param.div_input[i];
            idx_vec_1 = param.input_shape_1_broad[i] == 1 ? 0 : (idx % param.mod_vec_input[i]) / param.div_input[i];
            idx_input_0 += idx_vec_0 * param.mul_vec_input0[i];
            idx_input_1 += idx_vec_1 * param.mul_vec_input1[i];
        }
        half sum = input_0[idx_input_0] + input_1[idx_input_1];
        half val = (sum - mean_h) * rstd_h;
        if(has_affine)
            val = val * scale[tid] + shift[tid];
        cur_out_0[tid] = val;
    }
}

template<int32_t VPT, int32_t TPB>
__global__ void ppl_cukernel_fused_add_layernorm_int8(
    const int8_t* input_0,
    const int8_t* input_1,
    const float* weight,
    const float* bias,
    const float eps,
    const int32_t normalize_shape,
    int8_t* output,
    float input_quant_0,
    float input_quant_1,
    float output_quant
){
    const int32_t idx = normalize_shape * blockIdx.x + threadIdx.x * VPT;
    float sum[VPT]; float inputLocal[VPT]; float weightLocal[VPT]; float biasLocal[VPT];

    copy_convert<VPT, int8_t, float>(&input_0[idx], inputLocal, input_quant_0);
    copy_convert<VPT, int8_t, float>(&input_1[idx], sum, input_quant_1);

#pragma unroll
    for (int i = 0; i < VPT; i++){
        sum[i] += inputLocal[i];
    }

    float2 loc = make_float2(0.f, 0.f); // accumulator
    float r_normalize_shape = 1 / (float)(normalize_shape);

#pragma unroll
    for (int32_t it = 0; it < VPT; it++)
    {
        float v = sum[it];
        loc.x += v;
        loc.y += v * v;
    }

    copy<sizeof(float) * VPT>(&bias[threadIdx.x * VPT], biasLocal);
    copy<sizeof(float) * VPT>(&weight[threadIdx.x * VPT], weightLocal);

    __shared__ float mu;//__shared__ half mu;     // mean
    __shared__ float rsigma;//__shared__ half rsigma; // 1 / std.dev.
    #if (__CUDACC_VER_MAJOR__ >= 11)
        // workaround cub error(C500-20246)
        // const half2 reduced = BlockAllReduce<SumOp, half2, TPB>(loc);
        const float reduced_x =  BlockAllReduce<SumOp, float, TPB>(loc.x) * r_normalize_shape;//const half reduced_x =  __float2half(BlockAllReduce<SumOp, float, TPB>(loc.x) * r_normalize_shape);
        const float reduced_y =  BlockAllReduce<SumOp, float, TPB>(loc.y) * r_normalize_shape;//const half reduced_y =  __float2half(BlockAllReduce<SumOp, float, TPB>(loc.y) * r_normalize_shape);
    #else
        // workaround cub error(C500-20246)
        // const half2 reduced = blockReduceSum<half2>(loc);
        const float reduced_x =  blockReduceSum<float>(loc.x) * r_normalize_shape;//const half reduced_x =  __float2half(blockReduceSum<float>(loc.x) * r_normalize_shape);
        const float reduced_y =  blockReduceSum<float>(loc.y) * r_normalize_shape;//const half reduced_y =  __float2half(blockReduceSum<float>(loc.y) * r_normalize_shape);
    #endif

    if (threadIdx.x == 0)
    {
        mu = reduced_x;
        rsigma = rsqrt(reduced_y - mu * mu + eps);//rsigma = rsqrt(reduced_y - mu * mu + __float2half(eps));
    }
    __syncthreads();

    int8_t outLocal[VPT];
#pragma unroll
    for (int32_t it = 0; it < VPT; it++)
    {
        float output_f = (sum[it]- mu) * rsigma * (weightLocal[it]) + (biasLocal[it]);
        int int_val = __float2int_rn(output_quant * output_f);
        if (int_val < -128) int_val = -128;
        if (int_val > 127) int_val = 127;
        int8_t dst = static_cast<int8_t>(int_val);
        outLocal[it] = dst;
    }
    copy<sizeof(int8_t) * VPT>(outLocal, &output[idx]);
}

template<int32_t VPT, int32_t TPB>
__global__ void ppl_cukernel_fused_add_layernorm_int8(
    const int8_t* input_0,
    const int8_t* input_1,
    const float* weight,
    const float* bias,
    const float eps,
    const int32_t normalize_shape,
    int8_t* output,
    int8_t* output_add,
    float input_quant_0,
    float input_quant_1,
    float output_quant,
    float output_quant_add
){
    const int32_t idx = normalize_shape * blockIdx.x + threadIdx.x * VPT;
    float sum[VPT]; float inputLocal[VPT]; float weightLocal[VPT]; float biasLocal[VPT]; int8_t output_sum[VPT];

    copy_convert<VPT, int8_t, float>(&input_0[idx], inputLocal, input_quant_0);
    copy_convert<VPT, int8_t, float>(&input_1[idx], sum, input_quant_1);

#pragma unroll
    for (int i = 0; i < VPT; i++){
        sum[i] += inputLocal[i];
        int int_val_add = __float2int_rn(output_quant_add * sum[i]);
        if (int_val_add < -128) int_val_add = -128;
        if (int_val_add > 127) int_val_add = 127;
        int8_t dst_add = static_cast<int8_t>(int_val_add);
        output_sum[i] = dst_add;
    }
    copy<sizeof(int8_t) * VPT>(output_sum, &output_add[idx]);

    float2 loc = make_float2(0.f, 0.f); // accumulator
    float r_normalize_shape = 1 / (float)(normalize_shape);

#pragma unroll
    for (int32_t it = 0; it < VPT; it++)
    {
        float v = sum[it];
        loc.x += v;
        loc.y += v * v;
    }

    copy<sizeof(float) * VPT>(&bias[threadIdx.x * VPT], biasLocal);
    copy<sizeof(float) * VPT>(&weight[threadIdx.x * VPT], weightLocal);

    __shared__ float mu;//__shared__ half mu;     // mean
    __shared__ float rsigma;//__shared__ half rsigma; // 1 / std.dev.
    #if (__CUDACC_VER_MAJOR__ >= 11)
        // workaround cub error(C500-20246)
        // const half2 reduced = BlockAllReduce<SumOp, half2, TPB>(loc);
        const float reduced_x =  BlockAllReduce<SumOp, float, TPB>(loc.x) * r_normalize_shape;//const half reduced_x =  __float2half(BlockAllReduce<SumOp, float, TPB>(loc.x) * r_normalize_shape);
        const float reduced_y =  BlockAllReduce<SumOp, float, TPB>(loc.y) * r_normalize_shape;//const half reduced_y =  __float2half(BlockAllReduce<SumOp, float, TPB>(loc.y) * r_normalize_shape);
    #else
        // workaround cub error(C500-20246)
        // const half2 reduced = blockReduceSum<half2>(loc);
        const float reduced_x =  blockReduceSum<float>(loc.x) * r_normalize_shape;//const half reduced_x =  __float2half(blockReduceSum<float>(loc.x) * r_normalize_shape);
        const float reduced_y =  blockReduceSum<float>(loc.y) * r_normalize_shape;//const half reduced_y =  __float2half(blockReduceSum<float>(loc.y) * r_normalize_shape);
    #endif

    if (threadIdx.x == 0)
    {
        mu = reduced_x;
        rsigma = rsqrt(reduced_y - mu * mu + eps);//rsigma = rsqrt(reduced_y - mu * mu + __float2half(eps));
    }
    __syncthreads();

    int8_t outLocal[VPT];
#pragma unroll
    for (int32_t it = 0; it < VPT; it++)
    {
        float output_f = (sum[it]- mu) * rsigma * (weightLocal[it]) + (biasLocal[it]);
        int int_val = __float2int_rn(output_quant * output_f);
        if (int_val < -128) int_val = -128;
        if (int_val > 127) int_val = 127;
        int8_t dst = static_cast<int8_t>(int_val);
        outLocal[it] = dst;
    }
    copy<sizeof(int8_t) * VPT>(outLocal, &output[idx]);
}

template<int32_t VPT>
__global__ void ppl_cukernel_fused_add_layernorm_int8_multiple_VPT(
    const int8_t* input_0,
    const int8_t* input_1,
    const float* weight,
    const float* bias,
    const float eps,
    const int32_t normalize_shape,
    int8_t* output,
    int8_t* output_add,
    float input_quant_0,
    float input_quant_1,
    float output_quant,
    float output_quant_add
){
    const int32_t idx = normalize_shape * blockIdx.x + threadIdx.x * VPT;
    float sum[VPT]; float inputLocal[VPT]; float weightLocal[VPT]; float biasLocal[VPT]; int8_t output_sum[VPT];

    copy_convert<VPT, int8_t, float>(&input_0[idx], inputLocal, input_quant_0);
    copy_convert<VPT, int8_t, float>(&input_1[idx], sum, input_quant_1);

#pragma unroll
    for (int i = 0; i < VPT; i++){
        sum[i] += inputLocal[i];
        int int_val_add = __float2int_rn(output_quant_add * sum[i]);
        if (int_val_add < -128) int_val_add = -128;
        if (int_val_add > 127) int_val_add = 127;
        int8_t dst_add = static_cast<int8_t>(int_val_add);
        output_sum[i] = dst_add;
    }
    copy<sizeof(int8_t) * VPT>(output_sum, &output_add[idx]);

    float2 loc = make_float2(0.f, 0.f); // accumulator
    float r_normalize_shape = 1 / (float)(normalize_shape);

#pragma unroll
    for (int32_t it = 0; it < VPT; it++)
    {
        float v = sum[it];
        loc.x += v;
        loc.y += v * v;
    }

    copy<sizeof(float) * VPT>(&bias[threadIdx.x * VPT], biasLocal);
    copy<sizeof(float) * VPT>(&weight[threadIdx.x * VPT], weightLocal);

    __shared__ float mu;//__shared__ half mu;     // mean
    __shared__ float rsigma;//__shared__ half rsigma; // 1 / std.dev.

    const float reduced_x =  blockReduceSum<float>(loc.x) * r_normalize_shape;//const half reduced_x =  __float2half(blockReduceSum<float>(loc.x) * r_normalize_shape);
    const float reduced_y =  blockReduceSum<float>(loc.y) * r_normalize_shape;//const half reduced_y =  __float2half(blockReduceSum<float>(loc.y) * r_normalize_shape);


    if (threadIdx.x == 0)
    {
        mu = reduced_x;
        rsigma = rsqrt(reduced_y - mu * mu + eps);//rsigma = rsqrt(reduced_y - mu * mu + __float2half(eps));
    }
    __syncthreads();

    int8_t outLocal[VPT];
#pragma unroll
    for (int32_t it = 0; it < VPT; it++)
    {
        float output_f = (sum[it]- mu) * rsigma * (weightLocal[it]) + (biasLocal[it]);
        int int_val = __float2int_rn(output_quant * output_f);
        if (int_val < -128) int_val = -128;
        if (int_val > 127) int_val = 127;
        int8_t dst = static_cast<int8_t>(int_val);
        outLocal[it] = dst;
    }
    copy<sizeof(int8_t) * VPT>(outLocal, &output[idx]);
}

template<int32_t VPT>
__global__ void ppl_cukernel_fused_add_layernorm_int8_multiple_VPT(
    const int8_t* input_0,
    const int8_t* input_1,
    const float* weight,
    const float* bias,
    const float eps,
    const int32_t normalize_shape,
    int8_t* output,
    float input_quant_0,
    float input_quant_1,
    float output_quant
){
    const int32_t idx = normalize_shape * blockIdx.x + threadIdx.x * VPT;
    float sum[VPT]; float inputLocal[VPT]; float weightLocal[VPT]; float biasLocal[VPT];

    copy_convert<VPT, int8_t, float>(&input_0[idx], inputLocal, input_quant_0);
    copy_convert<VPT, int8_t, float>(&input_1[idx], sum, input_quant_1);

#pragma unroll
    for (int i = 0; i < VPT; i++){
        sum[i] += inputLocal[i];
    }

    float2 loc = make_float2(0.f, 0.f); // accumulator
    float r_normalize_shape = 1 / (float)(normalize_shape);

#pragma unroll
    for (int32_t it = 0; it < VPT; it++)
    {
        float v = sum[it];
        loc.x += v;
        loc.y += v * v;
    }

    copy<sizeof(float) * VPT>(&bias[threadIdx.x * VPT], biasLocal);
    copy<sizeof(float) * VPT>(&weight[threadIdx.x * VPT], weightLocal);
    __shared__ float mu;//__shared__ half mu;     // mean
    __shared__ float rsigma;//__shared__ half rsigma; // 1 / std.dev.

    const float reduced_x =  blockReduceSum<float>(loc.x) * r_normalize_shape;//const half reduced_x =  __float2half(blockReduceSum<float>(loc.x) * r_normalize_shape);
    const float reduced_y =  blockReduceSum<float>(loc.y) * r_normalize_shape;//const half reduced_y =  __float2half(blockReduceSum<float>(loc.y) * r_normalize_shape);

    if (threadIdx.x == 0)
    {
        mu = reduced_x;
        rsigma = rsqrt(reduced_y - mu * mu + eps);//rsigma = rsqrt(reduced_y - mu * mu + __float2half(eps));
    }
    __syncthreads();

    int8_t outLocal[VPT];
#pragma unroll
    for (int32_t it = 0; it < VPT; it++)
    {
        float output_f = (sum[it]- mu) * rsigma * (weightLocal[it]) + (biasLocal[it]);
        int int_val = __float2int_rn(output_quant * output_f);
        if (int_val < -128) int_val = -128;
        if (int_val > 127) int_val = 127;
        int8_t dst = static_cast<int8_t>(int_val);
        outLocal[it] = dst;
    }
    copy<sizeof(int8_t) * VPT>(outLocal, &output[idx]);
}

__global__ __launch_bounds__(256) void ppl_cukernel_fused_add_layernorm_int8_default(
    const int8_t* input_0,
    const int8_t* input_1,
    const float* scale,
    const float* shift,
    float eps,
    int32_t N,
    int8_t* output,
    bool has_affine,
    float input_scale_0,
    float input_scale_1,
    float output_scale
){
    auto cur_in_0 = input_0 + blockIdx.x * N;
    auto cur_in_1 = input_1 + blockIdx.x * N;
    auto cur_out = output + blockIdx.x * N;
    float2 loc = make_float2(0.f, 0.f);
    float r_normalize_shape = 1.0f / (float)(N);

    __shared__ float mu;
    __shared__ float rsigma;

    for (auto tid = threadIdx.x; tid < N; tid += blockDim.x) {
        float sum = (float)cur_in_0[tid] * input_scale_0 + (float)cur_in_1[tid] * input_scale_1;
        loc.x += sum;
        loc.y += sum * sum;
    }
    #if (__CUDACC_VER_MAJOR__ >= 11)
        const float reduce_x =  BlockAllReduce<SumOp, float, 256>(loc.x) * r_normalize_shape;
        const float reduce_y =  BlockAllReduce<SumOp, float, 256>(loc.y) * r_normalize_shape;
    #else
        const float reduce_x = blockReduceSum<float>(loc.x) * r_normalize_shape;
        const float reduce_y = blockReduceSum<float>(loc.y) * r_normalize_shape;
    #endif

    if(threadIdx.x == 0){
        mu = reduce_x;
        rsigma = rsqrt(reduce_y - mu * mu + eps);
    }
    __syncthreads();

    for(auto tid = threadIdx.x; tid < N; tid += blockDim.x) {
        float val = (((float)cur_in_0[tid] * input_scale_0 + (float)cur_in_1[tid] * input_scale_1) - mu) * rsigma;
        if(has_affine){
            val = val * scale[tid] + shift[tid];
        }
        int int_val = __float2int_rn(output_scale * val);
        if (int_val < -128) int_val = -128;
        if (int_val > 127) int_val = 127;
        int8_t dst = static_cast<int8_t>(int_val);
        cur_out[tid] = dst;
    }
}

__global__ __launch_bounds__(256) void ppl_cukernel_fused_add_layernorm_int8_default(
    const int8_t* input_0,
    const int8_t* input_1,
    const float* scale,
    const float* shift,
    float eps,
    int32_t N,
    int8_t* output,
    int8_t* output_add,
    bool has_affine,
    float input_scale_0,
    float input_scale_1,
    float output_scale,
    float output_scale_add
){
    auto cur_in_0 = input_0 + blockIdx.x * N;
    auto cur_in_1 = input_1 + blockIdx.x * N;
    auto cur_out = output + blockIdx.x * N;
    auto cur_out_add = output_add + blockIdx.x * N;
    float2 loc = make_float2(0.f, 0.f);
    float r_normalize_shape = 1.0f / (float)(N);

    __shared__ float mu;
    __shared__ float rsigma;

    for (auto tid = threadIdx.x; tid < N; tid += blockDim.x) {
        float sum = (float)cur_in_0[tid] * input_scale_0 + (float)cur_in_1[tid] * input_scale_1;
        int int_val_add = __float2int_rn(sum * output_scale_add);
        if (int_val_add < -128) int_val_add = -128;
        if (int_val_add > 127) int_val_add = 127;
        int8_t dst_add = static_cast<int8_t>(int_val_add);
        cur_out_add[tid] = dst_add;
        loc.x += sum;
        loc.y += sum * sum;
    }
    #if (__CUDACC_VER_MAJOR__ >= 11)
        const float reduce_x =  BlockAllReduce<SumOp, float, 256>(loc.x) * r_normalize_shape;
        const float reduce_y =  BlockAllReduce<SumOp, float, 256>(loc.y) * r_normalize_shape;
    #else
        const float reduce_x = blockReduceSum<float>(loc.x) * r_normalize_shape;
        const float reduce_y = blockReduceSum<float>(loc.y) * r_normalize_shape;
    #endif

    if(threadIdx.x == 0){
        mu = reduce_x;
        rsigma = rsqrt(reduce_y - mu * mu + eps);
    }
    __syncthreads();

    for(auto tid = threadIdx.x; tid < N; tid += blockDim.x) {
        float val = (((float)cur_in_0[tid] * input_scale_0 + (float)cur_in_1[tid] * input_scale_1) - mu) * rsigma;
        if(has_affine){
            val = val * scale[tid] + shift[tid];
        }
        int int_val = __float2int_rn(output_scale * val);
        if (int_val < -128) int_val = -128;
        if (int_val > 127) int_val = 127;
        int8_t dst = static_cast<int8_t>(int_val);
        cur_out[tid] = dst;
    }
}

void easyBroadCast(bool& easy_broadcast,int dim_0, int dim_1, int max_dim,
    ppl::common::TensorShape* input_shape_0,
    ppl::common::TensorShape* input_shape_1)
{
    for (int i = max_dim - 1; i >= 0; i--){
        if((i - max_dim + dim_0) >= 0 && (i - max_dim + dim_1) >= 0){
            if(input_shape_0->GetDim(i - max_dim + dim_0) == input_shape_1->GetDim(i - max_dim + dim_1)){
                continue;
            }
            if((i - max_dim + dim_0) > 0 && (i - max_dim + dim_1) > 0){
                bool all_one_0 = true;
                bool all_one_1 = true;
                for (int j = i - 1; j >= 0; j--) {
                    if ((j - max_dim + dim_0) >= 0){
                        if(input_shape_0->GetDim(j - max_dim + dim_0) != 1){
                            all_one_0 = false;
                        }
                    }
                    if ((j - max_dim + dim_1) >= 0){
                        if(input_shape_1->GetDim(j - max_dim + dim_1) != 1){
                            all_one_1 = false;
                        }
                    }
                }
                easy_broadcast = all_one_0 || all_one_1;
                break;
            }
            if ((i - max_dim + dim_0) == 0 && (i - max_dim + dim_1) == 0){
                easy_broadcast = true;
                break;
            }
            if((i - max_dim + dim_0) == 0) {
                if (input_shape_0->GetDim(i - max_dim + dim_0) == 1){
                    easy_broadcast = true;
                    break;
                }
                for (int j = i - 1; j >= 0; j++){
                    if ((j - max_dim + dim_1) >= 0){
                        if(input_shape_1->GetDim(j - max_dim + dim_1) != 1){
                            easy_broadcast = false;
                            break;
                        }
                    }
                }
                easy_broadcast = true;
                break;
            }
            if ((i - max_dim + dim_1) == 0) {
                if (input_shape_1->GetDim(i - max_dim + dim_1) == 1){
                    easy_broadcast = true;
                    break;
                }
                for (int j = i - 1; j >= 0; j++){
                    if ((j - max_dim + dim_0) >= 0){
                        if(input_shape_0->GetDim(j - max_dim + dim_0) != 1){
                            easy_broadcast = false;
                            break;
                        }
                    }
                }
                easy_broadcast = true;
                break;
            }
        }
    }
}

ppl::common::RetCode PPLCUDAFusedAddLayerNormForwardImp(
    cudaStream_t stream,
    ppl::common::TensorShape* input_shape_0,
    ppl::common::TensorShape* input_shape_1,
    const void* input_0,
    const void* input_1,
    const void* scale,
    const void* shift,
    void* output,
    ppl::common::TensorShape* output_shape,
    int outer,
    int inner,
    bool elementwise_affine,
    float eps,
    float in_scale_0,
    float in_scale_1,
    float out_scale){

    const int32_t norm_size = inner;
    int32_t grid_size = outer;
    int max_dim = output_shape->GetDimCount();
    if (output_shape->GetDataFormat() != ppl::common::DATAFORMAT_NDARRAY){
        return ppl::common::RC_UNSUPPORTED;
    }

    bool need_broadcast = false;
    for (int i = 0; i < max_dim; i++) {
        if (input_shape_0->GetDim(i) != input_shape_1->GetDim(i)){
            need_broadcast = true;
            break;
        }
    }

    if(input_shape_0->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        constexpr int32_t VPT = 16 / sizeof(half);
        if(!need_broadcast) {
            switch (norm_size)
            {
            case 128:
                ppl_cukernel_fused_add_layernorm_fp16<VPT, 128 / VPT><<<grid_size, 128 / VPT, 0, stream>>>
                ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output);
                break;
            case 256:
                ppl_cukernel_fused_add_layernorm_fp16<VPT, 256 / VPT><<<grid_size, 256 / VPT, 0, stream>>>
                ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output);
                break;
            case 320:
                ppl_cukernel_fused_add_layernorm_fp16<VPT, 320 / VPT><<<grid_size, 320 / VPT, 0, stream>>>
                ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output);
                break;
            case 512:
                ppl_cukernel_fused_add_layernorm_fp16<VPT, 512 / VPT><<<grid_size, 512 / VPT, 0, stream>>>
                ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output);
                break;
            case 640:
                ppl_cukernel_fused_add_layernorm_fp16<VPT, 640 / VPT><<<grid_size, 640 / VPT, 0, stream>>>
                ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output);
                break;
            case 768:
                ppl_cukernel_fused_add_layernorm_fp16<VPT, 768 / VPT><<<grid_size, 768 / VPT, 0, stream>>>
                ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output);
                break;
            case 1024:
                ppl_cukernel_fused_add_layernorm_fp16<VPT, 1024 / VPT><<<grid_size, 1024 / VPT, 0, stream>>>
                ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output);
                break;
            case 1280:
                ppl_cukernel_fused_add_layernorm_fp16<VPT, 1280 / VPT><<<grid_size, 1280 / VPT, 0, stream>>>
                ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output);
                break;
            case 2048:
                ppl_cukernel_fused_add_layernorm_fp16<VPT, 2048 / VPT><<<grid_size, 2048 / VPT, 0, stream>>>
                ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output);
                break;
            case 4096:
                ppl_cukernel_fused_add_layernorm_fp16<VPT, 4096 / VPT><<<grid_size, 4096 / VPT, 0, stream>>>
                ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output);
                break;
            default:
                bool if_opt = norm_size % 8 == 0 && norm_size < 1024 * VPT;
                if (if_opt){
                    ppl_cukernel_fused_add_layernorm_fp16_multiple_VPT<VPT><<<grid_size, norm_size / VPT, 0, stream>>>
                    ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output);
                } else {
                    ppl_cukernel_fused_add_layernorm_fp16_default<<<grid_size, 256, 0, stream>>>
                    ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output, true);
                }
                break;
            }
            return ppl::common::RC_SUCCESS;
        } else {
            int dim_0 = input_shape_0->GetDimCount();
            int dim_1 = input_shape_1->GetDimCount();
            bool easy_broadcast = false;
            easyBroadCast(easy_broadcast, dim_0, dim_1, max_dim, input_shape_0, input_shape_1);
            if (easy_broadcast){
                int num_input0 = 1;
                int num_input1 = 1;
                for (int i = 0; i < dim_0; i++){
                    num_input0 *= input_shape_0->GetDim(i);
                }
                for (int i = 0; i < dim_1; i++){
                    num_input1 *= input_shape_1->GetDim(i);
                }
                int div_num = num_input0 > num_input1 ? num_input1 : num_input0;
                constexpr int32_t VPT = 16 / sizeof(half);
                switch (norm_size)
                {
                case 128:
                    ppl_cukernel_fused_add_layernorm_fp16_single_output_easy_broadcast<VPT, 128 / VPT><<<grid_size, 128 / VPT, 0, stream>>>
                    ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output, div_num, (bool)(num_input0 > num_input1));
                    break;
                case 256:
                    ppl_cukernel_fused_add_layernorm_fp16_single_output_easy_broadcast<VPT, 256 / VPT><<<grid_size, 256 / VPT, 0, stream>>>
                    ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output, div_num, (bool)(num_input0 > num_input1));
                    break;
                case 320:
                    ppl_cukernel_fused_add_layernorm_fp16_single_output_easy_broadcast<VPT, 320 / VPT><<<grid_size, 320 / VPT, 0, stream>>>
                    ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output, div_num, (bool)(num_input0 > num_input1));
                    break;
                case 512:
                    ppl_cukernel_fused_add_layernorm_fp16_single_output_easy_broadcast<VPT, 512 / VPT><<<grid_size, 512 / VPT, 0, stream>>>
                    ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output, div_num, (bool)(num_input0 > num_input1));
                    break;
                case 640:
                    ppl_cukernel_fused_add_layernorm_fp16_single_output_easy_broadcast<VPT, 640 / VPT><<<grid_size, 640 / VPT, 0, stream>>>
                    ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output, div_num, (bool)(num_input0 > num_input1));
                    break;
                case 768:
                    ppl_cukernel_fused_add_layernorm_fp16_single_output_easy_broadcast<VPT, 768 / VPT><<<grid_size, 768 / VPT, 0, stream>>>
                    ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output, div_num, (bool)(num_input0 > num_input1));
                    break;
                case 1024:
                    ppl_cukernel_fused_add_layernorm_fp16_single_output_easy_broadcast<VPT, 1024 / VPT><<<grid_size, 1024 / VPT, 0, stream>>>
                    ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output, div_num, (bool)(num_input0 > num_input1));
                    break;
                case 1280:
                    ppl_cukernel_fused_add_layernorm_fp16_single_output_easy_broadcast<VPT, 1280 / VPT><<<grid_size, 1280 / VPT, 0, stream>>>
                    ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output, div_num, (bool)(num_input0 > num_input1));
                    break;
                case 2048:
                    ppl_cukernel_fused_add_layernorm_fp16_single_output_easy_broadcast<VPT, 2048 / VPT><<<grid_size, 2048 / VPT, 0, stream>>>
                    ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output, div_num, (bool)(num_input0 > num_input1));
                    break;
                case 4096:
                    ppl_cukernel_fused_add_layernorm_fp16_single_output_easy_broadcast<VPT, 4096 / VPT><<<grid_size, 4096 / VPT, 0, stream>>>
                    ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output, div_num, (bool)(num_input0 > num_input1));
                    break;
                default:
                    bool if_opt = norm_size % 8 == 0 && norm_size < 1024 * VPT;
                    if (if_opt){
                        ppl_cukernel_fused_add_layernorm_fp16_single_output_easy_broadcast_multiple_VPT<VPT><<<grid_size, norm_size / VPT, 0, stream>>>
                    ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output, div_num, (bool)(num_input0 > num_input1));
                    }else {
                        ppl_cukernel_fused_add_layernorm_fp16_easybroadcast_default<<<grid_size, 256, 0, stream>>>
                        ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output, true, num_input0, num_input1);
                    }
                    break;
                }
                return ppl::common::RC_SUCCESS;
            } else {
                return ppl::common::RC_UNSUPPORTED;
            }
        }
    } else if(input_shape_0->GetDataType() == ppl::common::DATATYPE_INT8){
        if (!need_broadcast) {
            constexpr int32_t VPT = 4 / sizeof(int8_t);
            switch (norm_size)
            {
            case 128:
                ppl_cukernel_fused_add_layernorm_int8<VPT, 128 / VPT><<<grid_size, 128 / VPT, 0, stream>>>
                ((const int8_t*)input_0, (const int8_t*)input_1, (float*)scale, (float*)shift, eps, norm_size, (int8_t*)output, in_scale_0, in_scale_1, out_scale);
                break;
            case 256:
                ppl_cukernel_fused_add_layernorm_int8<VPT, 256 / VPT><<<grid_size, 256 / VPT, 0, stream>>>
                ((const int8_t*)input_0, (const int8_t*)input_1, (float*)scale, (float*)shift, eps, norm_size, (int8_t*)output, in_scale_0, in_scale_1, out_scale);
                break;
            case 320:
                ppl_cukernel_fused_add_layernorm_int8<VPT, 320 / VPT><<<grid_size, 320 / VPT, 0, stream>>>
                ((const int8_t*)input_0, (const int8_t*)input_1, (float*)scale, (float*)shift, eps, norm_size, (int8_t*)output, in_scale_0, in_scale_1, out_scale);
                break;
            case 512:
                ppl_cukernel_fused_add_layernorm_int8<VPT, 512 / VPT><<<grid_size, 512 / VPT, 0, stream>>>
                ((const int8_t*)input_0, (const int8_t*)input_1, (float*)scale, (float*)shift, eps, norm_size, (int8_t*)output, in_scale_0, in_scale_1, out_scale);
                break;
            case 640:
                ppl_cukernel_fused_add_layernorm_int8<VPT, 640 / VPT><<<grid_size, 640 / VPT, 0, stream>>>
                ((const int8_t*)input_0, (const int8_t*)input_1, (float*)scale, (float*)shift, eps, norm_size, (int8_t*)output, in_scale_0, in_scale_1, out_scale);
                break;
            case 768:
                ppl_cukernel_fused_add_layernorm_int8<VPT, 768 / VPT><<<grid_size, 768 / VPT, 0, stream>>>
                ((const int8_t*)input_0, (const int8_t*)input_1, (float*)scale, (float*)shift, eps, norm_size, (int8_t*)output, in_scale_0, in_scale_1, out_scale);
                break;
            case 1024:
                ppl_cukernel_fused_add_layernorm_int8<VPT, 1024 / VPT><<<grid_size, 1024 / VPT, 0, stream>>>
                ((const int8_t*)input_0, (const int8_t*)input_1, (float*)scale, (float*)shift, eps, norm_size, (int8_t*)output, in_scale_0, in_scale_1, out_scale);
                break;
            case 1280:
                ppl_cukernel_fused_add_layernorm_int8<VPT, 1280 / VPT><<<grid_size, 1280 / VPT, 0, stream>>>
                ((const int8_t*)input_0, (const int8_t*)input_1, (float*)scale, (float*)shift, eps, norm_size, (int8_t*)output, in_scale_0, in_scale_1, out_scale);
                break;
            case 2048:
                ppl_cukernel_fused_add_layernorm_int8<VPT, 2048 / VPT><<<grid_size, 2048 / VPT, 0, stream>>>
                ((const int8_t*)input_0, (const int8_t*)input_1, (float*)scale, (float*)shift, eps, norm_size, (int8_t*)output, in_scale_0, in_scale_1, out_scale);
                break;
            case 4096:
                ppl_cukernel_fused_add_layernorm_int8<VPT, 4096 / VPT><<<grid_size, 4096 / VPT, 0, stream>>>
                ((const int8_t*)input_0, (const int8_t*)input_1, (float*)scale, (float*)shift, eps, norm_size, (int8_t*)output, in_scale_0, in_scale_1, out_scale);
                break;
            default:
                bool if_opt = norm_size % 8 == 0 && norm_size < 1024 * VPT;
                if (if_opt){
                    ppl_cukernel_fused_add_layernorm_int8_multiple_VPT<VPT><<<grid_size, norm_size / VPT, 0, stream>>>
                    ((const int8_t*)input_0, (const int8_t*)input_1, (float*)scale, (float*)shift, eps, norm_size, (int8_t*)output, in_scale_0, in_scale_1, out_scale);
                } else {
                    ppl_cukernel_fused_add_layernorm_int8_default<<<grid_size, 256, 0, stream>>>
                    ((const int8_t*)input_0, (const int8_t*)input_1, (float*)scale, (float*)shift, eps, norm_size, (int8_t*)output, true, in_scale_0, in_scale_1, out_scale);
                }
                break;
            }
        } else {
            return ppl::common::RC_UNSUPPORTED;
        }
        return ppl::common::RC_SUCCESS;
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
    return ppl::common::RC_UNSUPPORTED;
}

ppl::common::RetCode PPLCUDAFusedAddLayerNormForwardImp(
    cudaStream_t stream,
    ppl::common::TensorShape* input_shape_0,
    ppl::common::TensorShape* input_shape_1,
    const void* input_0,
    const void* input_1,
    const void* scale,
    const void* shift,
    void* output_0,
    void* output_1,
    ppl::common::TensorShape* output_shape,
    int outer,
    int inner,
    bool elementwise_affine,
    float eps,
    float in_scale_0,
    float in_scale_1,
    float out_scale,
    float out_scale_add){

    if (output_shape->GetDataFormat() != ppl::common::DATAFORMAT_NDARRAY){
        return ppl::common::RC_UNSUPPORTED;
    }
    const int32_t norm_size = inner;
    int32_t grid_size = outer;
    int max_dim = output_shape->GetDimCount();

    bool need_broadcast = false;
    for (int i = 0; i < max_dim; i++) {
        if (input_shape_0->GetDim(i) != input_shape_1->GetDim(i)){
            need_broadcast = true;
            break;
        }
    }

    if (!need_broadcast){
        if(input_shape_0->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
            constexpr int32_t VPT = 16 / sizeof(half);
            switch (norm_size)
            {
            case 128:
                ppl_cukernel_fused_add_layernorm_fp16_double_output<VPT, 128 / VPT><<<grid_size, 128 / VPT, 0, stream>>>
                ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output_0, (half*)output_1);
                break;
            case 256:
                ppl_cukernel_fused_add_layernorm_fp16_double_output<VPT, 256 / VPT><<<grid_size, 256 / VPT, 0, stream>>>
                ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output_0, (half*)output_1);
                break;
            case 320:
                ppl_cukernel_fused_add_layernorm_fp16_double_output<VPT, 320 / VPT><<<grid_size, 320 / VPT, 0, stream>>>
                ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output_0, (half*)output_1);
                break;
            case 512:
                ppl_cukernel_fused_add_layernorm_fp16_double_output<VPT, 512 / VPT><<<grid_size, 512 / VPT, 0, stream>>>
                ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output_0, (half*)output_1);
                break;
            case 640:
                ppl_cukernel_fused_add_layernorm_fp16_double_output<VPT, 640 / VPT><<<grid_size, 640 / VPT, 0, stream>>>
                ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output_0, (half*)output_1);
                break;
            case 768:
                ppl_cukernel_fused_add_layernorm_fp16_double_output<VPT, 768 / VPT><<<grid_size, 768 / VPT, 0, stream>>>
                ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output_0, (half*)output_1);
                break;
            case 1024:
                ppl_cukernel_fused_add_layernorm_fp16_double_output<VPT, 1024 / VPT><<<grid_size, 1024 / VPT, 0, stream>>>
                ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output_0, (half*)output_1);
                break;
            case 1280:
                ppl_cukernel_fused_add_layernorm_fp16_double_output<VPT, 1280 / VPT><<<grid_size, 1280 / VPT, 0, stream>>>
                ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output_0, (half*)output_1);
                break;
            case 2048:
                ppl_cukernel_fused_add_layernorm_fp16_double_output<VPT, 2048 / VPT><<<grid_size, 2048 / VPT, 0, stream>>>
                ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output_0, (half*)output_1);
                break;
            case 4096:
                ppl_cukernel_fused_add_layernorm_fp16_double_output<VPT, 4096 / VPT><<<grid_size, 4096 / VPT, 0, stream>>>
                ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output_0, (half*)output_1);
                break;
            default:
                bool if_opt = norm_size % 8 == 0 && norm_size < 1024 * VPT;
                if (if_opt){
                    ppl_cukernel_fused_add_layernorm_fp16_double_output_multiple_VPT<VPT><<<grid_size, norm_size / VPT, 0, stream>>>
                    ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output_0, (half*)output_1);
                } else {
                    ppl_cukernel_fused_add_layernorm_fp16_default<<<grid_size, 256, 0, stream>>>
                    ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output_0, (half*)output_1, true);
                }
                break;
            }
            return ppl::common::RC_SUCCESS;
        } else if(input_shape_0->GetDataType() == ppl::common::DATATYPE_INT8) {
            constexpr int32_t VPT = 4 / sizeof(int8_t);
            switch (norm_size)
            {
            case 128:
                ppl_cukernel_fused_add_layernorm_int8<VPT, 128 / VPT><<<grid_size, 128 / VPT, 0, stream>>>
                ((const int8_t*)input_0, (const int8_t*)input_1, (float*)scale, (float*)shift, eps, norm_size, (int8_t*)output_0, (int8_t*)output_1, in_scale_0, in_scale_1, out_scale, out_scale_add);
                break;
            case 256:
                ppl_cukernel_fused_add_layernorm_int8<VPT, 256 / VPT><<<grid_size, 256 / VPT, 0, stream>>>
                ((const int8_t*)input_0, (const int8_t*)input_1, (float*)scale, (float*)shift, eps, norm_size, (int8_t*)output_0, (int8_t*)output_1, in_scale_0, in_scale_1, out_scale, out_scale_add);
                break;
            case 320:
                ppl_cukernel_fused_add_layernorm_int8<VPT, 320 / VPT><<<grid_size, 320 / VPT, 0, stream>>>
                ((const int8_t*)input_0, (const int8_t*)input_1, (float*)scale, (float*)shift, eps, norm_size, (int8_t*)output_0, (int8_t*)output_1, in_scale_0, in_scale_1, out_scale, out_scale_add);
                break;
            case 512:
                ppl_cukernel_fused_add_layernorm_int8<VPT, 512 / VPT><<<grid_size, 512 / VPT, 0, stream>>>
                ((const int8_t*)input_0, (const int8_t*)input_1, (float*)scale, (float*)shift, eps, norm_size, (int8_t*)output_0, (int8_t*)output_1, in_scale_0, in_scale_1, out_scale, out_scale_add);
                break;
            case 640:
                ppl_cukernel_fused_add_layernorm_int8<VPT, 640 / VPT><<<grid_size, 640 / VPT, 0, stream>>>
                ((const int8_t*)input_0, (const int8_t*)input_1, (float*)scale, (float*)shift, eps, norm_size, (int8_t*)output_0, (int8_t*)output_1, in_scale_0, in_scale_1, out_scale, out_scale_add);
                break;
            case 768:
                ppl_cukernel_fused_add_layernorm_int8<VPT, 768 / VPT><<<grid_size, 768 / VPT, 0, stream>>>
                ((const int8_t*)input_0, (const int8_t*)input_1, (float*)scale, (float*)shift, eps, norm_size, (int8_t*)output_0, (int8_t*)output_1, in_scale_0, in_scale_1, out_scale, out_scale_add);
                break;
            case 1024:
                ppl_cukernel_fused_add_layernorm_int8<VPT, 1024 / VPT><<<grid_size, 1024 / VPT, 0, stream>>>
                ((const int8_t*)input_0, (const int8_t*)input_1, (float*)scale, (float*)shift, eps, norm_size, (int8_t*)output_0, (int8_t*)output_1, in_scale_0, in_scale_1, out_scale, out_scale_add);
                break;
            case 1280:
                ppl_cukernel_fused_add_layernorm_int8<VPT, 1280 / VPT><<<grid_size, 1280 / VPT, 0, stream>>>
                ((const int8_t*)input_0, (const int8_t*)input_1, (float*)scale, (float*)shift, eps, norm_size, (int8_t*)output_0, (int8_t*)output_1, in_scale_0, in_scale_1, out_scale, out_scale_add);
                break;
            case 2048:
                ppl_cukernel_fused_add_layernorm_int8<VPT, 2048 / VPT><<<grid_size, 2048 / VPT, 0, stream>>>
                ((const int8_t*)input_0, (const int8_t*)input_1, (float*)scale, (float*)shift, eps, norm_size, (int8_t*)output_0, (int8_t*)output_1, in_scale_0, in_scale_1, out_scale, out_scale_add);
                break;
            case 4096:
                ppl_cukernel_fused_add_layernorm_int8<VPT, 4096 / VPT><<<grid_size, 4096 / VPT, 0, stream>>>
                ((const int8_t*)input_0, (const int8_t*)input_1, (float*)scale, (float*)shift, eps, norm_size, (int8_t*)output_0, (int8_t*)output_1, in_scale_0, in_scale_1, out_scale, out_scale_add);
                break;
            default:
                bool if_opt = norm_size % 8 == 0 && norm_size < 1024 * VPT;
                if (if_opt){
                    ppl_cukernel_fused_add_layernorm_int8_multiple_VPT<VPT><<<grid_size, norm_size / VPT, 0, stream>>>
                    ((const int8_t*)input_0, (const int8_t*)input_1, (float*)scale, (float*)shift, eps, norm_size, (int8_t*)output_0, (int8_t*)output_1, in_scale_0, in_scale_1, out_scale, out_scale_add);
                }else{
                    ppl_cukernel_fused_add_layernorm_int8_default<<<grid_size, 256, 0, stream>>>
                    ((const int8_t*)input_0, (const int8_t*)input_1, (float*)scale, (float*)shift, eps, norm_size, (int8_t*)output_0, (int8_t*)output_1, true, in_scale_0, in_scale_1, out_scale, out_scale_add);
                }
                break;
            }
            return ppl::common::RC_SUCCESS;
        } else {
            return ppl::common::RC_UNSUPPORTED;
        }
    } else {
        int dim_0 = input_shape_0->GetDimCount();
        int dim_1 = input_shape_1->GetDimCount();
        bool easy_broadcast = false;
        for (int i = max_dim - 1; i >= 0; i--){
            if((i - max_dim + dim_0) >= 0 && (i - max_dim + dim_1) >= 0){
                if(input_shape_0->GetDim(i - max_dim + dim_0) == input_shape_1->GetDim(i - max_dim + dim_1)){
                    continue;
                }
                if((i - max_dim + dim_0) > 0 && (i - max_dim + dim_1) > 0){
                    bool all_one_0 = true;
                    bool all_one_1 = true;
                    for (int j = i - 1; j >= 0; j--) {
                        if ((j - max_dim + dim_0) >= 0){
                            if(input_shape_0->GetDim(j - max_dim + dim_0) != 1){
                                all_one_0 = false;
                            }
                        }
                        if ((j - max_dim + dim_1) >= 0){
                            if(input_shape_1->GetDim(j - max_dim + dim_1) != 1){
                                all_one_1 = false;
                            }
                        }
                    }
                    easy_broadcast = all_one_0 || all_one_1;
                    break;
                }
                if ((i - max_dim + dim_0) == 0 && (i - max_dim + dim_1) == 0){
                    easy_broadcast = true;
                    break;
                }
                if((i - max_dim + dim_0) == 0) {
                    if (input_shape_0->GetDim(i - max_dim + dim_0) == 1){
                        easy_broadcast = true;
                        break;
                    }
                    for (int j = i - 1; j >= 0; j++){
                        if ((j - max_dim + dim_1) >= 0){
                            if(input_shape_1->GetDim(j - max_dim + dim_1) != 1){
                                easy_broadcast = false;
                                break;
                            }
                        }
                    }
                    easy_broadcast = true;
                    break;
                }
                if ((i - max_dim + dim_1) == 0) {
                    if (input_shape_1->GetDim(i - max_dim + dim_1) == 1){
                        easy_broadcast = true;
                        break;
                    }
                    for (int j = i - 1; j >= 0; j++){
                        if ((j - max_dim + dim_0) >= 0){
                            if(input_shape_0->GetDim(j - max_dim + dim_0) != 1){
                                easy_broadcast = false;
                                break;
                            }
                        }
                    }
                    easy_broadcast = true;
                    break;
                }
            }
        }

        if (easy_broadcast){
            int num_input0 = 1;
            int num_input1 = 1;
            for (int i = 0; i < dim_0; i++){
                num_input0 *= input_shape_0->GetDim(i);
            }
            for (int i = 0; i < dim_1; i++){
                num_input1 *= input_shape_1->GetDim(i);
            }
            int div_num = num_input0 > num_input1 ? num_input1 : num_input0;
            if(input_shape_0->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
                constexpr int32_t VPT = 16 / sizeof(half);
                switch (norm_size)
                {
                case 128:
                    ppl_cukernel_fused_add_layernorm_fp16_double_output_easy_broadcast<VPT, 128 / VPT><<<grid_size, 128 / VPT, 0, stream>>>
                    ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output_0, (half*)output_1, div_num, (bool)(num_input0 > num_input1));
                    break;
                case 256:
                    ppl_cukernel_fused_add_layernorm_fp16_double_output_easy_broadcast<VPT, 256 / VPT><<<grid_size, 256 / VPT, 0, stream>>>
                    ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output_0, (half*)output_1, div_num, (bool)(num_input0 > num_input1));
                    break;
                case 320:
                    ppl_cukernel_fused_add_layernorm_fp16_double_output_easy_broadcast<VPT, 320 / VPT><<<grid_size, 320 / VPT, 0, stream>>>
                    ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output_0, (half*)output_1, div_num, (bool)(num_input0 > num_input1));
                    break;
                case 512:
                    ppl_cukernel_fused_add_layernorm_fp16_double_output_easy_broadcast<VPT, 512 / VPT><<<grid_size, 512 / VPT, 0, stream>>>
                    ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output_0, (half*)output_1, div_num, (bool)(num_input0 > num_input1));
                    break;
                case 640:
                    ppl_cukernel_fused_add_layernorm_fp16_double_output_easy_broadcast<VPT, 640 / VPT><<<grid_size, 640 / VPT, 0, stream>>>
                    ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output_0, (half*)output_1, div_num, (bool)(num_input0 > num_input1));
                    break;
                case 768:
                    ppl_cukernel_fused_add_layernorm_fp16_double_output_easy_broadcast<VPT, 768 / VPT><<<grid_size, 768 / VPT, 0, stream>>>
                    ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output_0, (half*)output_1, div_num, (bool)(num_input0 > num_input1));
                    break;
                case 1024:
                    ppl_cukernel_fused_add_layernorm_fp16_double_output_easy_broadcast<VPT, 1024 / VPT><<<grid_size, 1024 / VPT, 0, stream>>>
                    ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output_0, (half*)output_1, div_num, (bool)(num_input0 > num_input1));
                    break;
                case 1280:
                    ppl_cukernel_fused_add_layernorm_fp16_double_output_easy_broadcast<VPT, 1280 / VPT><<<grid_size, 1280 / VPT, 0, stream>>>
                    ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output_0, (half*)output_1, div_num, (bool)(num_input0 > num_input1));
                    break;
                case 2048:
                    ppl_cukernel_fused_add_layernorm_fp16_double_output_easy_broadcast<VPT, 2048 / VPT><<<grid_size, 2048 / VPT, 0, stream>>>
                    ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output_0, (half*)output_1, div_num, (bool)(num_input0 > num_input1));
                    break;
                case 4096:
                    ppl_cukernel_fused_add_layernorm_fp16_double_output_easy_broadcast<VPT, 4096 / VPT><<<grid_size, 4096 / VPT, 0, stream>>>
                    ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output_0, (half*)output_1, div_num, (bool)(num_input0 > num_input1));
                    break;
                default:
                    bool if_opt = norm_size % 8 == 0 && norm_size < 1024 * VPT;
                    if (if_opt){
                        ppl_cukernel_fused_add_layernorm_fp16_double_output_easy_broadcast_multiple_VPT<VPT><<<grid_size, norm_size / VPT, 0, stream>>>
                        ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output_0, (half*)output_1, div_num, (bool)(num_input0 > num_input1));
                    }else {
                        ppl_cukernel_fused_add_layernorm_fp16_easybroadcast_default<<<grid_size, 256, 0, stream>>>
                        ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output_0, (half*)output_1, true, num_input0, num_input1);
                    }
                    break;
                }
                return ppl::common::RC_SUCCESS;
            } else {
                return ppl::common::RC_UNSUPPORTED;
            }
        }else{
            broadcast_param param;
            int num_elems = 1;
            int num_elems_input_0 = 1;
            int num_elems_input_1 = 1;

            for (int32_t i = 0; i < max_dim; i++) {
                int count = output_shape->GetDim(i);
                num_elems *= count;
                if(i < max_dim - dim_0){
                    param.input_shape_0_broad[i] = 1;
                } else {
                    param.input_shape_0_broad[i] = input_shape_0->GetDim(i - max_dim + dim_0);
                }
                if(i < max_dim - dim_1){
                    param.input_shape_1_broad[i] = 1;
                } else {
                    param.input_shape_1_broad[i] = input_shape_1->GetDim(i - max_dim + dim_1);
                }
                num_elems_input_0 *= param.input_shape_0_broad[i];
                num_elems_input_1 *= param.input_shape_1_broad[i];
            }

            int div_input_num = 1;
            int mul_vec_input0_num = 1;
            int mul_vec_input1_num = 1;
            for (int32_t i = 0; i < max_dim; i++) {
                param.mod_vec_input[i] = num_elems / div_input_num;
                div_input_num *= output_shape->GetDim(i);
                param.div_input[i] = num_elems / div_input_num;

                mul_vec_input0_num *= param.input_shape_0_broad[i];
                mul_vec_input1_num *= param.input_shape_1_broad[i];
                param.mul_vec_input0[i] = num_elems_input_0 / mul_vec_input0_num;
                param.mul_vec_input1[i] = num_elems_input_1 / mul_vec_input1_num;
            }

            if(input_shape_0->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
                constexpr int32_t VPT = 16 / sizeof(half);
                switch (norm_size)
                {
                case 128:
                    ppl_cukernel_fused_add_layernorm_fp16_double_output_broadcast<VPT, 128 / VPT><<<grid_size, 128 / VPT, 0, stream>>>
                    ((half*)input_0, (half*)input_1, input_shape_0, input_shape_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output_0, (half*)output_1, param, max_dim);
                    break;
                case 256:
                    ppl_cukernel_fused_add_layernorm_fp16_double_output_broadcast<VPT, 256 / VPT><<<grid_size, 256 / VPT, 0, stream>>>
                    ((half*)input_0, (half*)input_1, input_shape_0, input_shape_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output_0, (half*)output_1, param, max_dim);
                    break;
                case 320:
                    ppl_cukernel_fused_add_layernorm_fp16_double_output_broadcast<VPT, 320 / VPT><<<grid_size, 320 / VPT, 0, stream>>>
                    ((half*)input_0, (half*)input_1, input_shape_0, input_shape_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output_0, (half*)output_1, param, max_dim);
                    break;
                case 512:
                    ppl_cukernel_fused_add_layernorm_fp16_double_output_broadcast<VPT, 512 / VPT><<<grid_size, 512 / VPT, 0, stream>>>
                    ((half*)input_0, (half*)input_1, input_shape_0, input_shape_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output_0, (half*)output_1, param, max_dim);
                    break;
                case 640:
                    ppl_cukernel_fused_add_layernorm_fp16_double_output_broadcast<VPT, 640 / VPT><<<grid_size, 640 / VPT, 0, stream>>>
                    ((half*)input_0, (half*)input_1, input_shape_0, input_shape_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output_0, (half*)output_1, param, max_dim);
                    break;
                case 768:
                    ppl_cukernel_fused_add_layernorm_fp16_double_output_broadcast<VPT, 768 / VPT><<<grid_size, 768 / VPT, 0, stream>>>
                    ((half*)input_0, (half*)input_1, input_shape_0, input_shape_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output_0, (half*)output_1, param, max_dim);
                    break;
                case 1024:
                    ppl_cukernel_fused_add_layernorm_fp16_double_output_broadcast<VPT, 1024 / VPT><<<grid_size, 1024 / VPT, 0, stream>>>
                    ((half*)input_0, (half*)input_1, input_shape_0, input_shape_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output_0, (half*)output_1, param, max_dim);
                    break;
                case 1280:
                    ppl_cukernel_fused_add_layernorm_fp16_double_output_broadcast<VPT, 1280 / VPT><<<grid_size, 1280 / VPT, 0, stream>>>
                    ((half*)input_0, (half*)input_1, input_shape_0, input_shape_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output_0, (half*)output_1, param, max_dim);
                    break;
                case 2048:
                    ppl_cukernel_fused_add_layernorm_fp16_double_output_broadcast<VPT, 2048 / VPT><<<grid_size, 2048 / VPT, 0, stream>>>
                    ((half*)input_0, (half*)input_1, input_shape_0, input_shape_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output_0, (half*)output_1, param, max_dim);
                    break;
                case 4096:
                    ppl_cukernel_fused_add_layernorm_fp16_double_output_broadcast<VPT, 4096 / VPT><<<grid_size, 4096 / VPT, 0, stream>>>
                    ((half*)input_0, (half*)input_1, input_shape_0, input_shape_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output_0, (half*)output_1, param, max_dim);
                    break;
                default:
                    bool if_opt = norm_size % 8 == 0 && norm_size < 1024 * VPT;
                    if (if_opt){
                        ppl_cukernel_fused_add_layernorm_fp16_double_output_broadcast_multiple_VPT<VPT><<<grid_size, norm_size / VPT, 0, stream>>>
                        ((half*)input_0, (half*)input_1, input_shape_0, input_shape_1, (half*)scale, (half*)shift, eps, norm_size, (half*)output_0, (half*)output_1, param, max_dim);
                    }else {
                        ppl_cukernel_fused_add_layernorm_fp16_broadcast_default<<<grid_size, 256, 0, stream>>>
                        ((half*)input_0, (half*)input_1, (half*)scale, (half*)shift, eps, (half*)output_0, (half*)output_1, true, max_dim, norm_size, param);
                    }
                    break;
                }
                return ppl::common::RC_SUCCESS;
            } else {
                return ppl::common::RC_UNSUPPORTED;
            }
        }
    }
    return ppl::common::RC_UNSUPPORTED;
}
