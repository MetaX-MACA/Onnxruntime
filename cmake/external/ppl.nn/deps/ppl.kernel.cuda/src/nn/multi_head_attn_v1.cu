// 2024 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#include "cudakernel/nn/multi_head_attn_v1.h"
#include "ppl/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include "cudakernel/common/common.cuh"
#include "cudakernel/common/common.h"
#include <cuda_fp16.h>

#include "mctlass/bfloat16.h"
#include "mctlass/half.h"

#include "flash_attn/flash_attn.h"

#if defined (MCFLASH_MAJOR_VERSION) && (MCFLASH_MAJOR_VERSION == 2)
#define PPLNN_USE_MCFLASHATTN2
#endif//MCFLASHATTN_VERSION

// #if __has_include("flash_attn/flash_attn2.h")
// # include "flash_attn/flash_attn2.h"
// # define PPLNN_USE_MCFLASHATTN2
// #else
// # include "flash_attn/flash_attn.h"
// #endif



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


template<int32_t THREAD_GROUP_SIZE, int32_t ELEMENT_NUM>
__device__ inline
float __AttnThreadGroupDot(half *local_Q, half *local_K){
    // Helper function for QK Dot.
    // [TODO] It should be optimized by type fp32x4.

    float qk = 0.0f;
# pragma unroll
    for(int32_t i = 0; i < ELEMENT_NUM; i++) {
        qk += __half2float(local_Q[i]) * __half2float(local_K[i]);
    }
#pragma unroll
    for (int32_t mask = THREAD_GROUP_SIZE / 2; mask >= 1; mask /= 2) {
        qk += __shfl_xor_sync(uint32_t(-1), qk, mask);
    }
    return qk;
}

template<int32_t WPT>
__device__ inline
float __AttnBlockReduceMax(float reducing, float *shared_mem){
    // Helper function for reduce softmax qkmax.
    constexpr int32_t WARP_SIZE = 32;
    const int32_t lane_id = threadIdx.x % WARP_SIZE;
    const int32_t warp_id = threadIdx.x / WARP_SIZE;

# pragma unroll
    for (int32_t mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
        reducing = fmaxf(reducing, __shfl_xor_sync(uint32_t(-1), reducing, mask));
    }

    if (lane_id == 0) {
        shared_mem[warp_id] = reducing;
    }
    __syncthreads();

    if (lane_id < WPT) reducing = shared_mem[lane_id];
    else reducing = -FLT_MAX;

# pragma unroll
    for (int32_t mask = WPT / 2; mask >= 1; mask /= 2) {
        reducing = fmaxf(reducing, __shfl_xor_sync(uint32_t(-1), reducing, mask));
    }

    reducing = __shfl_sync(uint32_t(-1), reducing, 0);
    return reducing;
}

template<int32_t WPT>
__device__ inline
float __AttnBlockReduceSum(float reducing, float *shared_mem){
    // Helper function for reduce softmax exp sum.
    constexpr int32_t WARP_SIZE = 32;
    const int32_t lane_id = threadIdx.x % WARP_SIZE;
    const int32_t warp_id = threadIdx.x / WARP_SIZE;

# pragma unroll
    for (int32_t mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
        reducing += __shfl_xor_sync(uint32_t(-1), reducing, mask);
    }

    if (lane_id == 0) shared_mem[warp_id] = reducing;
    __syncthreads();

    if (lane_id < WPT) reducing = shared_mem[lane_id];

# pragma unroll
    for (int32_t mask = WPT / 2; mask >= 1; mask /= 2) {
        reducing += __shfl_xor_sync(uint32_t(-1), reducing, mask);
    }
    reducing = __shfl_sync(uint32_t(-1), reducing, 0);
    return reducing;
}

template<
    int32_t HEAD_SIZE,
    int32_t THREAD_GROUP_SIZE,        // how many threads inside a group
    int32_t TPB>
__global__
void _DecodingAttention_fp16(
    half* __restrict__ out,          // [context_lens, num_heads, head_size]
    const half* __restrict__ Q,      // [seq_lens, num_heads, head_size]
    const half* __restrict__ K,      // [context_lens, num_heads, head_size]
    const half* __restrict__ V,      // [context_lens, num_heads, head_size]
    const float  attn_scale,
    const int64_t *context_lens_cumsum // something like: [0, 160, 160+2, 160+2+3]
) {
    /***
    * You have to remember that this Kernel was created by a brother on the night of July 20, 2023. On that day,
    * Beijing experienced the strongest rainstorm since the beginning of summer.

    DecodingAttention is a special operator designed specifically for large language models(LLM) decoding.

    It requires that the length of each input Query is always 1,
        while the Key and Value can have different lengths.

    This operator supports padding removal optimization, meaning that Q, K, and V all need to have their tokens
        concentrated in one sentence for input, with shapes like Q: [seq_lens, num_heads, head_size],
        and K: [context_lens, num_heads, head_size].

    Since the Query sentence length is always 1, this operator is literally a fused matrix-vector multiplications operation.
        It does not utilize tensor cores for computation.

    The calculation logic is divided into three steps: gemv(QK) + softmax(Attention) + gemv(KV).
        In the provided code, it has already been split into these three parts.
    ***/

    /* --- Decoding Attention Kernel Implementation --- */
    constexpr int64_t WARP_SIZE = 32;                              // warp size
    constexpr int64_t WPT       = TPB / WARP_SIZE;                 // warp per thread block
    constexpr int64_t GPW       = WARP_SIZE / THREAD_GROUP_SIZE;       // thread group per warp
    constexpr int64_t GPT       = WARP_SIZE / THREAD_GROUP_SIZE * WPT; // thread group per thread block

    const int64_t head_idx      = blockIdx.x;
    const int64_t seq_idx       = blockIdx.y;
    const int64_t num_heads     = gridDim.x;
    // const int64_t num_seqs      = gridDim.y;
    constexpr int64_t VEC_SIZE  = 16 / sizeof(half);  // 128 bits

    // ------------------------------------------------ //
    // Step 1. Load Q into Thread Reg.
    constexpr int64_t VEC_LEN = (HEAD_SIZE / VEC_SIZE) / THREAD_GROUP_SIZE;

    static_assert((HEAD_SIZE / THREAD_GROUP_SIZE) % VEC_SIZE == 0);
    static_assert(HEAD_SIZE % THREAD_GROUP_SIZE == 0);

    // The elements in Q, K, and V will be evenly distributed across each thread group.
    half local_Q[VEC_SIZE * VEC_LEN];

    const int64_t warp_id       = threadIdx.x / WARP_SIZE;
    const int64_t warp_lane_id  = threadIdx.x % WARP_SIZE;
    const int64_t group_id      = warp_lane_id / THREAD_GROUP_SIZE;
    const int64_t group_lane_id = warp_lane_id % THREAD_GROUP_SIZE;
    const int64_t token_offset  = context_lens_cumsum[seq_idx];
    const int64_t token_stride  = num_heads * HEAD_SIZE;

    #pragma unroll
    for (int64_t i = 0; i < VEC_LEN; i++) {
        // copy 128(16 * 8) bits from Q to Local Q

        copy<sizeof(half) * VEC_SIZE>(
            &Q[seq_idx * token_stride + head_idx * HEAD_SIZE +
            (group_lane_id + i * THREAD_GROUP_SIZE) * VEC_SIZE],
            &local_Q[i * VEC_SIZE]);
    }
    // ------------------------------------------------ //
    // Step 2. Solve QK Dot

    // In the process of handling the QK matrix multiplication, we will divide a complete Thread Warp into several Thread groups.
    // Each thread group reads the entire Query and saves it in registers.
    // Then, each thread group iterates through the vectors in the Key and performs dot products with the Query.
    // During this process, a WARP performs multiple vector dot product operations at once.
    // At the same time, we also record the maximum value of the dot product results for later use in the softmax operation.
    const int64_t context_len = context_lens_cumsum[seq_idx + 1] - context_lens_cumsum[seq_idx];
    extern __shared__ float logits[];
    float qk_max = -FLT_MAX;

    for (int64_t base_id = warp_id * GPW; base_id < context_len; base_id += GPT) {
        half local_K[VEC_SIZE * VEC_LEN];
        int64_t token_id = base_id + group_id;

        // all thread groups within a warp must be launched together.
        if (token_id >= context_len){
            // memset(local_K, 0, sizeof(local_K));
            #pragma unroll
            for(int i = 0; i < VEC_SIZE * VEC_LEN; i++) {
                local_K[i] = 0;
            }
        }
        else{
        # pragma unroll
            for (int64_t i = 0; i < VEC_LEN; i++) {
                // copy 128(16 * 8) bits from K to Local K
                copy<sizeof(half) * VEC_SIZE>(
                    &K[(token_offset + token_id) * token_stride + head_idx * HEAD_SIZE +
                    (group_lane_id + i * THREAD_GROUP_SIZE) * VEC_SIZE],
                    &local_K[i * VEC_SIZE]);
            }
        }

        // Ready for QK Dot
        float qk_dot = attn_scale * __AttnThreadGroupDot<THREAD_GROUP_SIZE, VEC_LEN * VEC_SIZE>(local_Q, local_K);

        if (group_lane_id == 0 && token_id < context_len) {
            logits[token_id] = qk_dot;
            qk_max = fmaxf(qk_dot, qk_max);
        }
    }

    // ------------------------------------------------ //
    // Step 3. Softmax

    // The process of solving softmax is divided into two stages.
    // First, we need to reduce qk_max in two dimensions: WARP and ThreadBlock.
    // Afterward, we use reduced qk_max to perform softmax calculations,
    //    the results will all be stored in shared memory.
    __shared__ float red_smem[WPT];

    // reduce qk_max in thread block and boardcast
    qk_max = __AttnBlockReduceMax<WPT>(qk_max, red_smem);

    // Softmax Kernel Logic Start here
    float exp_sum = 0.0f;
    for (int64_t token_id = threadIdx.x; token_id < context_len; token_id += TPB){
        logits[token_id] -= qk_max;
        logits[token_id] = exp(logits[token_id]);
        exp_sum += logits[token_id];
    }

    // block reduce sum on exp_sum
    // Warp per thread block must be power-of-2 for reducation, check __AttnBlockReduceSum kernel.
    static_assert(WPT == 2 || WPT == 4 || WPT == 8 || WPT == 16 || WPT == 32 || WPT == 64);
    exp_sum = __AttnBlockReduceSum<WPT>(exp_sum, red_smem);

    const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
    for (int64_t token_id = threadIdx.x; token_id < context_len; token_id += TPB) {
        logits[token_id] *= inv_sum;
    }
    __syncthreads(); // Must have this.

    // ------------------------------------------------ //
    // Step 4. Solve logits * V, this part should be carefully optimized. [TODO]
    for(int64_t i = threadIdx.x; i < HEAD_SIZE; i += TPB){
        float local_acc = 0.0f;

        for(int64_t token_id = 0; token_id < context_len; token_id++) {
            float logit = logits[token_id];
            float value = __half2float(V[
                (token_offset + token_id) * token_stride +
                head_idx * HEAD_SIZE + i
            ]);
            local_acc = fma(value, logit, local_acc);
        }

        out[seq_idx * token_stride + head_idx * HEAD_SIZE + i] = __float2half(local_acc);
    }
}

#ifndef  PPLNN_USE_MCFLASHATTN2
void set_params_fprop(mcFlashAttn::Flash_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      void *q, //const at::Tensor q,
                      void *k, //const at::Tensor k,
                      void *v, //const at::Tensor v,
                      void *out, //at::Tensor out,
                      void* cu_seqlens_q_d, //int* cu_seqlens_q_d, //void *cu_seqlens_q_d,
                      void* cu_seqlens_k_d, //int* cu_seqlens_k_d, //void *cu_seqlens_k_d,
                      void *p_d,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      bool is_causal,
                      const size_t head_dim,
                      bool has_attn_mask,
                      void *mask,
                      const size_t attn_mask_batch_stride) {

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.is_bf16 = 0; //q.dtype() == torch::kBFloat16;

    // Set the pointers and strides.
    params.q_ptr = q; //q.data_ptr();
    params.k_ptr = k; //k.data_ptr();
    params.v_ptr = v; //v.data_ptr();
    // All stride are in elements, not bytes.
    params.q_row_stride = h*head_dim; //q.stride(-3); //num_head*head_dim
    params.k_row_stride = h*head_dim; //k.stride(-3);
    params.v_row_stride = h*head_dim; //v.stride(-3);
    params.q_head_stride = head_dim; //q.stride(-2);
    params.k_head_stride = head_dim; //k.stride(-2);
    params.v_head_stride = head_dim; //v.stride(-2);
    params.o_ptr = out; //out.data_ptr();
    params.o_row_stride = h*head_dim; //out.stride(-3);
    params.o_head_stride = head_dim; //out.stride(-2);

    if (cu_seqlens_q_d == nullptr) {
        params.q_batch_stride =  (uint64_t)(seqlen_q * h * head_dim); //q.stride(0);
        params.k_batch_stride =  (uint64_t)(seqlen_k * h * head_dim); //k.stride(0);
        params.v_batch_stride =  (uint64_t)(seqlen_k * h * head_dim); //v.stride(0);
        params.o_batch_stride =  (uint64_t)(seqlen_q * h * head_dim); //out.stride(0);
    }
    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);


    // P = softmax(QK^T)
    params.p_ptr = p_d;

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    // Set the different scale values.
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
    //TORCH_CHECK(p_dropout < 1.f);

    params.is_causal = is_causal;
    int window_size_left = -1;
    int window_size_right= -1;
    if(is_causal) {
        window_size_right = 0;
    }

    if (window_size_left < 0 && window_size_right >= 0) { window_size_left = seqlen_k; }
    if (window_size_left >= 0 && window_size_right < 0) { window_size_right = seqlen_k; }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    params.is_seqlens_k_cumulative = true;

    params.has_attn_mask = has_attn_mask;
    params.attn_mask_ptr = mask;
    params.attn_mask_batch_stride = (uint64_t)(attn_mask_batch_stride);

}


template <typename T>
void LaunchMhaFwdKernel(mcFlashAttn::Flash_fwd_params &params, cudaStream_t stream) {

    if (std::getenv("MHA_PRINT_PARA")) {
        printf("==============Forward pass parameters recored start...\n");
        printf("----Bool switch:  Is_causal=%d, is_bf16=%d, is_causal=%d\n",
            params.is_causal, params.is_bf16, params.is_causal);
        printf("----dim: h=%d,h_k=%d,h_h_k_ratio=%d,b=%d, seqlen_q=%d, seqlen_k=%d, d=%d, seqlen_q_rounded=%d, seqlen_k_rounded=%d, d_rounded=%d\n",
            params.h, params.h_k, params.h_h_k_ratio, params.b, params.seqlen_q,params.seqlen_k,params.d,params.seqlen_q_rounded,params.seqlen_k_rounded,params.d_rounded);
        printf("----params.cu_seqlens_q=%p\n",params.cu_seqlens_q);

        printf("-----params.blockmask=%p\n",params.blockmask);
        printf("----params.q_ptr=%p\n", params.q_ptr);
        printf("----params.k_ptr=%p\n", params.k_ptr);
        printf("----params.v_ptr=%p\n", params.v_ptr);
        printf("----params.p_ptr=%p\n", params.p_ptr);
        printf("----params.softmax_lse_ptr=%p\n", params.softmax_lse_ptr);
        printf("----p_dropout_in_uint8_t=%d\n", params.p_dropout_in_uint8_t);
        printf("----o_batch_stride=%d\n", params.o_batch_stride);
        printf("----o_row_stride=%d\n",params.o_row_stride);
        printf("----o_head_stride=%d\n",params.o_head_stride);
        printf("----q_batch_stride=%d\n",params.q_batch_stride);
        printf("----k_batch_stride=%d\n",params.k_batch_stride);
        printf("----v_batch_stride=%d\n",params.v_batch_stride);
        printf("----q_row_stride=%d\n",params.q_row_stride);
        printf("----k_row_stride=%d\n",params.k_row_stride);
        printf("----v_row_stride=%d\n",params.v_row_stride);
        printf("----q_head_stride=%d\n",params.q_head_stride);
        printf("----k_head_stride=%d\n",params.k_head_stride);
        printf("----v_head_stride=%d\n",params.v_head_stride);
        printf("==============Forward pass parameters recored end...\n");
    }


    if (params.d <= 32) {
        mcFlashAttn::run_mha_fwd_<T, 32>(params, stream);
    } else if(params.d <= 64) {
        mcFlashAttn::run_mha_fwd_<T, 64>(params, stream);
    } else if (params.d <= 96) {
        mcFlashAttn::run_mha_fwd_<T, 96>(params, stream);
    }  else if (params.d <= 128) {
        mcFlashAttn::run_mha_fwd_<T, 128>(params, stream);
    } else if (params.d <= 160) {
        mcFlashAttn::run_mha_fwd_<T, 160>(params, stream);
    } else if (params.d <= 192) {
        mcFlashAttn::run_mha_fwd_<T, 192>(params, stream);
    } else if (params.d <= 224) {
        mcFlashAttn::run_mha_fwd_<T, 224>(params, stream);
    } else if (params.d <= 256) {
        mcFlashAttn::run_mha_fwd_<T, 256>(params, stream);
    }
}


#endif //  !PPLNN_USE_MCFLASHATTN2

void printShape(const ppl::common::TensorShape* shape) {
    for (int i=0; i<shape->GetDimCount();i++) {
        printf(" %ld ", shape->GetDim(i));
    }
}


void printIndexInt(void* cudaData, int dataSize) {
    int* hostData = new int[dataSize];

    cudaMemcpy(hostData, cudaData, dataSize*sizeof(int), cudaMemcpyDeviceToHost);

    for (int i=0; i<dataSize; i++) {
        printf("host_output[%d] = %d, ", i, (hostData[i]));
        }
        printf("\n");
}

void printIndex(void* cudaData, int dataSize) {
    half* hostData = new half[dataSize];

    cudaMemcpy(hostData, cudaData, dataSize*sizeof(half), cudaMemcpyDeviceToHost);

    for (int i=0; i<dataSize; i++) {
        printf("host_output[%d] = %f, ", i, __half2float(hostData[i]));
        }
        printf("\n");
}

void printIndexFloat(void* cudaData, int dataSize) {
    float* hostData = new float[dataSize];

    cudaMemcpy(hostData, cudaData, dataSize*sizeof(float), cudaMemcpyDeviceToHost);

    for (int i=0; i<dataSize; i++) {
        printf("host_output[%d] = %f, ", i, (hostData[i]));
        }
        printf("\n");
}


ppl::common::RetCode PPLCUDAMultiHeadAttentionForwardImp(
            const cudaStream_t &stream,
            const ppl::common::TensorShape* query_shape,
            void* query,
            const ppl::common::TensorShape* key_shape,
            void* key,
            void* value,
            const ppl::common::TensorShape* mask_shape,
            void* mask,
            const bool is_causal,
            const int64_t num_heads,
            const int64_t num_kv_heads,
            const int64_t head_dim,
            const ppl::common::TensorShape* output_shape,
            void* output) {


    const int64_t batch = query_shape->GetDim(0);
    const int64_t seqlen_q = query_shape->GetDim(1);
    const int64_t seqlen_k = key_shape->GetDim(1);

    // const int64_t head_dim =key_shape->GetDim(3);
    // const int64_t num_heads = query_shape->GetDim(2);
    // const int64_t num_kv_heads = key_shape->GetDim(2);

    const float attn_scale = float(1.0 / std::sqrt(float(head_dim)));
    int64_t seqstart_size = batch + 1;
    // printf("seqlen_q: %ld,  seqlen_k: %ld\n", seqlen_q, seqlen_k);
    // printIndex(query, 10);
    // printIndex(key, 10);
    // printIndex(value, 10);

    // if (mask != nullptr)
    //     printIndex(mask, 256);


    if(seqlen_q == 1) {
        int64_t* seqstart_k;
        cudaMalloc(&seqstart_k, seqstart_size * sizeof(int64_t));
        int64_t host_data[seqstart_size];

        for (auto i=0; i<seqstart_size; i++) {
            host_data[i] = seqlen_k * i;
        }

        cudaMemcpyAsync(seqstart_k, host_data, seqstart_size * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

        const int64_t WARP_SIZE = 32;
        const int64_t TPB = 256;
        const int64_t reduce_shm_size = TPB / WARP_SIZE * sizeof(float);
        const int64_t logits_size = seqlen_k * sizeof(float);
        const int64_t MAX_SHM_SIZE = 48 * 1024;
        if (reduce_shm_size + logits_size <= MAX_SHM_SIZE) {
            const dim3 grid_size = {(unsigned int)num_heads, (unsigned int)batch, 1};
            switch (head_dim){
                case 64:
                _DecodingAttention_fp16<64, 4, 256>
                <<<grid_size, 256, logits_size, stream>>>(
                    (half*)(output), (half*)(query), (half*)(key), (half*)(value),
                    attn_scale, (int64_t*)(seqstart_k)
                );
                break;
                case 96:
                _DecodingAttention_fp16<96, 4, 256>
                <<<grid_size, 256, logits_size, stream>>>(
                    (half*)(output), (half*)(query), (half*)(key), (half*)(value),
                    attn_scale, (int64_t*)(seqstart_k)
                );
                break;
                case 128:
                _DecodingAttention_fp16<128, 8, 256>
                <<<grid_size, 256, logits_size, stream>>>(
                    (half*)(output), (half*)(query), (half*)(key), (half*)(value),
                    attn_scale, (int64_t*)(seqstart_k)
                );
                break;
                case 256:
                _DecodingAttention_fp16<256, 16, 256>
                <<<grid_size, 256, logits_size, stream>>>(
                    (half*)(output), (half*)(query), (half*)(key), (half*)(value),
                    attn_scale, (int64_t*)(seqstart_k)
                );
                break;
                default:
                return ppl::common::RC_UNSUPPORTED;  //TODO
                // PPL_CHECK(false, "Failed to invoke this Decoding Attention Kernel, Head size is unsupported.");
            }
        }
    }
    else {
        #ifdef PPLNN_USE_MCFLASHATTN2
            // Need flash attn support.
            cudaDeviceProp device_prop;
            device_prop.major = 8;
            device_prop.minor = 1;

            auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
            const int head_dim_rounded = round_multiple(head_dim, 32);
            // const int head_dim_rounded = pad_head_size(head_dim);

            const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
            const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

            // const bool p_dropout = 0;
            const bool has_attn_mask = mask != nullptr ? true : false;

            // void *softmax_lse_ptr;
            // cudaError_t err = cudaMalloc((void **)&softmax_lse_ptr, batch * num_heads * seqlen_q_rounded * sizeof(float));
            // Tensor_t softmax_lse = make_contiguous_tensor4d(softmax_lse_ptr, MCFLASHATTN_DATATYPE_FP32, batch, num_heads ,seqlen_q_rounded, seqlen_k_rounded);

            // int query_size = batch * seqlen_q * num_heads * head_dim;
            // int key_size   = batch * seqlen_k * num_heads * head_dim;
            // int value_size = batch * seqlen_k * num_heads * head_dim;
            // int output_size = query_size;

            Tensor_t q = make_contiguous_tensor4d(query, MCFLASHATTN_DATATYPE_FP16, batch, seqlen_q, num_heads, head_dim);
            Tensor_t k = make_contiguous_tensor4d(key ,  MCFLASHATTN_DATATYPE_FP16, batch, seqlen_k, num_kv_heads, head_dim);
            Tensor_t v = make_contiguous_tensor4d(value, MCFLASHATTN_DATATYPE_FP16, batch, seqlen_k, num_kv_heads, head_dim);

            Tensor_t out = make_contiguous_tensor4d(output,  MCFLASHATTN_DATATYPE_FP16, batch, seqlen_q, num_heads, head_dim);

            // print_tensor_info(q);
            // print_tensor_info(k);
            // print_tensor_info(v);

            Tensor_t attn_mask;
            if (has_attn_mask){
                if (mask_shape->GetDimCount()==4){
                    attn_mask = make_contiguous_tensor4d(mask,  MCFLASHATTN_DATATYPE_FP16,
                                                                (int)(mask_shape->GetDim(0)),
                                                                (int)(mask_shape->GetDim(1)),
                                                                (int)(mask_shape->GetDim(2)),
                                                                (int)(mask_shape->GetDim(3)));
                }else{
                    return ppl::common::RC_UNSUPPORTED;
                }

                auto status = mha_fwd_inference(batch, seqlen_q, num_heads, seqlen_k, num_heads, head_dim, q, k, v, out, NULL, attn_mask,
                                            attn_scale, is_causal, -1, -1, stream);
            }else{
                auto status = mha_fwd_inference(batch, seqlen_q, num_heads, seqlen_k, num_heads, head_dim, q, k, v, out, NULL,NULL,
                                            attn_scale, is_causal, -1, -1, stream);
            }

            // cudaFree(softmax_lse_ptr);
            return ppl::common::RC_SUCCESS;
        #else
            // Need flash attn support.
            cudaDeviceProp device_prop;
            device_prop.major = 8;
            device_prop.minor = 1;

            auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
            const int head_size = round_multiple(head_dim, 8);
            const int head_size_rounded = round_multiple(head_size, 32);
            const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
            const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

            // const bool is_causal = 0;
            const bool p_dropout = 0;

            void *softmax_lse;
            cudaError_t err = cudaMalloc((void **)&softmax_lse, batch*num_heads*seqlen_q * sizeof(float));
            const bool has_attn_mask = mask != nullptr ? true : false;

            int64_t _attn_mask_batch_stride = 0;
            if (mask_shape != nullptr) {
                _attn_mask_batch_stride = 1;
                uint32_t dimcount = mask_shape->GetDimCount();
                for (uint32_t i=1; i < dimcount; i++) {
                    _attn_mask_batch_stride *= mask_shape->GetDim(i);
                }
            }
            const int64_t attn_mask_batch_stride = _attn_mask_batch_stride;

            mcFlashAttn::Flash_fwd_params params;
            set_params_fprop(params,
                            (int)batch,
                            (int)seqlen_q, (int)seqlen_k,
                            (int)seqlen_q_rounded, (int)seqlen_k_rounded,
                            (int)num_heads, (int)num_kv_heads,
                            (int)head_size, (int)head_size_rounded,
                            query, key, value, output,
                            nullptr, // (void *)seqstart_k, // prefill_seqstart_q,
                            nullptr, // (void *)seqstart_k, //prefill_seqstart_q,
                            nullptr,
                            softmax_lse, //softmax_lse.data_ptr(),
                            p_dropout,
                            attn_scale,
                            is_causal, (int)head_dim,
                            has_attn_mask,
                            mask,
                            attn_mask_batch_stride);

            // mcFlashAttn::run_mha_fwd_<mctlass::half_t, head_size_rounded>(params, stream);
            LaunchMhaFwdKernel<mctlass::half_t>(params, stream);
            cudaFree(softmax_lse);
            return ppl::common::RC_SUCCESS;
        #endif //PPLNN_USE_MCFLASHATTN2
    }
    return ppl::common::RC_SUCCESS;

}
