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

#ifndef PPLCUDA_REDUCE_REDUCE_ROW_KERNEL_H_
#define PPLCUDA_REDUCE_REDUCE_ROW_KERNEL_H_
#include "cudakernel/reduce/reduce_helper.h"
#include "cudakernel/reduce/block_warp_reduce.h"
#include "cudakernel/common/atomic.h"
#include "cudakernel/math/operators.h"
/*
 *  [BX, 1]
 *  [Gx, 1]
 *  length reduce length
 *  num_elements
 */
template <typename T, class Operator, int ReduceSize, bool MultiBlock>
__inline__ __device__ void ppl_reduce_all(Operator op, PPLReduceDimDes des, ReduceParam param)
{
    typedef typename Operator::srctype src_type;
    typedef typename Operator::acctype acc_type;
    typedef typename Operator::dsttype dst_type;

    __shared__ acc_type shared_data[ReduceSize];
    T val = (T)op.InitVal();

    int64_t grid_stride = blockIdx.x * des.num_elements;
    for (int64_t tid = threadIdx.x; tid < des.num_elements; tid += blockDim.x) {
        acc_type tmp = (tid + grid_stride) < des.n_reduce ? (acc_type)op.fetch(tid + grid_stride) : (acc_type)op.InitVal();
        val          = op.compute(val, tmp);
    }
    shared_data[threadIdx.x] = val;

    __syncthreads();
    block_reduce_row<T, Operator, ReduceSize>(op, shared_data, val);
    __syncthreads();
    if (MultiBlock) {
        if (threadIdx.x == 0) {
            if (param == ReduceMean) {
                val = Math<acc_type, dst_type, acc_type>::div(val, (long long)des.n_reduce);
            }
            PPLAtomicWrite<dst_type, Operator>(op.dst, static_cast<dst_type>(val), op);
        }
    }

    else {
        if (threadIdx.x == 0) {
            if (param == ReduceMean) {
                val = Math<acc_type, dst_type, acc_type>::div(val, (long long)des.n_reduce);
            }
            op.out(0, static_cast<dst_type>(val));
        }
    }
}

template <typename T, class Operator, int ReduceSize, bool MultiBlock>
__inline __device__ void ppl_reduce_row_li(Operator op, PPLReduceDimDes des, ReduceParam param)
{
    typedef typename Operator::srctype src_type;
    typedef typename Operator::acctype acc_type;
    typedef typename Operator::dsttype dst_type;

    __shared__ acc_type shared_data[ReduceSize];
    int64_t tid                = threadIdx.x + threadIdx.y * blockDim.x;
    T val                      = (T)op.InitVal();
    int64_t offset             = blockIdx.x * des.n_reduce;
    int64_t blocksize          = blockDim.x * blockDim.y;
    int64_t multi_block_offset = blockIdx.y * des.num_elements;
    for (int64_t i = tid; i < des.num_elements; i += blocksize) {
        acc_type tmp = multi_block_offset + i < des.n_reduce ? (acc_type)op.fetch(i + offset + multi_block_offset) : (acc_type)op.InitVal();
        val          = op.compute(val, tmp);
    }
    shared_data[tid] = val;
    __syncthreads();
    block_reduce_row<T, Operator, ReduceSize>(op, shared_data, val);
    __syncthreads();
    if (MultiBlock) {
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            if (param == ReduceMean) {
                shared_data[0] = Math<acc_type, dst_type, acc_type>::div(shared_data[0], (long long)des.n_reduce);
            }
            PPLAtomicWrite<dst_type, Operator>(op.dst + blockIdx.x, static_cast<dst_type>(shared_data[0]), op);
        }
    } else {
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            if (param == ReduceMean) {
                shared_data[0] = Math<acc_type, dst_type, acc_type>::div(shared_data[0], (long long)des.n_reduce);
            }
            op.out(blockIdx.x, shared_data[0]);
        }
    }
}

/*

*/

template <typename T, class Operator, int ReduceSize, bool MultiBlock>
__inline__ __device__ void ppl_reduce_row_si(Operator op, PPLReduceDimDes des, ReduceParam param)
{
    typedef typename Operator::srctype src_type;
    typedef typename Operator::acctype acc_type;
    typedef typename Operator::dsttype dst_type;

    const uint64_t block_size  = blockDim.x * blockDim.y;
    const uint64_t grid_stride = gridDim.x * block_size;
    const int base             = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * block_size;
    for (uint64_t outer = base; outer < des.n_outer; outer += grid_stride) {
        T val           = (T)op.InitVal();
        uint64_t offset = outer * des.n_reduce;
        for (uint64_t i = 0; i < des.n_reduce; i++) {
            T tmp = (T)op.fetch(offset + i);
            val   = op.compute(val, tmp);
        }
        if (param == ReduceMean) {
            val = Math<acc_type, dst_type, acc_type>::div(val, (long long)des.n_reduce);
        }
        op.out(outer, val);
    }
}

#ifdef PPLNN_USE_MACA
//[32,32]
//Notes:
//Each thread will process 4 times data then original implementation, and
//thread number will decrease 4 times
//1. the outer loop only run once, but compile do not know, and compiler will trying to
//   calculate the loop times so a `n_outer / grid_stride` is generated which will rapidly
//   slow the kernel
//2. `uint64_t offset` seems to generate more instructions than `uint32_t offset` and will
//   certainly cost more time
//3. each thread will process 4 rows which will hide a lot of load->arrive latency
//4. use `__shlf_down_sync` with 4 values seems will hide bsm arrive latency
//5. div 4 values with same base in one thread will be faster than div in different threads
//6. a store of 4 values will combine into stg_128 instruction for T=float
//TODO:
//1. I only do optimization on ppl_reduce_row_mi, but on ppl_reduce_row_si, ppl_reduce_row_li
//   the optimization should also have effects
//2. does 8x or 16x computation in one thread will gain more efficiency?
template <typename T, class Operator, int ReduceSize, bool MultiBlock, int ReduceType, bool USE_4X>
__inline__ __device__ void ppl_reduce_row_mi(Operator op, PPLReduceDimDes des)
{
    //typedef typename Operator::srctype src_type;
    typedef typename Operator::acctype acc_type;
    typedef typename Operator::dsttype dst_type;

    //const uint64_t grid_stride = gridDim.x * blockDim.y;
    const uint32_t base             = threadIdx.y + blockIdx.x * blockDim.y;
    uint32_t n_reduce = des.n_reduce;
    //for (uint64_t outer = base*4; outer < des.n_outer; outer += grid_stride) {
    if (USE_4X) {
        constexpr unsigned FULL_MASK = 0xffffffff;
        uint32_t outer = base * 4;
        T val0           = (T)op.InitVal();
        T val1           = (T)op.InitVal();
        T val2           = (T)op.InitVal();
        T val3           = (T)op.InitVal();
        uint32_t offset0 = outer * n_reduce;
        uint32_t offset1 = offset0 + n_reduce;
        uint32_t offset2 = offset1 + n_reduce;
        uint32_t offset3 = offset2 + n_reduce;
        for (uint32_t i = threadIdx.x; i < n_reduce; i += blockDim.x) {
            val0 = op.compute(val0, (T)op.fetch(offset0 + i));
            val1 = op.compute(val1, (T)op.fetch(offset1 + i));
            val2 = op.compute(val2, (T)op.fetch(offset2 + i));
            val3 = op.compute(val3, (T)op.fetch(offset3 + i));
        }
        if (ReduceSize >= 32) {
            val0 = op.compute(val0, __shfl_down_sync(FULL_MASK, val0, 16));
            val1 = op.compute(val1, __shfl_down_sync(FULL_MASK, val1, 16));
            val2 = op.compute(val2, __shfl_down_sync(FULL_MASK, val2, 16));
            val3 = op.compute(val3, __shfl_down_sync(FULL_MASK, val3, 16));
        }
        if (ReduceSize >= 16) {
            val0 = op.compute(val0, __shfl_down_sync(FULL_MASK, val0, 8));
            val1 = op.compute(val1, __shfl_down_sync(FULL_MASK, val1, 8));
            val2 = op.compute(val2, __shfl_down_sync(FULL_MASK, val2, 8));
            val3 = op.compute(val3, __shfl_down_sync(FULL_MASK, val3, 8));
        }
        if (ReduceSize >= 8) {
            val0 = op.compute(val0, __shfl_down_sync(FULL_MASK, val0, 4));
            val1 = op.compute(val1, __shfl_down_sync(FULL_MASK, val1, 4));
            val2 = op.compute(val2, __shfl_down_sync(FULL_MASK, val2, 4));
            val3 = op.compute(val3, __shfl_down_sync(FULL_MASK, val3, 4));
        }
        if (ReduceSize >= 4) {
            val0 = op.compute(val0, __shfl_down_sync(FULL_MASK, val0, 2));
            val1 = op.compute(val1, __shfl_down_sync(FULL_MASK, val1, 2));
            val2 = op.compute(val2, __shfl_down_sync(FULL_MASK, val2, 2));
            val3 = op.compute(val3, __shfl_down_sync(FULL_MASK, val3, 2));
        }
        if (ReduceSize >= 2) {
            val0 = op.compute(val0, __shfl_down_sync(FULL_MASK, val0, 1));
            val1 = op.compute(val1, __shfl_down_sync(FULL_MASK, val1, 1));
            val2 = op.compute(val2, __shfl_down_sync(FULL_MASK, val2, 1));
            val3 = op.compute(val3, __shfl_down_sync(FULL_MASK, val3, 1));
        }

        if (threadIdx.x == 0) {
            if (ReduceType == ReduceMean) {
                val0 = Math<acc_type, dst_type, acc_type>::div(val0, (long long)n_reduce);
                val1 = Math<acc_type, dst_type, acc_type>::div(val1, (long long)n_reduce);
                val2 = Math<acc_type, dst_type, acc_type>::div(val2, (long long)n_reduce);
                val3 = Math<acc_type, dst_type, acc_type>::div(val3, (long long)n_reduce);
            }
            op.out(outer, val0);
            op.out(outer+1, val1);
            op.out(outer+2, val2);
            op.out(outer+3, val3);
        }
    } else {
        const uint32_t grid_stride = gridDim.x * blockDim.y;
        for (uint32_t outer = base; outer < des.n_outer; outer += grid_stride) {
            T val           = (T)op.InitVal();
            uint32_t offset = outer * n_reduce;
            for (uint32_t i = threadIdx.x; i < n_reduce; i += blockDim.x) {
                val = op.compute(val, (T)op.fetch(offset + i));
            }
            warp_reduce_unroll<T, Operator, ReduceSize>(op, nullptr, val);
            if (threadIdx.x == 0) {
                if (ReduceType == ReduceMean) {
                    val = Math<acc_type, dst_type, acc_type>::div(val, (long long)des.n_reduce);
                }
                op.out(outer, val);
            }
        }
    }
}


template <typename T, class Operator, int ReduceSize, bool MultiBlock, int ReduceVersion, int ReduceType, bool USE_4X>
__device__ __inline__ void ppl_reduce_rows(Operator op, PPLReduceDimDes des, ReduceParam param)
{
    if (ReduceVersion == ReduceVersionSI) {
        ppl_reduce_row_si<T, Operator, ReduceSize, MultiBlock>(op, des, param);
    } else if (ReduceVersion == ReduceVersionMI) {
        ppl_reduce_row_mi<T, Operator, ReduceSize, MultiBlock, ReduceType, USE_4X>(op, des);
    } else {
        ppl_reduce_row_li<T, Operator, ReduceSize, MultiBlock>(op, des, param);
    }
}
#else
//[32,32]
template <typename T, class Operator, int ReduceSize, bool MultiBlock>
__inline__ __device__ void ppl_reduce_row_mi(Operator op, PPLReduceDimDes des, ReduceParam param)
{
    typedef typename Operator::srctype src_type;
    typedef typename Operator::acctype acc_type;
    typedef typename Operator::dsttype dst_type;

    const uint64_t grid_stride = gridDim.x * blockDim.y;
    const int base             = threadIdx.y + blockIdx.x * blockDim.y;
    for (uint64_t outer = base; outer < des.n_outer; outer += grid_stride) {
        T val           = (T)op.InitVal();
        uint64_t offset = outer * des.n_reduce;
        for (uint64_t i = threadIdx.x; i < des.n_reduce; i += blockDim.x) {
            val = op.compute(val, (T)op.fetch(offset + i));
        }
        warp_reduce_unroll<T, Operator, ReduceSize>(op, nullptr, val);
        if (threadIdx.x == 0) {
            if (param == ReduceMean) {
                val = Math<acc_type, dst_type, acc_type>::div(val, (long long)des.n_reduce);
            }
            op.out(outer, val);
        }
    }
}

template <typename T, class Operator, int ReduceSize, bool MultiBlock>
__device__ __inline__ void ppl_reduce_rows(Operator op, PPLReduceDimDes des, ReduceParam param)
{
    if (des.n_reduce < 32 && des.split_k_num == 1) {
        ppl_reduce_row_si<T, Operator, ReduceSize, MultiBlock>(op, des, param);
    } else if (des.n_reduce < 1024 && des.split_k_num == 1) {
        ppl_reduce_row_mi<T, Operator, ReduceSize, MultiBlock>(op, des, param);
    } else {
        ppl_reduce_row_li<T, Operator, ReduceSize, MultiBlock>(op, des, param);
    }
}
#endif

#endif
