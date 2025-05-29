// 2024 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#include "ppl/common/tensor_shape.h"

#include "ppl/common/retcode.h"


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
            void* output);
