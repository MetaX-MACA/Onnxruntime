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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_MATMUL_HELPER_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_MATMUL_HELPER_H_

#include "ppl/nn/engines/cuda/kernel.h"
#include "ppl/nn/engines/cuda/params/gemm_extra_param.h"
#include <cublas_api.h>

namespace ppl { namespace nn { namespace cuda {

inline void TensorShapeCopyDims(const TensorShape& shape, int64_t* dims, size_t num_dims) {
  size_t n = std::min(num_dims, static_cast<size_t>(shape.GetDimCount()));
  for (size_t i = 0; i != n; ++i)
    dims[i] = shape.GetDim(i);
}

class MatMulComputeHelper {
 public:
  ppl::common::RetCode Compute(const TensorShape& orig_left_shape, const TensorShape& orig_right_shape,
                 bool transa = false, bool transb = false,
                 bool trans_batch_a = false, bool trans_batch_b = false,
                 bool fill_offsets = true) {
    size_t left_num_dims = orig_left_shape.GetDimCount();
    size_t right_num_dims = orig_right_shape.GetDimCount();
    int64_t ele_to_dim = orig_right_shape.CalcElementsToDimensionExcludingPadding(right_num_dims - 1);

    if (!transa && !trans_batch_a && !trans_batch_b && left_num_dims >= 2 && right_num_dims >= 2 && left_num_dims >= right_num_dims &&
        ele_to_dim == orig_right_shape.GetDim(right_num_dims - 2)) {
      M_ = static_cast<int>(orig_left_shape.CalcElementsToDimensionExcludingPadding(left_num_dims - 1));
      K_ = static_cast<int>(orig_left_shape.GetDim(left_num_dims - 1));
      N_ = static_cast<int>(transb ? orig_right_shape.GetDim(right_num_dims - 2) : orig_right_shape.GetDim(right_num_dims - 1));
      output_shape_ = orig_left_shape;
      output_shape_.SetDim(left_num_dims - 1,  N_);
      output_offsets_ = {0};
      left_offsets_ = {0};
      right_offsets_ = {0};

      return ppl::common::RC_SUCCESS;
    }

    std::vector<int64_t> dims_left(left_num_dims);
    std::vector<int64_t> dims_right(right_num_dims);
    TensorShapeCopyDims(orig_left_shape, &dims_left[0], left_num_dims);
    TensorShapeCopyDims(orig_right_shape, &dims_right[0], right_num_dims);
    left_stride_factor_ = right_stride_factor_ = 1;
    left_ld_factor_ = right_ld_factor_ = 1;

    TensorShape left_shape;
    TensorShape right_shape;
    left_shape.Reshape(dims_left);
    right_shape.Reshape(dims_right);

    bool has_1D_input = (left_num_dims == 1 || right_num_dims == 1);

    size_t num_input_dims = std::max(left_num_dims, right_num_dims);

    // use padded dims to compute matrix offsets, right 1D would be padded
    size_t num_dims_with_pad = num_input_dims + (right_num_dims == 1 ? 1 : 0);

    // output shape would squeeze the reduced 1D dimension
    size_t num_output_dims = num_input_dims - (has_1D_input ? 1 : 0);

    left_padded_dims_ = std::vector<int64_t>(num_dims_with_pad, 1);
    right_padded_dims_ = std::vector<int64_t>(num_dims_with_pad, 1);

    if (right_num_dims == 1) {
      // right padded to (1,...,K,1)
      right_padded_dims_[num_dims_with_pad - 2] = static_cast<int>(right_shape.GetDim(0));

      if (num_input_dims >= 2) {
        // left padded to (...,1,K)
        TensorShapeCopyDims(left_shape, &left_padded_dims_[0], left_num_dims - 2);
        left_padded_dims_[num_dims_with_pad - 3] = static_cast<int64_t>(left_shape.GetDim(transa ? left_num_dims - 1 : left_num_dims - 2));
        left_padded_dims_[num_dims_with_pad - 1] = static_cast<int64_t>(left_shape.GetDim(transa ? left_num_dims - 2 : left_num_dims - 1));
      } else {
        // pad 1 in the front
        TensorShapeCopyDims(left_shape, &left_padded_dims_[num_dims_with_pad - left_num_dims], left_num_dims);
      }
    } else {
      // pad 1 in the front for left
      TensorShapeCopyDims(left_shape, &left_padded_dims_[num_dims_with_pad - left_num_dims], left_num_dims);
      // pad 1 in the front for right
      TensorShapeCopyDims(right_shape, &right_padded_dims_[num_dims_with_pad - right_num_dims], right_num_dims);
    }

    // validate input shape and generate output shape
    std::vector<int64_t> output_dims(num_output_dims);

    // broadcasting for all output dims except last two
    for (size_t idx_dim = 0; idx_dim < num_dims_with_pad - 2; ++idx_dim) {
      output_dims[idx_dim] = std::max(left_padded_dims_[idx_dim], right_padded_dims_[idx_dim]);
    }
    if (transa) {
      M_ = static_cast<int>(has_1D_input ? 1 : left_shape.GetDim(left_num_dims - 1));
      K_ = static_cast<int>(left_shape.GetDim(left_num_dims - 2));
    } else {
      M_ = static_cast<int>(has_1D_input ? 1 : left_shape.GetDim(left_num_dims - 2));
      K_ = static_cast<int>(left_shape.GetDim(left_num_dims - 1));
    }

    if (transb) {
      N_ = static_cast<int>((right_num_dims == 1) ? 1 : right_shape.GetDim(right_num_dims - 2));
    } else {
      N_ = static_cast<int>((right_num_dims == 1) ? 1 : right_shape.GetDim(right_num_dims - 1));
    }

    if (!has_1D_input) {
      output_dims[num_output_dims - 2] = M_;
      output_dims[num_output_dims - 1] = N_;
    } else {
      if (num_output_dims == 0) {
        // for left and right being both vector, output is scalar thus no shape
        // ORT_RETURN_IF_NOT(M_ == 1 && N_ == 1, "M_ == 1 && N_ == 1 was false");
      } else {
        if (left_num_dims == 1) {
        //   ORT_RETURN_IF_NOT(num_dims_with_pad - 1 == num_output_dims, "num_dims_with_pad - 1 != num_output_dims");
        //   ORT_RETURN_IF_NOT(K_ == right_shape[transb ? right_num_dims - 1 : right_num_dims - 2],
                            // "MatMul dimension mismatch");
          // left (K), right (...K,N), output (...N)
          output_dims[num_output_dims - 1] = N_;
        } else {
        //   ORT_RETURN_IF_NOT(num_dims_with_pad - 2 == num_output_dims, "num_dims_with_pad - 2 != num_output_dims");
        //   ORT_RETURN_IF_NOT(K_ == right_shape[0], "MatMul dimension mismatch");
          // left(...K), right (K), output (...), already assigned
        }
      }
    }

    // assign shape
    output_shape_.Reshape(output_dims);

    // compute broadcast offsets
    ComputeBroadcastOffsets(fill_offsets);

    return ppl::common::RC_SUCCESS;
  }

  // Move this piece of code to public so that we don't need to call this if we can use GemmStridedBatched.
  void FillOffsets() {
    // stride in mats and dims for broadcasting
    left_padded_strides_.resize(num_broadcasted_dims_);
    right_padded_strides_.resize(num_broadcasted_dims_);
    output_broadcast_strides_.resize(num_broadcasted_dims_);
    output_broadcast_dims_.resize(num_broadcasted_dims_);
    for (size_t i = num_broadcasted_dims_; i > 0; --i) {
      size_t idx = i - 1;
      output_broadcast_dims_[idx] = std::max(left_padded_dims_[idx], right_padded_dims_[idx]);
      output_broadcast_strides_[idx] = ((i == num_broadcasted_dims_) ? 1 : output_broadcast_strides_[idx + 1] * output_broadcast_dims_[idx + 1]);
      left_padded_strides_[idx] = ((i == num_broadcasted_dims_) ? 1 : left_padded_strides_[idx + 1] * left_padded_dims_[idx + 1]);
      right_padded_strides_[idx] = ((i == num_broadcasted_dims_) ? 1 : right_padded_strides_[idx + 1] * right_padded_dims_[idx + 1]);
    }

    size_t num_offsets = output_broadcast_dims_[0] * output_broadcast_strides_[0];
    left_offsets_.resize(num_offsets);
    right_offsets_.resize(num_offsets);
    output_offsets_.resize(num_offsets);

    RecursiveFill(0, 0, 0, 0);
  }

 private:
  void ComputeBroadcastOffsets(bool fill_offsets) {
    num_broadcasted_dims_ = left_padded_dims_.size() - 2;

    if (num_broadcasted_dims_ == 0) {
      left_offsets_ = {0};
      right_offsets_ = {0};
      output_offsets_ = {0};
      return;
    }

    left_mat_size_ = M_ * K_ / left_stride_factor_;
    right_mat_size_ = K_ * N_ / right_stride_factor_;
    output_mat_size_ = M_ * N_;
  }

  void RecursiveFill(size_t idx_dim, size_t idx_left, size_t idx_right, size_t idx_out) {
    if (idx_dim == num_broadcasted_dims_) {
      left_offsets_[idx_out] = idx_left * left_mat_size_;
      right_offsets_[idx_out] = idx_right * right_mat_size_;
      output_offsets_[idx_out] = idx_out * output_mat_size_;
    } else {
      auto left_dim = left_padded_dims_[idx_dim];
      auto right_dim = right_padded_dims_[idx_dim];
      auto output_dim = output_broadcast_dims_[idx_dim];
      for (int i = 0; i < output_dim; ++i) {
        RecursiveFill(idx_dim + 1,
                      idx_left + i * (left_dim == 1 ? 0 : left_padded_strides_[idx_dim]),
                      idx_right + i * (right_dim == 1 ? 0 : right_padded_strides_[idx_dim]),
                      idx_out + i * output_broadcast_strides_[idx_dim]);
      }
    }
  }

 private:
  size_t left_mat_size_ = 0;
  size_t right_mat_size_ = 0;
  size_t output_mat_size_ = 0;

  size_t num_broadcasted_dims_ = 0;

  std::vector<int64_t> left_padded_dims_;
  std::vector<int64_t> right_padded_dims_;
  std::vector<int64_t> output_broadcast_dims_;

  std::vector<size_t> left_padded_strides_;
  std::vector<size_t> right_padded_strides_;
  std::vector<size_t> output_broadcast_strides_;

  TensorShape output_shape_;

  int M_ = 0;
  int N_ = 0;
  int K_ = 0;

  std::vector<size_t> left_offsets_;
  std::vector<size_t> right_offsets_;
  std::vector<size_t> output_offsets_;

  std::vector<size_t> right_zp_offsets_;
  std::vector<size_t> right_scale_offsets_;

  size_t left_stride_factor_ = 1;
  size_t right_stride_factor_ = 1;
  int left_ld_factor_ = 1;
  int right_ld_factor_ = 1;

 public:
  // output shape
  const TensorShape& OutputShape() const {
    return output_shape_;
  }

  // left and output matrices' first dim
  int M() const {
    return M_;
  }

  // right and output matrices' second dim
  int N() const {
    return N_;
  }

  // left matrices' second dim, and right matrices' first dim
  int K() const {
    return K_;
  }

  int Lda(bool is_trans) const {
    return (is_trans ? static_cast<int>(M_) : static_cast<int>(K_)) * left_ld_factor_;
  }

  int Ldb(bool is_trans) const {
    return (is_trans ? static_cast<int>(K_) : static_cast<int>(N_)) * right_ld_factor_;
  }

  int Ldc() const {
    return static_cast<int>(N_);
  }

  // Batched Gemm offsets in left matrices
  const std::vector<size_t>& LeftOffsets() const {
    return left_offsets_;
  }

  // Batched Gemm offsets in right matrices
  const std::vector<size_t>& RightOffsets() const {
    return right_offsets_;
  }

  // Batched Gemm offsets in output matrices
  const std::vector<size_t>& OutputOffsets() const {
    return output_offsets_;
  }

  // Batched Scale Offset for right matrices
  const std::vector<size_t>& RightScaleOffsets() const {
    return right_scale_offsets_;
  }

  // Batched Zero Point Offset for right matrices
  const std::vector<size_t>& RightZeroPointOffsets() const {
    return right_zp_offsets_;
  }
};

}}} // namespace ppl::nn::cuda

#endif
