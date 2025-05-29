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

#ifndef _ST_HPC_PPL_COMMON_TENSOR_SHAPE_H_
#define _ST_HPC_PPL_COMMON_TENSOR_SHAPE_H_

#include "ppl/common/common.h"
#include "ppl/common/types.h"
#include <vector>

namespace ppl { namespace common {

static constexpr int64_t INVALID_DIM_VALUE = -1;

class PPLCOMMON_PUBLIC TensorShape final {
private:
    static const uint32_t kAxisC = 1;

private:
    bool is_scalar_;
    ppl::common::datatype_t data_type_ = ppl::common::DATATYPE_UNKNOWN;
    ppl::common::dataformat_t data_format_ = ppl::common::DATAFORMAT_UNKNOWN;
    ppl::common::dataformat_t data_format_force_to_ = ppl::common::DATAFORMAT_UNKNOWN;
    std::vector<int64_t> dims_;
    std::vector<int64_t> dims_nc1hwc0_;
    std::vector<int64_t> c_split_dims_;
    std::vector<uint16_t> padding0_;
    std::vector<uint16_t> padding1_;

private:
    static int16_t CalcPadding(int64_t dim, uint32_t alignment) {
        return static_cast<int16_t>((((uintptr_t)dim + (uintptr_t)alignment - 1) & ~((uintptr_t)alignment - 1)) -
                                    (uintptr_t)dim);
    }

public:
    TensorShape()
        : is_scalar_(false), data_type_(ppl::common::DATATYPE_UNKNOWN), data_format_(ppl::common::DATAFORMAT_UNKNOWN) {}

    TensorShape(const TensorShape& other) = default;
    TensorShape& operator=(const TensorShape& other) = default;
    bool operator==(const TensorShape& other) const{
        return (is_scalar_==other.is_scalar_ && data_type_==other.data_type_ && data_format_==other.data_format_
        && data_format_force_to_==other.data_format_force_to_&& dims_==other.dims_
         && dims_nc1hwc0_==other.dims_nc1hwc0_&& c_split_dims_==other.c_split_dims_
         && padding0_==other.padding0_&& padding1_==other.padding1_);
    }

    uint32_t GetRealDimCount(bool is_nc1hw0 = false) const {
        return is_nc1hw0 ? dims_nc1hwc0_.size() : dims_.size();
    };
    uint32_t GetDimCount(bool is_nc1hw0 = false) const {
        if (is_scalar_) {
            return 1;
        }
        return is_nc1hw0 ? dims_nc1hwc0_.size() : dims_.size();
    }
    int64_t GetDim(uint32_t which, bool is_nc1hw0 = false) const {
        if (is_scalar_) {
            return 1;
        }
        return is_nc1hw0 ? dims_nc1hwc0_[which] : dims_[which];
    }
    const int64_t* GetDims(bool is_nc1hw0 = false) const {
        return is_nc1hw0 ? dims_nc1hwc0_.data() : dims_.data();
    }
    uint16_t GetPadding0(uint32_t which) const {
        if (is_scalar_) {
            return 0;
        }
        return padding0_[which];
    }
    uint16_t GetPadding1(uint32_t which) const {
        if (is_scalar_) {
            return 0;
        }
        return padding1_[which];
    }
    const uint16_t* GetPadding0s() const {
        return padding0_.data();
    }
    const uint16_t* GetPadding1s() const {
        return padding1_.data();
    }

    ppl::common::datatype_t GetDataType() const {
        return data_type_;
    }
    ppl::common::dataformat_t GetDataFormat() const {
        return data_format_;
    }
    ppl::common::dataformat_t GetDataFormatForceTo() const {
        return data_format_force_to_;
    }
    void SetDataType(ppl::common::datatype_t dt) {
        data_type_ = dt;
    }
    void CalcPadding() {
        if (data_format_ == ppl::common::DATAFORMAT_NDARRAY || data_format_ == ppl::common::DATAFORMAT_NHWC) {
            for (uint32_t i = 0; i < dims_.size(); ++i) {
                padding0_[i] = 0;
                padding1_[i] = 0;
            }
        } else if (dims_.size() >= 2) {
            if (data_format_ == ppl::common::DATAFORMAT_N2CX) {
                padding1_[TensorShape::kAxisC] = CalcPadding(dims_[1], 2);
            } else if (data_format_ == ppl::common::DATAFORMAT_N4CX) {
                padding1_[TensorShape::kAxisC] = CalcPadding(dims_[1], 4);
            } else if (data_format_ == ppl::common::DATAFORMAT_N8CX) {
                padding1_[TensorShape::kAxisC] = CalcPadding(dims_[1], 8);
            } else if (data_format_ == ppl::common::DATAFORMAT_N16CX) {
                padding1_[TensorShape::kAxisC] = CalcPadding(dims_[1], 16);
            } else if (data_format_ == ppl::common::DATAFORMAT_N32CX) {
                padding1_[TensorShape::kAxisC] = CalcPadding(dims_[1], 32);
            } else if (data_format_ == ppl::common::DATAFORMAT_N16CX) {
                padding1_[TensorShape::kAxisC] = CalcPadding(dims_[1], 16);
            } else if (data_format_ == ppl::common::DATAFORMAT_NHWC8) {
                padding1_[TensorShape::kAxisC] = CalcPadding(dims_[1], 8);
            } else if (data_format_ == ppl::common::DATAFORMAT_NHWC16) {
                padding1_[TensorShape::kAxisC] = CalcPadding(dims_[1], 16);
            } else if (data_format_ == ppl::common::DATAFORMAT_NCHW8) {
                padding1_[TensorShape::kAxisC] = CalcPadding(dims_[1], 8);
            } else if (data_format_ == ppl::common::DATAFORMAT_NCHW16) {
                padding1_[TensorShape::kAxisC] = CalcPadding(dims_[1], 16);
            } else if (data_format_ == ppl::common::DATAFORMAT_NHWC4) {
                padding1_[TensorShape::kAxisC] = CalcPadding(dims_[1], 4);
            }
        }
    }

    void SetDataFormat(ppl::common::dataformat_t data_format) {
        // do not change padding if data format is not changed
        if (data_format != data_format_) {
            data_format_ = data_format;
            CalcPadding();
        }
    }

    void SetDataFormatForceTo(ppl::common::dataformat_t data_format_force_to) {
        if (data_format_force_to != data_format_
            && (data_format_force_to == ppl::common::DATAFORMAT_NHWC || data_format_force_to == ppl::common::DATAFORMAT_NDARRAY)
            && (data_format_ == ppl::common::DATAFORMAT_NHWC || data_format_ == ppl::common::DATAFORMAT_NDARRAY)) {
            data_format_force_to_ = data_format_force_to;
        }
    }

    void ReshapeAsScalar() {
        dims_.clear();
        padding0_.clear();
        padding1_.clear();
        is_scalar_ = true;
    }

    void Reshape(const int64_t* dims, uint32_t dim_count) {
        DoResize(dim_count);
        if (dim_count == 0) {
            is_scalar_ = true;
            return;
        }

        is_scalar_ = false;
        for (uint32_t i = 0; i < dim_count; ++i) {
            dims_[i] = dims[i];
        }
        CalcPadding();
        if(dim_count > 2 && dim_count < 5){
            dims_nc1hwc0_[0] = dims_[0];
            if(data_format_ == DATAFORMAT_NCHW8){
                dims_nc1hwc0_[1] = (dims_[1] + 7) / 8;
                dims_nc1hwc0_[2] = dim_count > 2 ? dims_[2] : 1;
                dims_nc1hwc0_[3] = dim_count > 3 ? dims_[3] : 1;
                dims_nc1hwc0_[4] = 8;
            }
            if(data_format_ == DATAFORMAT_NCHW16){
                dims_nc1hwc0_[1] = (dims_[1] + 15) / 16;
                dims_nc1hwc0_[2] = dim_count > 2 ? dims_[2] : 1;
                dims_nc1hwc0_[3] = dim_count > 3 ? dims_[3] : 1;
                dims_nc1hwc0_[4] = 16;
            }
        }
    }

    void Reshape(const std::vector<int64_t>& dims) {
        return Reshape(dims.data(), dims.size());
    }

    void SetChannelSpilt(const std::vector<int64_t>& c_split_dims) {
        c_split_dims_.clear();
        for(int i = 0; i < c_split_dims.size(); i++)
            c_split_dims_.push_back(c_split_dims[i]);
    }

    const int64_t* GetChannelSpilt(void) const {
        return c_split_dims_.data();
    }

    uint32_t GetChannelSpiltCount(void) const {
        return c_split_dims_.size();
    };

    void SetPadding0(uint32_t which, uint16_t padding) {
        padding0_[which] = padding;
    }
    void SetPadding1(uint32_t which, uint16_t padding) {
        padding1_[which] = padding;
    }
    void SetDim(uint32_t which, int64_t dim) {
        dims_[which] = dim;
        Reshape(dims_.data(), dims_.size());
    }
    void SetDimCount(uint32_t dc) {
        DoResize(dc);
        is_scalar_ = (dc == 0);
    }

    uint64_t CalcElementsIncludingPadding() const {
        if (dims_.empty()) {
            return is_scalar_ ? 1 : 0;
        }
        uint64_t accu = 1;
        for (uint32_t i = 0; i < dims_.size(); ++i) {
            accu *= (dims_[i] + padding0_[i] + padding1_[i]);
        }
        return accu;
    }
    uint64_t CalcElementsExcludingPadding() const {
        if (dims_.empty()) {
            return is_scalar_ ? 1 : 0;
        }
        uint64_t accu = 1;
        for (uint32_t i = 0; i < dims_.size(); ++i) {
            accu *= dims_[i];
        }
        return accu;
    }
    uint64_t CalcBytesIncludingPadding() const {
        return ppl::common::GetSizeOfDataType(data_type_) * CalcElementsIncludingPadding();
    }
    uint64_t CalcBytesExcludingPadding() const {
        return ppl::common::GetSizeOfDataType(data_type_) * CalcElementsExcludingPadding();
    }
    uint64_t CalcElementsFromDimensionIncludingPadding(uint32_t which) const {
        if (dims_.empty()) {
            return is_scalar_ ? 1 : 0;
        }
        uint64_t accu = 1;
        for (uint32_t i = which; i < dims_.size(); ++i) {
            accu *= (dims_[i] + padding0_[i] + padding1_[i]);
        }
        return accu;
    }
    uint64_t CalcElementsToDimensionIncludingPadding(uint32_t which) const {
        if (dims_.empty()) {
            return is_scalar_ ? 1 : 0;
        }
        uint64_t accu = 1;
        for (uint32_t i = 0; i < which; ++i) {
            accu *= (dims_[i] + padding0_[i] + padding1_[i]);
        }
        return accu;
    }
    uint64_t CalcElementsFromDimensionExcludingPadding(uint32_t which) const {
        if (dims_.empty()) {
            return is_scalar_ ? 1 : 0;
        }
        uint64_t accu = 1;
        for (uint32_t i = which; i < dims_.size(); ++i) {
            accu *= dims_[i];
        }
        return accu;
    }
    uint64_t CalcElementsToDimensionExcludingPadding(uint32_t which) const {
        if (dims_.empty()) {
            return is_scalar_ ? 1 : 0;
        }
        uint64_t accu = 1;
        for (uint32_t i = 0; i < which; ++i) {
            accu *= dims_[i];
        }
        return accu;
    }
    uint64_t CalcBytesToDimesionIncludingPadding(uint32_t which) const {
        return ppl::common::GetSizeOfDataType(data_type_) * CalcElementsToDimensionIncludingPadding(which);
    }
    uint64_t CalcBytesToDimesionExcludingPadding(uint32_t which) const {
        return ppl::common::GetSizeOfDataType(data_type_) * CalcElementsToDimensionExcludingPadding(which);
    }
    uint64_t CalcBytesFromDimesionIncludingPadding(uint32_t which) const {
        return ppl::common::GetSizeOfDataType(data_type_) * CalcElementsFromDimensionIncludingPadding(which);
    }
    uint64_t CalcBytesFromDimesionExcludingPadding(uint32_t which) const {
        return ppl::common::GetSizeOfDataType(data_type_) * CalcElementsFromDimensionExcludingPadding(which);
    }

    void Clear() {
        is_scalar_ = false;
        data_type_ = ppl::common::DATATYPE_UNKNOWN;
        data_format_ = ppl::common::DATAFORMAT_UNKNOWN;
        data_format_force_to_ = ppl::common::DATAFORMAT_UNKNOWN;
        dims_.clear();
        dims_nc1hwc0_.resize(5, 0);
        c_split_dims_.clear();
        padding0_.clear();
        padding1_.clear();
    }

    bool IsScalar() const {
        return is_scalar_;
    }

    bool IsEmpty() const {
        if (is_scalar_) {
            return false;
        }
        if (dims_.empty()) {
            return true;
        }

        uint64_t accu = 1;
        for (uint32_t i = 0; i < dims_.size(); ++i) {
            accu *= dims_[i];
        }
        return (accu == 0);
    }

private:
    void DoResize(uint32_t dim_count) {
        dims_.resize(dim_count, 0);
        dims_nc1hwc0_.resize(5, 0);
        padding0_.resize(dim_count, 0);
        padding1_.resize(dim_count, 0);
    }
};

}} // namespace ppl::common

#endif
