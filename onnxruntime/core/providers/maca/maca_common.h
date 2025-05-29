// Copyright 2023 metax-tech.com Inc.

#pragma once

#include "core/common/status.h"
#include "maca_call.h"
#include "gsl/gsl"

namespace onnxruntime {
namespace maca {

#define CUDA_RETURN_IF_ERROR(expr)               \
  ORT_RETURN_IF_ERROR(CUDA_CALL(expr)            \
                          ? common::Status::OK() \
                          : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "CUDA error executing !!", #expr))

}  // namespace maca
}  // namespace onnxruntime
