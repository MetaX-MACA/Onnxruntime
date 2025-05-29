// 2024 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/exceptions.h"

// Setup the infrastructure to throw ORT exceptions so they're caught by existing handlers.

template <typename E>
class SafeIntExceptionHandler;

template <>
class SafeIntExceptionHandler<onnxruntime::OnnxRuntimeException> {
 public:
  static void SafeIntOnOverflow() {
    ORT_THROW("Integer overflow");
  }

  static void SafeIntOnDivZero() {
    ORT_THROW("Divide by zero");
  }
};

#define SAFEINT_EXCEPTION_HANDLER_CPP 1
#define SafeIntDefaultExceptionHandler SafeIntExceptionHandler<onnxruntime::OnnxRuntimeException>

#if defined(__GNUC__)
#include "onnxruntime_config.h"
#pragma GCC diagnostic push
#ifdef HAS_UNUSED_BUT_SET_PARAMETER
  #ifndef USE_MACA
    #pragma GCC diagnostic ignored "-Wunused-but-set-parameter"
  #endif
#endif
#endif
#include "safeint/SafeInt.hpp"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
