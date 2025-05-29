// 2024 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#if defined(_MSC_VER)
#pragma warning(push)
//cuda_fp16.hpp(394,38): warning C4505: '__float2half_rz': unreferenced local function has been removed
#pragma warning(disable : 4505)
#endif

#ifdef USE_MACA
#include "cuda_runtime_wrapper.h"
//#define CUDNN_BN_MIN_EPSILON  0.0
//#define CUDNN_RNN_PADDED_IO_ENABLED (1U << 0)
#endif


#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <curand.h>
#include <cudnn.h>
#include <cufft.h>

#ifdef ORT_USE_NCCL
#include <nccl.h>
#endif

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
