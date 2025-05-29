// Copyright 2024 metax-tech.com Inc. All Rights Reserved.
#pragma once
#include "onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

ORT_EXPORT OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api);

#ifdef __cplusplus
}
#endif
