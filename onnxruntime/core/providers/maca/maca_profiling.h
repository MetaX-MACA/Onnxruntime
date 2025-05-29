// Copyright 2024 metax-tech.com Inc. All Rights Reserved.

#include <string>
#include <string.h>
#include <chrono>
#include <string>
#include <memory>
#include <random>
#include <map>
#include <fstream>
#include <sstream>
#include <iostream>
#include <functional>
#include <algorithm>

#include "ppl/nn/runtime/options.h"
#include "ppl/nn/runtime/runtime.h"

#include "core/common/logging/logging.h"

namespace onnxruntime{
bool macaProfiling(std::shared_ptr<ppl::nn::Runtime> runtime, double* run_dur, uint32_t* run_count);
}
