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

#include "cudakernel/arithmetic/arithmetic.h"
#include "cudakernel/common/divmod_fast.h"
#include "ppl/common/types.h"
#include <cuda_fp16.h>
#include <algorithm>
#define MAX_BLOCK_DIM_YZ (65536)
enum ArithmeticOpType {
    Arithmetic_Unknown = 0,
    Arithmetic_Add,
    Arithmetic_Sub,
    Arithmetic_Mul,
    Arithmetic_Div,
    Arithmetic_Max,
    Arithmetic_Min,
    Arithmetic_Pow,
    Arithmetic_PRelu, // similar to arithmetic
    Arithmetic_Mod,
    Arithmetic_FMod,
    Arithmetic_OpNum,
    Arithmetic_ForceWord = INT_MAX,
};
#ifdef __MACACC__
#define OPT_ARITHM
#endif//__MACACC__

struct half8_ {
    half x0;
    half y0;
    half z0;
    half w0;
    half x1;
    half y1;
    half z1;
    half w1;
};

template<ArithmeticOpType op_type, typename T>
__device__ inline T ppl_arithmetic_scalar(T a, T b);

template<> __device__ inline float ppl_arithmetic_scalar<Arithmetic_Add, float>(float a, float b) {
    return a + b;
}
template<> __device__ inline float ppl_arithmetic_scalar<Arithmetic_Sub, float>(float a, float b) {
    return a - b;
}
template<> __device__ inline float ppl_arithmetic_scalar<Arithmetic_Mul, float>(float a, float b) {
    return a * b;
}
template<> __device__ inline float ppl_arithmetic_scalar<Arithmetic_Div, float>(float a, float b) {
#ifdef OPT_ARITHM
    return a * __builtin_mxc_rcpf(b);
#else//!OPT_ARITHM
    return a / b;
#endif//OPT_ARITHM
}

template<> __device__ inline float ppl_arithmetic_scalar<Arithmetic_Max, float>(float a, float b) {
    return (a > b) ? a : b;
}
template<> __device__ inline float ppl_arithmetic_scalar<Arithmetic_Min, float>(float a, float b) {
    return (a > b) ? b : a;
}
template<> __device__ inline float ppl_arithmetic_scalar<Arithmetic_Pow, float>(float a, float b) {
    return powf(a ,b);
}

template<> __device__ inline float ppl_arithmetic_scalar<Arithmetic_PRelu, float>(float a, float b) {
    float res = a;
    res = (a > 0) ? res : res * b;
    return res;
}
template<> __device__ inline float ppl_arithmetic_scalar<Arithmetic_Mod, float>(float a, float b) {
    int64_t r = (int64_t)(a/b);
    return a - r * b;
}

template<> __device__ inline float ppl_arithmetic_scalar<Arithmetic_FMod, float>(float a, float b) {
    return fmod(a, b);
}

template<> __device__ inline double ppl_arithmetic_scalar<Arithmetic_Add, double>(double a, double b) {
    return a + b;
}
template<> __device__ inline double ppl_arithmetic_scalar<Arithmetic_Sub, double>(double a, double b) {
    return a - b;
}
template<> __device__ inline double ppl_arithmetic_scalar<Arithmetic_Mul, double>(double a, double b) {
    return a * b;
}
template<> __device__ inline double ppl_arithmetic_scalar<Arithmetic_Div, double>(double a, double b) {
    return a / b;
}
template<> __device__ inline double ppl_arithmetic_scalar<Arithmetic_Max, double>(double a, double b) {
    return (a > b) ? a : b;
}
template<> __device__ inline double ppl_arithmetic_scalar<Arithmetic_Min, double>(double a, double b) {
    return (a > b) ? b : a;
}
template<> __device__ inline double ppl_arithmetic_scalar<Arithmetic_Pow, double>(double a, double b) {
    return powf(a ,b);
}

template<> __device__ inline double ppl_arithmetic_scalar<Arithmetic_PRelu, double>(double a, double b) {
    float res = a;
    res = (a > 0) ? res : res * b;
    return res;
}
template<> __device__ inline double ppl_arithmetic_scalar<Arithmetic_Mod, double>(double a, double b) {
    int64_t r = (int64_t)(a/b);
    return a - r * b;
}

template<> __device__ inline double ppl_arithmetic_scalar<Arithmetic_FMod, double>(double a, double b) {
    return fmod(a, b);
}

template<> __device__ inline int64_t ppl_arithmetic_scalar<Arithmetic_Add, int64_t>(int64_t a, int64_t b) {
    return a + b;
}
template<> __device__ inline int64_t ppl_arithmetic_scalar<Arithmetic_Sub, int64_t>(int64_t a, int64_t b) {
    return a - b;
}
template<> __device__ inline int64_t ppl_arithmetic_scalar<Arithmetic_Mul, int64_t>(int64_t a, int64_t b) {
    return a * b;
}
template<> __device__ inline int64_t ppl_arithmetic_scalar<Arithmetic_Div, int64_t>(int64_t a, int64_t b) {
    return a / b;
}
template<> __device__ inline int64_t ppl_arithmetic_scalar<Arithmetic_Max, int64_t>(int64_t a, int64_t b) {
    return (a > b) ? a : b;
}
template<> __device__ inline int64_t ppl_arithmetic_scalar<Arithmetic_Min, int64_t>(int64_t a, int64_t b) {
    return (a > b) ? b : a;
}
template<> __device__ inline int64_t ppl_arithmetic_scalar<Arithmetic_Pow, int64_t>(int64_t a, int64_t b) {
    return powf(a ,b);
}
template<> __device__ inline int64_t ppl_arithmetic_scalar<Arithmetic_PRelu, int64_t>(int64_t a, int64_t b) {
    int64_t res = a;
    res = (a > 0) ? res : res * b;
    return res;
}
template<> __device__ inline int64_t ppl_arithmetic_scalar<Arithmetic_Mod, int64_t>(int64_t a, int64_t b) {
    int64_t temp_value = a % b;
    return temp_value == 0 ? temp_value: ((b < 0 || temp_value < 0) ? (temp_value + b) :  temp_value);
}

template<> __device__ inline int64_t ppl_arithmetic_scalar<Arithmetic_FMod, int64_t>(int64_t a, int64_t b) {
    int64_t temp_value = a / b;
    return a - temp_value * b;
}

template<> __device__ inline int32_t ppl_arithmetic_scalar<Arithmetic_Add, int32_t>(int32_t a, int32_t b) {
    return a + b;
}
template<> __device__ inline int32_t ppl_arithmetic_scalar<Arithmetic_Sub, int32_t>(int32_t a, int32_t b) {
    return a - b;
}
template<> __device__ inline int32_t ppl_arithmetic_scalar<Arithmetic_Mul, int32_t>(int32_t a, int32_t b) {
    return a * b;
}
template<> __device__ inline int32_t ppl_arithmetic_scalar<Arithmetic_Div, int32_t>(int32_t a, int32_t b) {
    return a / b;
}
template<> __device__ inline int32_t ppl_arithmetic_scalar<Arithmetic_Max, int32_t>(int32_t a, int32_t b) {
    return (a > b) ? a : b;
}
template<> __device__ inline int32_t ppl_arithmetic_scalar<Arithmetic_Min, int32_t>(int32_t a, int32_t b) {
    return (a > b) ? b : a;
}
template<> __device__ inline int32_t ppl_arithmetic_scalar<Arithmetic_Pow, int32_t>(int32_t a, int32_t b) {
    return powf(a ,b);
}

template<> __device__ inline int32_t ppl_arithmetic_scalar<Arithmetic_PRelu, int32_t>(int32_t a, int32_t b) {
    int32_t res = a;
    res = (a > 0) ? res : res * b;
    return res;
}
template<> __device__ inline int32_t ppl_arithmetic_scalar<Arithmetic_Mod, int32_t>(int32_t a, int32_t b) {
    int32_t temp_value = a % b;
    return temp_value == 0 ? temp_value: ((b < 0 || temp_value < 0) ? (temp_value + b) :  temp_value);
}

template<> __device__ inline int32_t ppl_arithmetic_scalar<Arithmetic_FMod, int32_t>(int32_t a, int32_t b) {
    int32_t temp_value = a / b;
    return a - temp_value * b;
}

template<> __device__ inline int16_t ppl_arithmetic_scalar<Arithmetic_Add, int16_t>(int16_t a, int16_t b) {
    return a + b;
}
template<> __device__ inline int16_t ppl_arithmetic_scalar<Arithmetic_Sub, int16_t>(int16_t a, int16_t b) {
    return a - b;
}
template<> __device__ inline int16_t ppl_arithmetic_scalar<Arithmetic_Mul, int16_t>(int16_t a, int16_t b) {
    return a * b;
}
template<> __device__ inline int16_t ppl_arithmetic_scalar<Arithmetic_Div, int16_t>(int16_t a, int16_t b) {
    return a / b;
}
template<> __device__ inline int16_t ppl_arithmetic_scalar<Arithmetic_Max, int16_t>(int16_t a, int16_t b) {
    return (a > b) ? a : b;
}
template<> __device__ inline int16_t ppl_arithmetic_scalar<Arithmetic_Min, int16_t>(int16_t a, int16_t b) {
    return (a > b) ? b : a;
}
template<> __device__ inline int16_t ppl_arithmetic_scalar<Arithmetic_Pow, int16_t>(int16_t a, int16_t b) {
    return powf(a ,b);
}

template<> __device__ inline int16_t ppl_arithmetic_scalar<Arithmetic_PRelu, int16_t>(int16_t a, int16_t b) {
    int32_t res = a;
    res = (a > 0) ? res : res * b;
    return res;
}
template<> __device__ inline int16_t ppl_arithmetic_scalar<Arithmetic_Mod, int16_t>(int16_t a, int16_t b) {
    int16_t temp_value = a % b;
    return temp_value == 0 ? temp_value: ((b < 0 || temp_value < 0) ? (temp_value + b) :  temp_value);
}

template<> __device__ inline int16_t ppl_arithmetic_scalar<Arithmetic_FMod, int16_t>(int16_t a, int16_t b) {
    int16_t temp_value = a / b;
    return a - temp_value * b;
}

#ifdef OPT_ARITHM
template<ArithmeticOpType op_type, typename T>
__device__ inline T ppl_arithmetic_scalar_int8_no_scale(T a, T b);

template<> __device__ inline int8_t ppl_arithmetic_scalar_int8_no_scale<Arithmetic_Add, int8_t>(int8_t a, int8_t b) {
    int res = a + b;
    if(res > 127) res = 127;
    else if(res < -128) res = -128;
    return res;
}
template<> __device__ inline int8_t ppl_arithmetic_scalar_int8_no_scale<Arithmetic_Sub, int8_t>(int8_t a, int8_t b) {
    int res = a - b;
    if(res > 127) res = 127;
    else if(res < -128) res = -128;
    return res;
}
template<> __device__ inline int8_t ppl_arithmetic_scalar_int8_no_scale<Arithmetic_Mul, int8_t>(int8_t a, int8_t b) {
    int res = a * b;
    if(res > 127) res = 127;
    else if(res < -128) res = -128;
    return res;
}
template<> __device__ inline int8_t ppl_arithmetic_scalar_int8_no_scale<Arithmetic_Div, int8_t>(int8_t a, int8_t b) {
    int res = a / b;
    if(res > 127) res = 127;
    else if(res < -128) res = -128;
    return res;
}
template<> __device__ inline int8_t ppl_arithmetic_scalar_int8_no_scale<Arithmetic_Max, int8_t>(int8_t a, int8_t b) {
    return (a > b) ? a : b;
}
template<> __device__ inline int8_t ppl_arithmetic_scalar_int8_no_scale<Arithmetic_Min, int8_t>(int8_t a, int8_t b) {
    return (a > b) ? b : a;
}
template<> __device__ inline int8_t ppl_arithmetic_scalar_int8_no_scale<Arithmetic_Pow, int8_t>(int8_t a, int8_t b) {
    int res = powf(a, b);
    if(res > 127) res = 127;
    else if(res < -128) res = -128;
    return res;
}
template<> __device__ inline int8_t ppl_arithmetic_scalar_int8_no_scale<Arithmetic_PRelu, int8_t>(int8_t a, int8_t b) {
    int res = a;
    res = (a > 0) ? res : res * b;
    if(res > 127) res = 127;
    else if(res < -128) res = -128;
    return res;
}
template<> __device__ inline int8_t ppl_arithmetic_scalar_int8_no_scale<Arithmetic_Mod, int8_t>(int8_t a, int8_t b) {
    int8_t temp_value = a % b;
    return temp_value == 0 ? temp_value: ((b < 0 || temp_value < 0) ? (temp_value + b) :  temp_value);
}
template<> __device__ inline int8_t ppl_arithmetic_scalar_int8_no_scale<Arithmetic_FMod, int8_t>(int8_t a, int8_t b) {
    int8_t temp_value = a / b;
    return a - temp_value * b;
}
#endif // OPT_ARITHM

template<ArithmeticOpType op_type, typename T>
__device__ inline T ppl_arithmetic_scalar_int8(T a, T b, float in_scale0, float in_scale1, float out_scale);

#ifdef OPT_ARITHM
template<> __device__ inline int8_t ppl_arithmetic_scalar_int8<Arithmetic_Add, int8_t>(int8_t a, int8_t b, float in_scale0, float in_scale1, float out_scale) {
    //int res = round((a * in_scale0 + b * in_scale1) / out_scale);
    //arithmetic optimization: in_scale0 = in_scale0_ori/out_scale; in_scale1 = in_scale1_ori/out_scale
    int res = round(a*in_scale0+b*in_scale1);
    if(res > 127) res = 127;
    else if(res < -128) res = -128;
    return res;
}
template<> __device__ inline int8_t ppl_arithmetic_scalar_int8<Arithmetic_Sub, int8_t>(int8_t a, int8_t b, float in_scale0, float in_scale1, float out_scale) {
    //int res = round((a * in_scale0 - b * in_scale1) / out_scale);
    //arithmetic optimization: in_scale0 = in_scale0_ori/out_scale; in_scale1 = in_scale1_ori/out_scale
    int res = round(a*in_scale0-b*in_scale1);
    if(res > 127) res = 127;
    else if(res < -128) res = -128;
    return res;
}
template<> __device__ inline int8_t ppl_arithmetic_scalar_int8<Arithmetic_Mul, int8_t>(int8_t a, int8_t b, float in_scale0, float in_scale1, float out_scale) {
    //int res = round(a * b * in_scale0 * in_scale1 / out_scale);
    //arithmetic optimization: in_scale0=in_scale0_ori*in_scale1/out_scale
    int res = round(a*b*in_scale0);
    if(res > 127) res = 127;
    else if(res < -128) res = -128;
    return res;
}
template<> __device__ inline int8_t ppl_arithmetic_scalar_int8<Arithmetic_Div, int8_t>(int8_t a, int8_t b, float in_scale0, float in_scale1, float out_scale) {
    //int res = round((float(a) / b * in_scale0 * in_scale1) * out_scale);
    //arithmetic optimization: in_scale0=in_scale0_ori*in_scale1/out_scale
    int res = round(float(a) / b * in_scale0);
    if(res > 127) res = 127;
    else if(res < -128) res = -128;
    return res;
}
#else
template<> __device__ inline int8_t ppl_arithmetic_scalar_int8<Arithmetic_Add, int8_t>(int8_t a, int8_t b, float in_scale0, float in_scale1, float out_scale) {
    int res = round((a * in_scale0 + b * in_scale1) / out_scale);
    if(res > 127) res = 127;
    else if(res < -128) res = -128;
    return res;
}

template<> __device__ inline int8_t ppl_arithmetic_scalar_int8<Arithmetic_Sub, int8_t>(int8_t a, int8_t b, float in_scale0, float in_scale1, float out_scale) {
    int res = round((a * in_scale0 - b * in_scale1) / out_scale);
    if(res > 127) res = 127;
    else if(res < -128) res = -128;
    return res;
}
template<> __device__ inline int8_t ppl_arithmetic_scalar_int8<Arithmetic_Mul, int8_t>(int8_t a, int8_t b, float in_scale0, float in_scale1, float out_scale) {
    int res = round(a * b * in_scale0 * in_scale1 / out_scale);
    if(res > 127) res = 127;
    else if(res < -128) res = -128;
    return res;
}
template<> __device__ inline int8_t ppl_arithmetic_scalar_int8<Arithmetic_Div, int8_t>(int8_t a, int8_t b, float in_scale0, float in_scale1, float out_scale) {
    int res = round((float(a) / b * in_scale0 / in_scale1) / out_scale);
    if(res > 127) res = 127;
    else if(res < -128) res = -128;
    return res;
}
#endif
template<> __device__ inline int8_t ppl_arithmetic_scalar_int8<Arithmetic_Max, int8_t>(int8_t a, int8_t b, float in_scale0, float in_scale1, float out_scale) {
    return (a > b) ? a : b;
}
template<> __device__ inline int8_t ppl_arithmetic_scalar_int8<Arithmetic_Min, int8_t>(int8_t a, int8_t b, float in_scale0, float in_scale1, float out_scale) {
    return (a > b) ? b : a;
}
template<> __device__ inline int8_t ppl_arithmetic_scalar_int8<Arithmetic_Pow, int8_t>(int8_t a, int8_t b, float in_scale0, float in_scale1, float out_scale) {
    int res = powf(a, b);
    if(res > 127) res = 127;
    else if(res < -128) res = -128;
    return res;
}

#ifdef OPT_ARITHM
template<> __device__ inline int8_t ppl_arithmetic_scalar_int8<Arithmetic_PRelu, int8_t>(int8_t a, int8_t b, float in_scale0, float in_scale1, float out_scale) {
    int res = a;
    res = (a > 0) ? res : res * b * in_scale1;
    res = round(res * in_scale0 / out_scale);
    if(res > 127) res = 127;
    else if(res < -128) res = -128;
    return res;
}
#else
template<> __device__ inline int8_t ppl_arithmetic_scalar_int8<Arithmetic_PRelu, int8_t>(int8_t a, int8_t b, float in_scale0, float in_scale1, float out_scale) {
    int res = a;
    res = (a > 0) ? res : res * b;
    res = round(res * in_scale0 / out_scale);
    if(res > 127) res = 127;
    else if(res < -128) res = -128;
    return res;
}
#endif
template<> __device__ inline int8_t ppl_arithmetic_scalar_int8<Arithmetic_Mod, int8_t>(int8_t a, int8_t b, float in_scale0, float in_scale1, float out_scale) {
    int8_t temp_value = a % b;
    return temp_value == 0 ? temp_value: ((b < 0 || temp_value < 0) ? (temp_value + b) :  temp_value);
}
template<> __device__ inline int8_t ppl_arithmetic_scalar_int8<Arithmetic_FMod, int8_t>(int8_t a, int8_t b, float in_scale0, float in_scale1, float out_scale) {
    int8_t temp_value = a / b;
    return a - temp_value * b;
}

template<> __device__ inline uint64_t ppl_arithmetic_scalar<Arithmetic_Add, uint64_t>(uint64_t a, uint64_t b) {
    return a + b;
}
template<> __device__ inline uint64_t ppl_arithmetic_scalar<Arithmetic_Sub, uint64_t>(uint64_t a, uint64_t b) {
    return a - b;
}
template<> __device__ inline uint64_t ppl_arithmetic_scalar<Arithmetic_Mul, uint64_t>(uint64_t a, uint64_t b) {
    return a * b;
}
template<> __device__ inline uint64_t ppl_arithmetic_scalar<Arithmetic_Div, uint64_t>(uint64_t a, uint64_t b) {
    return a / b;
}
template<> __device__ inline uint64_t ppl_arithmetic_scalar<Arithmetic_Max, uint64_t>(uint64_t a, uint64_t b) {
    return (a > b) ? a : b;
}
template<> __device__ inline uint64_t ppl_arithmetic_scalar<Arithmetic_Min, uint64_t>(uint64_t a, uint64_t b) {
    return (a > b) ? b : a;
}
template<> __device__ inline uint64_t ppl_arithmetic_scalar<Arithmetic_Pow, uint64_t>(uint64_t a, uint64_t b) {
    return powf(a ,b);
}
template<> __device__ inline uint64_t ppl_arithmetic_scalar<Arithmetic_PRelu, uint64_t>(uint64_t a, uint64_t b) {
    uint64_t res = a;
    res = (a > 0) ? res : res * b;
    return res;
}
template<> __device__ inline uint64_t ppl_arithmetic_scalar<Arithmetic_Mod, uint64_t>(uint64_t a, uint64_t b) {
    return a % b ;
}
template<> __device__ inline uint64_t ppl_arithmetic_scalar<Arithmetic_FMod, uint64_t>(uint64_t a, uint64_t b) {
    uint64_t temp_value = a / b;
    return a - temp_value * b;
}
template<> __device__ inline uint32_t ppl_arithmetic_scalar<Arithmetic_Add, uint32_t>(uint32_t a, uint32_t b) {
    return a + b;
}
template<> __device__ inline uint32_t ppl_arithmetic_scalar<Arithmetic_Sub, uint32_t>(uint32_t a, uint32_t b) {
    return a - b;
}
template<> __device__ inline uint32_t ppl_arithmetic_scalar<Arithmetic_Mul, uint32_t>(uint32_t a, uint32_t b) {
    return a * b;
}
template<> __device__ inline uint32_t ppl_arithmetic_scalar<Arithmetic_Div, uint32_t>(uint32_t a, uint32_t b) {
    return a / b;
}
template<> __device__ inline uint32_t ppl_arithmetic_scalar<Arithmetic_Max, uint32_t>(uint32_t a, uint32_t b) {
    return (a > b) ? a : b;
}
template<> __device__ inline uint32_t ppl_arithmetic_scalar<Arithmetic_Min, uint32_t>(uint32_t a, uint32_t b) {
    return (a > b) ? b : a;
}
template<> __device__ inline uint32_t ppl_arithmetic_scalar<Arithmetic_Pow, uint32_t>(uint32_t a, uint32_t b) {
    return powf(a ,b);
}

template<> __device__ inline uint32_t ppl_arithmetic_scalar<Arithmetic_PRelu, uint32_t>(uint32_t a, uint32_t b) {
    uint32_t res = a;
    res = (a > 0) ? res : res * b;
    return res;
}
template<> __device__ inline uint32_t ppl_arithmetic_scalar<Arithmetic_Mod, uint32_t>(uint32_t a, uint32_t b) {
    return a % b;
}
template<> __device__ inline uint32_t ppl_arithmetic_scalar<Arithmetic_FMod, uint32_t>(uint32_t a, uint32_t b) {
    uint32_t temp_value = a / b;
    return a - temp_value * b;
}
template<> __device__ inline uint16_t ppl_arithmetic_scalar<Arithmetic_Add, uint16_t>(uint16_t a, uint16_t b) {
    return a + b;
}
template<> __device__ inline uint16_t ppl_arithmetic_scalar<Arithmetic_Sub, uint16_t>(uint16_t a, uint16_t b) {
    return a - b;
}
template<> __device__ inline uint16_t ppl_arithmetic_scalar<Arithmetic_Mul, uint16_t>(uint16_t a, uint16_t b) {
    return a * b;
}
template<> __device__ inline uint16_t ppl_arithmetic_scalar<Arithmetic_Div, uint16_t>(uint16_t a, uint16_t b) {
    return a / b;
}
template<> __device__ inline uint16_t ppl_arithmetic_scalar<Arithmetic_Max, uint16_t>(uint16_t a, uint16_t b) {
    return (a > b) ? a : b;
}
template<> __device__ inline uint16_t ppl_arithmetic_scalar<Arithmetic_Min, uint16_t>(uint16_t a, uint16_t b) {
    return (a > b) ? b : a;
}
template<> __device__ inline uint16_t ppl_arithmetic_scalar<Arithmetic_Pow, uint16_t>(uint16_t a, uint16_t b) {
    return powf(a ,b);
}

template<> __device__ inline uint16_t ppl_arithmetic_scalar<Arithmetic_PRelu, uint16_t>(uint16_t a, uint16_t b) {
    uint32_t res = a;
    res = (a > 0) ? res : res * b;
    return res;
}
template<> __device__ inline uint16_t ppl_arithmetic_scalar<Arithmetic_Mod, uint16_t>(uint16_t a, uint16_t b) {
    return a % b;
}
template<> __device__ inline uint16_t ppl_arithmetic_scalar<Arithmetic_FMod, uint16_t>(uint16_t a, uint16_t b) {
    uint16_t temp_value = a / b;
    return a - temp_value * b;
}

template<> __device__ inline uint8_t ppl_arithmetic_scalar<Arithmetic_Add, uint8_t>(uint8_t a, uint8_t b) {
    int res = a + b;
    if(res > 255) res = 255;
    else if(res < 0) res = 0;
    return res;
}
template<> __device__ inline uint8_t ppl_arithmetic_scalar<Arithmetic_Sub, uint8_t>(uint8_t a, uint8_t b) {
    int res = a - b;
    if(res > 255) res = 255;
    else if(res < 0) res = 0;
    return res;
}
template<> __device__ inline uint8_t ppl_arithmetic_scalar<Arithmetic_Mul, uint8_t>(uint8_t a, uint8_t b) {
    int res = a * b;
    if(res > 255) res = 255;
    else if(res < 0) res = 0;
    return res;
}
template<> __device__ inline uint8_t ppl_arithmetic_scalar<Arithmetic_Div, uint8_t>(uint8_t a, uint8_t b) {
    int res = a / b;
    if(res > 255) res = 255;
    else if(res < 0) res = 0;
    return res;
}
template<> __device__ inline uint8_t ppl_arithmetic_scalar<Arithmetic_Max, uint8_t>(uint8_t a, uint8_t b) {
    return (a > b) ? a : b;
}
template<> __device__ inline uint8_t ppl_arithmetic_scalar<Arithmetic_Min, uint8_t>(uint8_t a, uint8_t b) {
    return (a > b) ? b : a;
}
template<> __device__ inline uint8_t ppl_arithmetic_scalar<Arithmetic_Pow, uint8_t>(uint8_t a, uint8_t b) {
    int res = powf(a ,b);
    if(res > 255) res = 255;
    else if(res < 0) res = 0;
    return res;
}

template<> __device__ inline uint8_t ppl_arithmetic_scalar<Arithmetic_PRelu, uint8_t>(uint8_t a, uint8_t b) {
    uint32_t res = a;
    res = (a > 0) ? res : res * b;
    if(res > 255) res = 255;
    else if(res < 0) res = 0;
    return res;
}
template<> __device__ inline uint8_t ppl_arithmetic_scalar<Arithmetic_Mod, uint8_t>(uint8_t a, uint8_t b) {
    int res = a % b;
    if(res > 255) res = 255;
    else if(res < 0) res = 0;
    return res;
}
template<> __device__ inline uint8_t ppl_arithmetic_scalar<Arithmetic_FMod, uint8_t>(uint8_t a, uint8_t b) {
    uint8_t temp_value = a / b;
    int res = a - temp_value * b;
    if(res > 255) res = 255;
    else if(res < 0) res = 0;
    return res;
}

template <>
__device__ inline half ppl_arithmetic_scalar<Arithmetic_Add, half>(half a, half b)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    return __hadd(a, b);
#else
    return 0;
#endif
}
template <>
__device__ inline half ppl_arithmetic_scalar<Arithmetic_Sub, half>(half a, half b)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    return __hsub(a, b);
#else
    return 0;
#endif
}
template <>
__device__ inline half ppl_arithmetic_scalar<Arithmetic_Mul, half>(half a, half b)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    return __hmul(a, b);
#else
    return 0;
#endif
}
template <>
__device__ inline half ppl_arithmetic_scalar<Arithmetic_Div, half>(half a, half b)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    return __hdiv(a, b);
#else
    return 0;
#endif
}
template <>
__device__ inline half ppl_arithmetic_scalar<Arithmetic_Max, half>(half a, half b)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    return __hgt(a, b) ? a : b;
#else
    return 0;
#endif
}
template <>
__device__ inline half ppl_arithmetic_scalar<Arithmetic_Min, half>(half a, half b)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    return __hgt(a, b) ? b : a;
#else
    return 0;
#endif
}
template <>
__device__ inline half ppl_arithmetic_scalar<Arithmetic_Pow, half>(half a, half b)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    return __float2half(powf(__half2float(a), __half2float(b)));
#else
    return 0;
#endif
}

template <>
__device__ inline half ppl_arithmetic_scalar<Arithmetic_PRelu, half>(half a, half b)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    half res = a;
    res      = __hgt(a, 0) ? res : __hmul(res, b);
    return res;
#else
    return 0;
#endif
}
template<>
__device__ inline half ppl_arithmetic_scalar<Arithmetic_Mod, half>(half a, half b)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9

    int r = __half2int_rz(__hdiv(a, b));
    return __hsub(a, __hmul(__int2half_rz(r), b));
#else
    return 0;
#endif
}

template<>
__device__ inline half ppl_arithmetic_scalar<Arithmetic_FMod, half>(half a, half b)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9

    int r = __half2int_rz(__hdiv(a, b));
    return __hsub(a, __hmul(__half2int_rz(r), b));
#else
    return 0;
#endif
}

template <ArithmeticOpType op_type>
static __device__ inline half ppl_arithmetic_vector_fp16(half a, half b)
{
    half res;
    res = ppl_arithmetic_scalar<op_type, half>(a, b);
    return res;
}


template <ArithmeticOpType op_type>
static __device__ inline half8_ ppl_arithmetic_vector_fp16(half8_ a, half8_ b)
{
    half8_ res;
    res.x0 = ppl_arithmetic_scalar<op_type, half>(a.x0, b.x0);
    res.y0 = ppl_arithmetic_scalar<op_type, half>(a.y0, b.y0);
    res.z0 = ppl_arithmetic_scalar<op_type, half>(a.z0, b.z0);
    res.w0 = ppl_arithmetic_scalar<op_type, half>(a.w0, b.w0);
    res.x1 = ppl_arithmetic_scalar<op_type, half>(a.x1, b.x1);
    res.y1 = ppl_arithmetic_scalar<op_type, half>(a.y1, b.y1);
    res.z1 = ppl_arithmetic_scalar<op_type, half>(a.z1, b.z1);
    res.w1 = ppl_arithmetic_scalar<op_type, half>(a.w1, b.w1);
    return res;
}

static void ppl_pad_tensor_shape(const ppl::common::TensorShape *tensor_shape0,
                          const ppl::common::TensorShape *tensor_shape1,
                          ppl::common::TensorShape *pad_tensor_shape0,
                          ppl::common::TensorShape *pad_tensor_shape1) {
    int max_dims = std::max(tensor_shape0->GetDimCount(), tensor_shape1->GetDimCount());
    if (pad_tensor_shape0->GetDimCount() < pad_tensor_shape1->GetDimCount()) {
        pad_tensor_shape0->SetDimCount(max_dims);
        // pad 1 to shape_min_pad's higher dim
        int offset = max_dims - tensor_shape0->GetDimCount();
        for (int i = 0; i < offset; i++) {
            pad_tensor_shape0->SetDim(i, 1);
        }
        for (int i = offset; i < max_dims; i++) {
            pad_tensor_shape0->SetDim(i, tensor_shape0->GetDim(i - offset));
        }
    } else {
        pad_tensor_shape1->SetDimCount(max_dims);
        // pad 1 to shape_min_pad's higher dim
        int offset = max_dims - tensor_shape1->GetDimCount();
        for (int i = 0; i < offset; i++) {
            pad_tensor_shape1->SetDim(i, 1);
        }
        for (int i = offset; i < max_dims; i++) {
            pad_tensor_shape1->SetDim(i, tensor_shape1->GetDim(i - offset));
        }
    }
}

static void ppl_refine_tensor_shape(ppl::common::TensorShape *input_shape0,
                                    ppl::common::TensorShape *input_shape1,
                                    ppl::common::TensorShape *output_shape) {
    int dim_count = output_shape->GetDimCount();
    int real_dim_count = dim_count;
    int c_dim_idx = 1;
    for (int i = dim_count - 1; i >= c_dim_idx + 1; i--) {
        bool cur_dim_input0_need_broadcast =
            input_shape0->GetDim(i) != input_shape1->GetDim(i) && input_shape0->GetDim(i) == 1;
        bool cur_dim_input1_need_broadcast =
            input_shape0->GetDim(i) != input_shape1->GetDim(i) && input_shape1->GetDim(i) == 1;
        bool prev_dim_input0_need_broadcast =
            input_shape0->GetDim(i - 1) != input_shape1->GetDim(i - 1) && input_shape0->GetDim(i - 1) == 1;
        bool prev_dim_input1_need_broadcast =
            input_shape0->GetDim(i - 1) != input_shape1->GetDim(i - 1) && input_shape1->GetDim(i - 1) == 1;

        if (cur_dim_input0_need_broadcast == prev_dim_input0_need_broadcast && // can merge
            cur_dim_input1_need_broadcast == prev_dim_input1_need_broadcast) {
            input_shape0->SetDim(i - 1, input_shape0->GetDim(i) * input_shape0->GetDim(i - 1));
            input_shape1->SetDim(i - 1, input_shape1->GetDim(i) * input_shape1->GetDim(i - 1));
            output_shape->SetDim(i - 1, output_shape->GetDim(i) * output_shape->GetDim(i - 1));
            real_dim_count--;
        } else {
            break;
        }
    }
    int dim_diff = dim_count - real_dim_count;
    for (int i = 0; i < dim_diff; ++i) {
        input_shape0->SetDim(dim_count - 1 - i, 1);
        input_shape1->SetDim(dim_count - 1 - i, 1);
        output_shape->SetDim(dim_count - 1 - i, 1);
    }
    input_shape0->SetDimCount(real_dim_count);
    input_shape1->SetDimCount(real_dim_count);
    output_shape->SetDimCount(real_dim_count);
}

static bool ppl_can_one_not_broadcast(const ppl::common::TensorShape *input_shape0,
                                  const ppl::common::TensorShape *input_shape1, int& axis) {
    bool first_shorter = false;
    if (input_shape0->GetRealDimCount() != input_shape1->GetRealDimCount() ||
        input_shape0->CalcElementsExcludingPadding() == input_shape1->CalcElementsExcludingPadding()) {
        return false;
    }
    if (input_shape0->CalcElementsExcludingPadding() < input_shape1->CalcElementsExcludingPadding())  {
        first_shorter = true;
    }
    int dim_count = input_shape0->GetDimCount();
    int not_one_cnt = 0;
    bool pre_not_one = false;
    const ppl::common::TensorShape* test_shape = first_shorter ? input_shape0 : input_shape1;
    for (int i = 0; i < dim_count; ++i) {
        if (test_shape->GetDim(i) != 1 && !pre_not_one) {
            ++not_one_cnt;
            axis = i;
            pre_not_one = true;
        } else if (test_shape->GetDim(i) != 1) {
            axis = i;
        } else if (test_shape->GetDim(i) == 1) {
            pre_not_one = false;
        }
    }
    return (not_one_cnt == 1);
}

static int ppl_get_num_broadcast_dims(const ppl::common::TensorShape *tensor_shape0,
                            const ppl::common::TensorShape *tensor_shape1,
                            int &aixs, bool &bidirectional) {
    ppl::common::TensorShape pad_tensor_shape0 = *tensor_shape0;
    ppl::common::TensorShape pad_tensor_shape1 = *tensor_shape1;
    ppl_pad_tensor_shape(tensor_shape0, tensor_shape1,
            &pad_tensor_shape0, &pad_tensor_shape1);
    int dim_count = pad_tensor_shape0.GetDimCount();
    int num_broadcast_dims = 0;
    int fisrt_broadcast = 0, second_broadcast = 0;
    for(int it = 0; it < dim_count; ++it) {
        if (pad_tensor_shape0.GetDim(it) < pad_tensor_shape1.GetDim(it)) {
            ++num_broadcast_dims; ++fisrt_broadcast;
        } else if (pad_tensor_shape0.GetDim(it) > pad_tensor_shape1.GetDim(it)) {
            ++num_broadcast_dims; ++second_broadcast;
        }
    }
    if (fisrt_broadcast > 0 && second_broadcast > 0) bidirectional = true;
    if (num_broadcast_dims == 1) {
        for(int it = 0; it < dim_count; ++it) {
            if (pad_tensor_shape0.GetDim(it) != pad_tensor_shape1.GetDim(it))
                aixs = it;
        }
    }
    return num_broadcast_dims;
}

bool ppl_feature_broadcast(
    const ppl::common::TensorShape *tensor_shape0,
    const ppl::common::TensorShape *tensor_shape1,
    int *axis)
{
    bool bidirectional = false;
    ppl::common::TensorShape pad_tensor_shape0 = *tensor_shape0;
    ppl::common::TensorShape pad_tensor_shape1 = *tensor_shape1;
    ppl_pad_tensor_shape(tensor_shape0, tensor_shape1,
            &pad_tensor_shape0, &pad_tensor_shape1);
    int dim_count = pad_tensor_shape0.GetDimCount();
    int num_broadcast_dims = 0;
    int fisrt_broadcast = 0, second_broadcast = 0;
    for(int it = 0; it < dim_count; ++it) {
        if (pad_tensor_shape0.GetDim(it) < pad_tensor_shape1.GetDim(it)) {
            ++num_broadcast_dims; ++fisrt_broadcast;
        } else if (pad_tensor_shape0.GetDim(it) > pad_tensor_shape1.GetDim(it)) {
            ++num_broadcast_dims; ++second_broadcast;
        }
    }
    if (fisrt_broadcast > 0 && second_broadcast > 0) bidirectional = true;
    for(int it = 0; it < dim_count; ++it) {
        if (pad_tensor_shape0.GetDim(it) != pad_tensor_shape1.GetDim(it))
        {
            *axis = it;
            break;
        }
    }
    return !bidirectional && (num_broadcast_dims == dim_count - *axis) && (dim_count > 2) && (*axis == 2);
}
void ppl_arithmetic_prepare_strides(
    const ppl::common::TensorShape *tensor_shape0,
    const ppl::common::TensorShape *tensor_shape1,
    const ppl::common::TensorShape *tensor_shape_out,
    const int packed_channel,
    uint32_t *stride_in0,
    uint32_t *stride_in1,
    uint32_t *stride_out)
{
    ppl::common::TensorShape pad_tensor_shape0 = *tensor_shape0;
    ppl::common::TensorShape pad_tensor_shape1 = *tensor_shape1;
    ppl_pad_tensor_shape(tensor_shape0, tensor_shape1,
            &pad_tensor_shape0, &pad_tensor_shape1);

    const int dimCount   = tensor_shape_out->GetDimCount();
    uint32_t stride0     = 1;
    uint32_t stride1     = 1;
    uint32_t stride_out0 = 1;

    for (int i = dimCount - 1; i >= 0; i--) {
        stride_in0[i] = pad_tensor_shape0.GetDim(i) == 1 ? 0 : stride0;
        stride_in1[i] = pad_tensor_shape1.GetDim(i) == 1 ? 0 : stride1;
        stride_out[i] = stride_out0;
        if (i == 1) { // for channel dim, div packed_channel
            stride0 *= (pad_tensor_shape0.GetDim(i) + packed_channel - 1) / packed_channel;
            stride1 *= (pad_tensor_shape1.GetDim(i) + packed_channel - 1) / packed_channel;
            stride_out0 *= (tensor_shape_out->GetDim(i) + packed_channel - 1) / packed_channel;
        } else {
            stride0 *= pad_tensor_shape0.GetDim(i);
            stride1 *= pad_tensor_shape1.GetDim(i);
            stride_out0 *= tensor_shape_out->GetDim(i);
        }
    }
}

void ppl_arithmetic_prepare_strides_nhwc(
    const ppl::common::TensorShape *tensor_shape0,
    const ppl::common::TensorShape *tensor_shape1,
    const ppl::common::TensorShape *tensor_shape_out,
    const int packed_channel,
    uint32_t *stride_in0,
    uint32_t *stride_in1,
    uint32_t *stride_out,
    int suppled_channel = 1)
{
    if (tensor_shape_out->GetDimCount() < 2) return;
    ppl::common::TensorShape pad_tensor_shape0 = *tensor_shape0;
    ppl::common::TensorShape pad_tensor_shape1 = *tensor_shape1;
    ppl_pad_tensor_shape(tensor_shape0, tensor_shape1,
            &pad_tensor_shape0, &pad_tensor_shape1);

    const int dimCount   = tensor_shape_out->GetDimCount();
    uint32_t stride0     = 1;
    uint32_t stride1     = 1;
    uint32_t stride_out0 = 1;

    for (int stride_pos = dimCount - 1; stride_pos >= 0; stride_pos--) {
        int i = stride_pos;
        if (stride_pos == dimCount - 1) i = 1;
        else if (stride_pos == 0) i = 0;
        else i = stride_pos + 1;
        stride_in0[stride_pos] = pad_tensor_shape0.GetDim(i) == 1 ? 0 : stride0;
        stride_in1[stride_pos] = pad_tensor_shape1.GetDim(i) == 1 ? 0 : stride1;
        stride_out[stride_pos] = stride_out0;
        if (i == 1) { // for channel dim, div packed_channel
            stride0 *= (pad_tensor_shape0.GetDim(i) + packed_channel - 1) / packed_channel * suppled_channel;
            stride1 *= (pad_tensor_shape1.GetDim(i) + packed_channel - 1) / packed_channel * suppled_channel;
            stride_out0 *= (tensor_shape_out->GetDim(i) + packed_channel - 1) / packed_channel * suppled_channel;
        } else {
            stride0 *= pad_tensor_shape0.GetDim(i);
            stride1 *= pad_tensor_shape1.GetDim(i);
            stride_out0 *= tensor_shape_out->GetDim(i);
        }
    }
}

static void calculate_nhwc_stride(uint32_t *strides,
    const ppl::common::TensorShape *tensor_shape, int max_dim_count, int packed_channel) {
    if (tensor_shape->IsScalar()) {
        for(int i = 0; i < max_dim_count; ++i) strides[i] = 0;
        return;
    }
    int dim_count = tensor_shape->GetDimCount();
    if (dim_count == 1) {
        for(int i = 0; i < max_dim_count; ++i) strides[i] = 0;
        strides[max_dim_count - 1] = 1;
        return;
    }
    int chl_dim = tensor_shape->GetDim(1);
    if(chl_dim==1)
        strides[1] = 0;
    else
        strides[1] = 1; // chl stride

    int acc_stride = (chl_dim + packed_channel - 1) / packed_channel * packed_channel;
    for(int i = max_dim_count - 1; i >= 0; --i) {
        if (i == 1) continue;
        strides[i] = acc_stride;
        acc_stride *= tensor_shape->GetDim(i);
    }
}

void ppl_arithmetic_prepare_strides_limit_nhwc(
    const ppl::common::TensorShape *tensor_shape0,
    const ppl::common::TensorShape *tensor_shape1,
    const ppl::common::TensorShape *tensor_shape_out,
    const int packed_channel,
    uint32_t *stride_in0,
    uint32_t *stride_in1,
    uint32_t *stride_out)
{
    int max_dim_count = tensor_shape_out->GetDimCount();
    calculate_nhwc_stride(stride_in0, tensor_shape0, max_dim_count, packed_channel);
    calculate_nhwc_stride(stride_in1, tensor_shape1, max_dim_count, packed_channel);
    calculate_nhwc_stride(stride_out, tensor_shape_out, max_dim_count, packed_channel);
}

#define MAXDIMENSIONS 7

struct ArithmeticParam {
    uint32_t stride_in0[MAXDIMENSIONS];
    uint32_t stride_in1[MAXDIMENSIONS];
    uint32_t stride_out[MAXDIMENSIONS];
    uint32_t lastdim_in0;
    uint32_t lastdim_in1;
    uint32_t lastdim_out;
};

template <ArithmeticOpType op_type, typename T1, typename T2>
__global__ void ppl_cukernel_arithmetic_fp16(
    const uint64_t num_elems,
    const int dim_count,
    ArithmeticParam param,
    const T1 *input0,
    const T1 *input1,
    T1 *output)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    int tid = threadIdx.x;
    __shared__ T2 transm[512];
    T1 *transm_half      = reinterpret_cast<T1 *>(transm);
    const T2 *input0_ptr = reinterpret_cast<const T2 *>(input0);
    const T2 *input1_ptr = reinterpret_cast<const T2 *>(input1);
    T2 *output_ptr       = reinterpret_cast<T2 *>(output);

    uint64_t out_index = index;
    uint64_t offset0   = 0;
    uint64_t offset1   = 0;
    for (int i = 0; i < dim_count; i++) {
        uint64_t dim_off = index / param.stride_out[i];
        offset0 += dim_off * param.stride_in0[i];
        offset1 += dim_off * param.stride_in1[i];
        index = index % param.stride_out[i];
    }

    transm[tid + 0]   = input0_ptr[offset0];
    transm[tid + 256] = input1_ptr[offset1];

    transm_half[tid] = ppl_arithmetic_vector_fp16<op_type>(transm_half[tid + 0], transm_half[tid + 256]);

    output_ptr[out_index] = transm[tid];
#endif
}

#ifdef OPT_ARITHM
static int condition_arithmetic(ArithmeticParam param, const ppl::common::TensorShape* input_shape0, const ppl::common::TensorShape* input_shape1){
    int64_t input0_dim_one = input_shape0->GetDim(1) + input_shape0->GetPadding0(1) + input_shape0->GetPadding1(1);
    int64_t input1_dim_one = input_shape1->GetDim(1) + input_shape1->GetPadding0(1) + input_shape1->GetPadding1(1);
    if(input0_dim_one == input1_dim_one)
    {
        return 1;
    }
    return 0;
}

template <ArithmeticOpType op_type>
__global__ void ppl_cukernel_arithmetic_fp16_opt_sc(
    const uint64_t num_elems,
    const int dim_count,
    ArithmeticParam param,
    const half *input0,
    const half *input1,
    half *output)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    uint64_t index = (blockIdx.x * blockDim.x + threadIdx.x) << 3;
    if(index >= num_elems) return;
    uint64_t out_index = index;
    uint64_t offset0 = 0;
    uint64_t offset1 = 0;
    for (int i = 0; i < dim_count; i++) {
        uint64_t dim_off = index / param.stride_out[i];
        offset0 += dim_off * param.stride_in0[i];
        offset1 += dim_off * param.stride_in1[i];
        index = index % param.stride_out[i];
    }
    float4 vIn0, vIn1, vdst;
    vIn0 = *(float4*)(input0 + offset0);
    vIn1 = *(float4*)(input1 + offset1);
    half*ptr_vIn0 = (half*)&vIn0;
    half*ptr_vIn1 = (half*)&vIn1;
    half*ptr_vdst = (half*)&vdst;
    #pragma unroll 8
    for(int i = 0; i < 8; i++){
        ptr_vdst[i] = ppl_arithmetic_vector_fp16<op_type>(ptr_vIn0[i], ptr_vIn1[i]);
    }
    *(float4*)(output + out_index) = vdst;
#endif
}

template <ArithmeticOpType op_type, int N, typename T1, typename T2>
__global__ void ppl_cukernel_arithmetic_fp16_nhwc8_opt(
    const uint64_t num_elems,
    const T2 *input0,
    const T2 *input1,
    T2 *output,
    DivModFast axis_width_mod_fast)
{
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;

    int inner_idx = axis_width_mod_fast.mod(index);
    float4 input0_val = input0[index];
    float4 input1_val = input1[inner_idx];
    float4 output_val = make_float4(0.f, 0.f, 0.f, 0.f);

    T1* input0_ptr = (T1*)&input0_val;
    T1* input1_ptr = (T1*)&input1_val;
    T1* output_ptr = (T1*)&output_val;

    #pragma unroll N
    for (int i = 0; i < N; i++) {
        output_ptr[i] = ppl_arithmetic_scalar<op_type, T1>(input0_ptr[i], input1_ptr[i]);
    }
    output[index] = output_val;
}
#endif//OPT_ARITHM

template<ArithmeticOpType op_type, typename T>
__global__ void ppl_cukernel_arithmetic(
    const uint64_t num_elems,
    const int dim_count,
    ArithmeticParam param,
    const T *input0,
    const T* input1,
    T *output) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems) return;

    uint64_t out_index = index;
    uint64_t offset0 = 0;
    uint64_t offset1 = 0;
    for (int i = 0; i < dim_count; i++) {
        uint64_t dim_off = index / param.stride_out[i];
        offset0 += dim_off * param.stride_in0[i];
        offset1 += dim_off * param.stride_in1[i];
        index = index % param.stride_out[i];
    }

    output[out_index] = ppl_arithmetic_scalar<op_type, T>(input0[offset0], input1[offset1]);
}

#ifdef OPT_ARITHM
struct DivModMetadata {
    DivModFast modFast[MAXDIMENSIONS];
};

template<ArithmeticOpType op_type, typename T, int N>
__global__ void ppl_cukernel_arithmetic_opt(
    const uint64_t num_elems,
    const int dim_count,
    ArithmeticParam param,
    const T *input0,
    const T* input1,
    T *output,
    DivModMetadata meta_data) {
    int64_t offset = blockIdx.x * blockDim.x * N;
    int64_t blocklength = min(num_elems - offset,blockDim.x * N);
    if(blocklength <= 0) return;
    T * ptr_block_output = output + offset;
    for(int i = threadIdx.x; i < blocklength; i += blockDim.x) {
        uint64_t index = offset + i;
        uint64_t out_index = index;
        uint64_t offset0 = 0;
        uint64_t offset1 = 0;
        int dim_off, idx_off;
        for (int i = 0; i < (dim_count); i++) {
            meta_data.modFast[i].divmod(index, dim_off, idx_off);
            offset0 += dim_off * param.stride_in0[i];
            offset1 += dim_off * param.stride_in1[i];
            index = idx_off;
        }
        
        ptr_block_output[i] = ppl_arithmetic_scalar<op_type, T>(input0[offset0], input1[offset1]);
    }
}

template<ArithmeticOpType op_type, typename T, typename Vec_T, int N>
__global__ void ppl_cukernel_arithmetic_opt1(
    const uint64_t num_elems,
    const int dim_count,
    ArithmeticParam param,
    const T *input0,
    const T* input1,
    T *output,
    DivModMetadata meta_data) {
    uint64_t index = (blockIdx.x * blockDim.x + threadIdx.x) * N;
    if (index >= num_elems) return;

    uint64_t out_index = index;
    uint64_t offset0 = 0;
    uint64_t offset1 = 0;
    int dim_off, idx_off;
    for (int i = 0; i < dim_count; i++) {
        meta_data.modFast[i].divmod(index, dim_off, idx_off);
        offset0 += dim_off * param.stride_in0[i];
        offset1 += dim_off * param.stride_in1[i];
        index = idx_off;
    }
    Vec_T dst;
    Vec_T src0 , src1;
    T* ptr_dst = (T*)&dst;
    T* ptr_src0 = (T*)&src0;
    T* ptr_src1 = (T*)&src1;
    src0 = *(Vec_T*)(input0 + offset0);
    src1 = *(Vec_T*)(input1 + offset1);
    #pragma unroll N
    for(int i = 0; i < N; i++) {
        ptr_dst[i] = ppl_arithmetic_scalar<op_type, T>(ptr_src0[i], ptr_src1[i]);
    }
    *(Vec_T*)(output + out_index) = dst;
}

template<ArithmeticOpType op_type, typename T, int N>
__global__ void ppl_cukernel_arithmetic_XN(
    const uint64_t num_elems,
    const int dim_count,
    ArithmeticParam param,
    const T *input0,
    const T* input1,
    T *output) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    index *= N;
    if (index >= num_elems) return;

    uint64_t out_index = index;
    uint64_t offset0 = 0;
    uint64_t offset1 = 0;
    for (int i = 0; i < dim_count; i++) {
        uint64_t dim_off = index / param.stride_out[i];
        offset0 += dim_off * param.stride_in0[i];
        offset1 += dim_off * param.stride_in1[i];
        index = index % param.stride_out[i];
    }
    const float4* i0 = (float4*)(input0 + offset0);
    const float4* i1 = (float4*)(input1 + offset1);
    const float4 x0 = i0[0];
    const T* x0_ptr = reinterpret_cast<const T*>(&x0);
    const float4 x1 = i1[0];
    const T* x1_ptr = reinterpret_cast<const T*>(&x1);

    T ip0[N], ip1[N];
    #pragma unroll N
    for (int i = 0; i < N; i++) ip0[i] = x0_ptr[i];
    #pragma unroll N
    for (int i = 0; i < N; i++) ip1[i] = x1_ptr[i];
    float4 out_val;
    T* out_val_ptr = reinterpret_cast<T*>(&out_val);
    #pragma unroll N
    for (int i = 0; i < N; i++) out_val_ptr[i] = ppl_arithmetic_scalar<op_type, T>(ip0[i], ip1[i]);
    *(float4*)(output+out_index) = out_val;
}

template<ArithmeticOpType op_type, typename T, int N>
__global__ void ppl_cukernel_arithmetic_limit_nhwc8_broadcast_c_float4_opt(
    const uint64_t num_elems,
    const int64_t channels,
    const T *input0,
    const T* input1,
    T *output) {
    uint64_t index = (blockIdx.x * blockDim.x + threadIdx.x) * N;
    if (index >= num_elems) return;

    float4 dst;
    uint64_t offset = index / channels;
    float4 val1 = *(float4 *)(input0 + index);
    T val2      = *(input1 + offset * 8); // nhwc8, padding c1 -> c8
    T* ptr_val1 = (T*)&val1;
    T* ptr_dst  = (T*)&dst;
    #pragma unroll N
    for(int i = 0; i < N; i++){
        ptr_dst[i] = ppl_arithmetic_scalar<op_type, T>(ptr_val1[i], val2);
    }
    *(float4*)(output + index) = dst;
}
#endif

template<ArithmeticOpType op_type, typename T>
__global__ void ppl_cukernel_arithmetic_limit_nhwc(
    const uint64_t num_elems,
    const int dim_count,
    ArithmeticParam param_ndarray,
    ArithmeticParam param_nhwc,
    const T *input0,
    const T* input1,
    T *output) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems) return;

    uint64_t offset0 = 0;
    uint64_t offset1 = 0;
    uint64_t out_offset = 0;
    for (int i = 0; i < dim_count; i++) {
        uint64_t dim_off = index / param_ndarray.stride_out[i];
        offset0 += dim_off * param_nhwc.stride_in0[i];
        offset1 += dim_off * param_nhwc.stride_in1[i];
        out_offset += dim_off * param_nhwc.stride_out[i];
        index = index % param_ndarray.stride_out[i];
    }

    output[out_offset] = ppl_arithmetic_scalar<op_type, T>(input0[offset0], input1[offset1]);
}

template<ArithmeticOpType op_type, typename T>
__global__ void ppl_cukernel_arithmetic_limit_nhwc_int8(
    const uint64_t num_elems,
    const int dim_count,
    ArithmeticParam param_ndarray,
    ArithmeticParam param_nhwc,
    const T *input0,
    const T* input1,
    T *output,
    float in_scale0,
    float in_scale1,
    float out_scale) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems) return;

    uint64_t offset0 = 0;
    uint64_t offset1 = 0;
    uint64_t out_offset = 0;
    for (int i = 0; i < dim_count; i++) {
        uint64_t dim_off = index / param_ndarray.stride_out[i];
        offset0 += dim_off * param_nhwc.stride_in0[i];
        offset1 += dim_off * param_nhwc.stride_in1[i];
        out_offset += dim_off * param_nhwc.stride_out[i];
        index = index % param_ndarray.stride_out[i];
    }

    output[out_offset] = ppl_arithmetic_scalar_int8<op_type, T>(input0[offset0], input1[offset1],
            in_scale0, in_scale1, out_scale);
}

template<ArithmeticOpType op_type, typename T>
__global__ void ppl_cukernel_arithmetic_int8(
    const uint64_t num_elems,
    const int dim_count,
    ArithmeticParam param,
    const T *input0,
    const T* input1,
    T *output,
    float in_scale0 = 0,
    float in_scale1 = 0,
    float out_scale = 0) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems) return;

    uint64_t out_index = index;
    uint64_t offset0 = 0;
    uint64_t offset1 = 0;
    for (int i = 0; i < dim_count; i++) {
        uint64_t dim_off = index / param.stride_out[i];
        offset0 += dim_off * param.stride_in0[i];
        offset1 += dim_off * param.stride_in1[i];
        index = index % param.stride_out[i];
    }

    output[out_index] = ppl_arithmetic_scalar_int8<op_type, T>(input0[offset0], input1[offset1], in_scale0, in_scale1, out_scale);
}

template<ArithmeticOpType op_type, typename T>
__global__ void ppl_cukernel_arithmetic_nobroadcast(
    const uint64_t num_elems,
    const T *input0,
    const T* input1,
    T *output) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems) return;
    output[index] = ppl_arithmetic_scalar<op_type, T>(input0[index], input1[index]);
}

#ifdef OPT_ARITHM
template<ArithmeticOpType op_type, typename T>
__global__ void ppl_cukernel_arithmetic_nobroadcast_opt_4(
    const uint64_t num_elems,
    const T *input0,
    const T* input1,
    T *output) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    index *= 4;
    if (index >= num_elems) return;
    T in_val00 = input0[index], in_val01 = input0[index+1];
    T in_val10 = input1[index], in_val11 = input1[index+1];
    T out_val0 = ppl_arithmetic_scalar<op_type, T>(in_val00, in_val10);
    T in_val02 = input0[index+2];
    T in_val12 = input1[index+2];
    T out_val1 = ppl_arithmetic_scalar<op_type, T>(in_val01, in_val11);
    T in_val03 = input0[index+3];
    T in_val13 = input1[index+3];
    T out_val2 = ppl_arithmetic_scalar<op_type, T>(in_val02, in_val12);
    T out_val3 = ppl_arithmetic_scalar<op_type, T>(in_val03, in_val13);
    output[index] = out_val0;
    output[index+1] = out_val1;
    output[index+2] = out_val2;
    output[index+3] = out_val3;
}

template<ArithmeticOpType op_type, typename T, int N>
__global__ void ppl_cukernel_arithmetic_nobroadcast_opt(
    const uint64_t num_elems,
    const T *input0,
    const T* input1,
    T *output) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    index *= N;
    if (index >= num_elems) return;
    T in_val0[N];
    T in_val1[N];
    T out_val[N];
    #pragma unroll N
    for (int i = 0; i < N; i++) {
        in_val0[i] = input0[index+i];
    }
    #pragma unroll N
    for (int i = 0; i < N; i++) {
        in_val1[i] = input1[index+i];
    }

    #pragma unroll N
    for (int i = 0; i < N; i++) {
        out_val[i] = ppl_arithmetic_scalar<op_type, T>(in_val0[i], in_val1[i]);
    }

    #pragma unroll N
    for (int i = 0; i < N; i++) {
        output[index+i] = out_val[i];
    }
}
#endif

template<ArithmeticOpType op_type, typename T>
__global__ void ppl_cukernel_arithmetic_nobroadcast_float4(
    const uint64_t num_elems,
    const T *input0,
    const T* input1,
    T *output) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems) return;
    T val_a = input0[index];
    T val_b = input1[index];
    T val_out;
    const half* val_a_ptr = reinterpret_cast<const half*>(&val_a);
    const half* val_b_ptr = reinterpret_cast<const half*>(&val_b);
    half* val_out_ptr = reinterpret_cast<half*>(&val_out);
    constexpr int VEC_SIZE = 8;
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        val_out_ptr[i] = ppl_arithmetic_scalar<op_type, half>(val_a_ptr[i], val_b_ptr[i]);
    }
    output[index] = val_out;
}

template<ArithmeticOpType op_type, typename T>
__global__ void ppl_cukernel_arithmetic_nobroadcast_int8(
    const uint64_t num_elems,
    const T *input0,
    const T* input1,
    T *output,
    float in_scale0 = 0,
    float in_scale1 = 0,
    float out_scale = 0) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems) return;
    output[index] = ppl_arithmetic_scalar_int8<op_type, T>(input0[index], input1[index], in_scale0, in_scale1, out_scale);
}


#ifdef OPT_ARITHM
template<ArithmeticOpType op_type, typename T>
__global__ void ppl_cukernel_arithmetic_nobroadcast_int8_opt_16(
    const uint64_t num_elems,
    const T *input0,
    const T* input1,
    T *output,
    float in_scale0 = 0,
    float in_scale1 = 0,
    float out_scale = 0) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    index <<= 4;
    if (index >= num_elems) return;
    const uint64_t* ip = (uint64_t*)(input0+index);
    uint64_t val0_0 = ip[0], val0_1 = ip[1];
    ip = (uint64_t*)(input1+index);
    uint64_t val1_0 = ip[0], val1_1 = ip[1];
    uint64_t out0 = 0, out1 = 0;
    for (int i = 0; i < 8; i++) {
        T in_val0 = (int8_t)(val0_0&0xFF), in_val1 = (int8_t)(val1_0&0xFF);
        T out_val = ppl_arithmetic_scalar_int8<op_type, T>(in_val0, in_val1, in_scale0, in_scale1, out_scale);
        out0 |= ((((uint64_t)out_val) & 0xFF) << (i<<3));
        val0_0 >>= 8;
        val1_0 >>= 8;
    }
    for (int i = 0; i < 8; i++) {
        T in_val0 = (int8_t)(val0_1&0xFF), in_val1 = (int8_t)(val1_1&0xFF);
        T out_val = ppl_arithmetic_scalar_int8<op_type, T>(in_val0, in_val1, in_scale0, in_scale1, out_scale);
        out1 |= ((((uint64_t)out_val) & 0xFF) << (i<<3));
        val0_1 >>= 8;
        val1_1 >>= 8;
    }
    uint64_t* op = (uint64_t*)(output+index);
    op[0] = out0;
    op[1] = out1;
}

template<ArithmeticOpType op_type, typename T>
__global__ void ppl_cukernel_arithmetic_nobroadcast_int8_opt_16_no_scale(
    const uint64_t num_elems,
    const T *input0,
    const T* input1,
    T *output) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems) return;
    float4 in0_val = input0[index];
    float4 in1_val = input1[index];
    float4 out_val;

    int8_t* in0_ptr = (int8_t*)&in0_val;
    int8_t* in1_ptr = (int8_t*)&in1_val;
    int8_t* out_ptr = (int8_t*)&out_val;

    #pragma unroll 16
    for (int i = 0; i < 16; i++) {
        out_ptr[i] = ppl_arithmetic_scalar_int8_no_scale<op_type, int8_t>(in0_ptr[i], in1_ptr[i]);
    }

    output[index] = out_val;
}
#endif

template<ArithmeticOpType op_type, typename T>
__global__ void ppl_cukernel_arithmetic_one_scalar(
    const uint64_t num_elems,
    const bool first_shorter,
    const T *input0,
    const T* input1,
    T *output) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems) return;
    int calc_index = 0;
    uint64_t offset0 = first_shorter ? calc_index : index;
    uint64_t offset1 = first_shorter ? index : calc_index;
    output[index] = ppl_arithmetic_scalar<op_type, T>(input0[offset0], input1[offset1]);
}

template<ArithmeticOpType op_type, typename T>
__global__ void ppl_cukernel_arithmetic_one_scalar_int8(
    const uint64_t num_elems,
    const bool first_shorter,
    const T *input0,
    const T* input1,
    T *output,
    float in_scale0 = 0,
    float in_scale1 = 0,
    float out_scale = 0) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems) return;
    int calc_index = 0;
    uint64_t offset0 = first_shorter ? calc_index : index;
    uint64_t offset1 = first_shorter ? index : calc_index;
    output[index] = ppl_arithmetic_scalar_int8<op_type, T>(input0[offset0], input1[offset1], in_scale0, in_scale1, out_scale);
}

#ifdef OPT_ARITHM
template<ArithmeticOpType op_type, typename T, int N>
__global__ void ppl_cukernel_arithmetic_one_scalar_first_shorter_opt(
    const uint64_t num_elems,
    const T *input0,
    const T *input1,
    T *output) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    index*=N;
    if (index >= num_elems) return;
    T in_val1[N];
    T out_val[N];
    T ip0 = input0[0];
    #pragma unroll N
    for (int i = 0; i < N; i++) in_val1[i] = input1[index+i];
    #pragma unroll N
    for (int i = 0; i < N; i++) out_val[i] = ppl_arithmetic_scalar<op_type, T>(ip0, in_val1[i]);
    #pragma unroll N
    for (int i = 0; i < N; i++) output[index+i] = out_val[i];
}

template<ArithmeticOpType op_type, typename T, int N>
__global__ void ppl_cukernel_arithmetic_one_scalar_not_first_shorter_opt(
    const uint64_t num_elems,
    const T *input0,
    const T *input1,
    T *output) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    index*=N;
    if (index >= num_elems) return;
    T in_val0[N];
    T out_val[N];
    T ip1 = input1[0];
    #pragma unroll N
    for (int i = 0; i < N; i++) in_val0[i] = input0[index+i];
    #pragma unroll N
    for (int i = 0; i < N; i++) out_val[i] = ppl_arithmetic_scalar<op_type, T>(in_val0[i], ip1);
    #pragma unroll N
    for (int i = 0; i < N; i++) output[index+i] = out_val[i];
}

template<ArithmeticOpType op_type, typename T, int N>
__global__ void ppl_cukernel_arithmetic_one_scalar_int8_first_shorter_opt(
    const uint64_t num_elems,
    const T *input0,
    const T* input1,
    T *output,
    float in_scale0 = 0,
    float in_scale1 = 0,
    float out_scale = 0) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    index *= N;
    if (index >= num_elems) return;
    T in_val1[N];
    T out_val[N];
    T ip0 = input0[0];
    #pragma unroll N
    for (int i = 0; i < N; i++) in_val1[i] = input1[index+i];
    #pragma unroll N
    for (int i = 0; i < N; i++) out_val[i] = ppl_arithmetic_scalar_int8<op_type, T>(ip0, in_val1[i], in_scale0, in_scale1, out_scale);
    #pragma unroll N
    for (int i = 0; i < N; i++) output[index+i] = out_val[i];
}

template<ArithmeticOpType op_type, typename T, int N>
__global__ void ppl_cukernel_arithmetic_one_scalar_int8_not_first_shorter_opt(
    const uint64_t num_elems,
    const T *input0,
    const T *input1,
    T *output,
    float in_scale0 = 0,
    float in_scale1 = 0,
    float out_scale = 0) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    index *= N;
    if (index >= num_elems) return;
    T in_val0[N];
    T out_val[N];
    T ip1 = input1[0];
    #pragma unroll N
    for (int i = 0; i < N; i++) in_val0[i] = input0[index+i];
    #pragma unroll N
    for (int i = 0; i < N; i++) out_val[i] = ppl_arithmetic_scalar_int8<op_type, T>(in_val0[i], ip1, in_scale0, in_scale1, out_scale);
    #pragma unroll N
    for (int i = 0; i < N; i++) output[index+i] = out_val[i];
}

template<ArithmeticOpType op_type, typename T, int N>
__global__ void ppl_cukernel_arithmetic_one_scalar_first_shorter_combopt(
    const uint64_t num_elems,
    const T *input0,
    const T *input1,
    T *output) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    index *= N;
    if (index >= num_elems) return;

    float4 vIn1, vdst;
    T vIn0 = *input0;
    vIn1 = *(float4*)(input1 + index);

    T *ptr_vIn1 = (T *)&vIn1;
    T *ptr_vdst = (T *)&vdst;

    #pragma unroll N
    for (int i = 0; i < N; i++)
    {
        ptr_vdst[i] = ppl_arithmetic_scalar<op_type, T>(vIn0, ptr_vIn1[i]);
    }
    *(float4*)(output + index) = vdst;
}

template<ArithmeticOpType op_type, typename T, int N>
__global__ void ppl_cukernel_arithmetic_one_scalar_not_first_shorter_combopt(
    const uint64_t num_elems,
    const T *input0,
    const T *input1,
    T *output) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    index *= N;
    if (index >= num_elems) return;

    float4 vIn0, vdst;
    vIn0 = *(float4*)(input0 + index);
    T vIn1 = *input1;

    T *ptr_vIn0 = (T *)&vIn0;
    T *ptr_vdst = (T *)&vdst;

    #pragma unroll N
    for (int i = 0; i < N; i++)
    {
        ptr_vdst[i] = ppl_arithmetic_scalar<op_type, T>(ptr_vIn0[i], vIn1);
    }
    *(float4*)(output + index) = vdst;
}

template<ArithmeticOpType op_type, typename T1, typename T2, int N>
__global__ void ppl_cukernel_arithmetic_one_dimension_opt(
    const uint64_t num_elems,
    const int32_t inner_dim,
    const int32_t channel_dim,
    const bool first_shorter,
    const T2 *input0,
    const T2* input1,
    T2 *output) {
    uint64_t index = (blockIdx.x * blockDim.x + threadIdx.x) * N;
    if (index >= num_elems) return;
    int calc_index = (index / channel_dim) % inner_dim;

    T1 val1 = *(T1*)(input0 + index);
    T1 val2 = *(T1*)(input1 + calc_index);
    T1 out_val;

    T2* val1_ptr = (T2*)&val1;
    T2* val2_ptr = (T2*)&val2;
    T2* out_val_ptr = (T2*)&out_val;

    #pragma unroll N
    for (int i = 0; i < N; i++) {
        out_val_ptr[i] = ppl_arithmetic_scalar<op_type, T2>(val1_ptr[i], val2_ptr[i]);
    }
    *(T1*)(output + index) = out_val;
}
#endif

template<ArithmeticOpType op_type, typename T>
__global__ void ppl_cukernel_arithmetic_one_dimension(
    const uint64_t num_elems,
    const int32_t inner_dim,
    const int32_t channel_dim,
    const bool first_shorter,
    const T *input0,
    const T* input1,
    T *output) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems) return;
    int calc_index = (index / channel_dim) % inner_dim;
    uint64_t offset0 = first_shorter ? calc_index : index;
    uint64_t offset1 = first_shorter ? index : calc_index;
    output[index] = ppl_arithmetic_scalar<op_type, T>(input0[offset0], input1[offset1]);
}

#ifdef OPT_ARITHM
template<ArithmeticOpType op_type, typename T, bool first_shorter>
__global__ void ppl_cukernel_arithmetic_one_dimension_s_t2c1x8(
    const uint64_t num_elems,
    int inner_dim,
    DivModFast fast_inner_dim,
    const T *input0,
    const T* input1,
    T *output) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    index*=8;
    __shared__ T shorter[4096];
    if (index >= num_elems) return;
    for (int i = threadIdx.x; i < inner_dim; i+=blockDim.x) {
        shorter[i] = first_shorter ? input0[i] : input1[i];
    }
    __syncthreads();
    const float4* ptr = (float4*)((first_shorter ? input1 : input0) + index);
    const float4 x = ptr[0];

    T vs[8];
    #pragma unroll 8
    for (int i = 0; i < 8; i++) {
        vs[i] = shorter[fast_inner_dim.mod(index+i)];
    }

    const T* t_ptr = (const T*)&x;
    float4 out_v;
    T* out_v_ptr = (T*)&out_v;
    #pragma unroll 8
    for (int i = 0; i < 8; i++) {
        out_v_ptr[i] = ppl_arithmetic_scalar<op_type, T>(t_ptr[i], vs[i]);
    }

    float4* out_ptr = (float4*)(output + index);
    out_ptr[0] = out_v;
}

template<ArithmeticOpType op_type, typename T, bool first_shorter>
__global__ void ppl_cukernel_arithmetic_one_dimension_t2c1x8(
    const uint64_t num_elems,
    int inner_dim,
    DivModFast fast_inner_dim,
    const T *input0,
    const T* input1,
    T *output) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    index*=8;

    if (index >= num_elems) return;
    const float4* ptr = (float4*)((first_shorter ? input1 : input0) + index);
    const float4 x = ptr[0];

    T vs[8];
    const T* shorter = first_shorter ? input0 : input1;
    #pragma unroll 8
    for (int i = 0; i < 8; i++) {
        vs[i] = shorter[fast_inner_dim.mod(index+i)];
    }

    const T* t_ptr = (const T*)&x;
    float4 out_v;
    T* out_v_ptr = (T*)&out_v;
    #pragma unroll 8
    for (int i = 0; i < 8; i++) {
        out_v_ptr[i] = ppl_arithmetic_scalar<op_type, T>(t_ptr[i], vs[i]);
    }

    float4* out_ptr = (float4*)(output + index);
    out_ptr[0] = out_v;
}

template<ArithmeticOpType op_type, typename T, bool first_shorter>
__global__ void ppl_cukernel_arithmetic_one_dimension_s_t2cnx8(
    const uint64_t num_elems,
    int inner_dim,
    DivModFast fast_inner_dim,
    DivModFast fast_channel_dim,
    const T *input0,
    const T* input1,
    T *output) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    index*=8;
    __shared__ T shorter[4096];
    if (index >= num_elems) return;
    for (int i = threadIdx.x; i < inner_dim; i+=blockDim.x) {
        shorter[i] = first_shorter ? input0[i] : input1[i];
    }
    __syncthreads();
    const float4* ptr = (float4*)((first_shorter ? input1 : input0) + index);
    const float4 x = ptr[0];
    T vs[8];
    #pragma unroll 8
    for (int i = 0; i < 8; i++) {
        vs[i] = shorter[fast_inner_dim.mod(fast_channel_dim.div(index+i))];
    }

    const T* t_ptr = (const T*)&x;
    float4 out_v;
    T* out_v_ptr = (T*)&out_v;
    #pragma unroll 8
    for (int i = 0; i < 8; i++) {
        out_v_ptr[i] = ppl_arithmetic_scalar<op_type, T>(t_ptr[i], vs[i]);
    }

    float4* out_ptr = (float4*)(output + index);
    out_ptr[0] = out_v;
}

template<ArithmeticOpType op_type, typename T, bool first_shorter>
__global__ void ppl_cukernel_arithmetic_one_dimension_t2cnx8(
    const uint64_t num_elems,
    int inner_dim,
    DivModFast fast_inner_dim,
    DivModFast fast_channel_dim,
    const T *input0,
    const T* input1,
    T *output) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    index*=8;

    if (index >= num_elems) return;
    const float4* ptr = (float4*)((first_shorter ? input1 : input0) + index);
    const float4 x = ptr[0];

    T vs[8];
    const T* shorter = first_shorter ? input0 : input1;
    #pragma unroll 8
    for (int i = 0; i < 8; i++) {
        vs[i] = shorter[fast_inner_dim.mod(fast_channel_dim.div(index+i))];
    }

    const T* t_ptr = (const T*)&x;
    float4 out_v;
    T* out_v_ptr = (T*)&out_v;
    #pragma unroll 8
    for (int i = 0; i < 8; i++) {
        out_v_ptr[i] = ppl_arithmetic_scalar<op_type, T>(t_ptr[i], vs[i]);
    }

    float4* out_ptr = (float4*)(output + index);
    out_ptr[0] = out_v;
}

template<ArithmeticOpType op_type, typename T, int N>
__global__ void ppl_cukernel_arithmetic_one_not_broadcast_float4_opt(
    const uint64_t num_elems,
    const int32_t axis_lgt,
    const int32_t inner_dim,
    const bool first_shorter,
    const T *input0,
    const T* input1,
    T *output) {
    uint64_t index = (blockIdx.x * blockDim.x + threadIdx.x) * N;
    if (index >= num_elems) return;
    int calc_index = (index / inner_dim) % axis_lgt;

    float4 val0 = *(float4*)(input0 + index);
    T val1      = *(input1 + calc_index);
    float4 dst;
    T* ptr_val0 = (T*)&val0;
    T* ptr_dst  = (T*)&dst;

    #pragma unroll N
    for (int i = 0; i < N; i++) {
        ptr_dst[i] = ppl_arithmetic_scalar<op_type, T>(ptr_val0[i], val1);
    }
    *(float4*)(output + index) = dst;
}
#endif

template<ArithmeticOpType op_type, typename T>
__global__ void ppl_cukernel_arithmetic_one_not_broadcast(
    const uint64_t num_elems,
    const int32_t axis_lgt,
    const int32_t inner_dim,
    const bool first_shorter,
    const T *input0,
    const T* input1,
    T *output) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems) return;
    int calc_index = (index / inner_dim) % axis_lgt;
    uint64_t offset0 = first_shorter ? calc_index : index;
    uint64_t offset1 = first_shorter ? index : calc_index;
    output[index] = ppl_arithmetic_scalar<op_type, T>(input0[offset0], input1[offset1]);
}

template<ArithmeticOpType op_type, typename T>
__global__ void ppl_cukernel_arithmetic_one_dimension_int8(
    const uint64_t num_elems,
    const int32_t inner_dim,
    const bool first_shorter,
    const T *input0,
    const T* input1,
    T *output,
    float in_scale0 = 0,
    float in_scale1 = 0,
    float out_scale = 0) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems) return;
    int calc_index = index % inner_dim;
    uint64_t offset0 = first_shorter ? calc_index : index;
    uint64_t offset1 = first_shorter ? index : calc_index;
    output[index] = ppl_arithmetic_scalar_int8<op_type, T>(input0[offset0], input1[offset1], in_scale0, in_scale1, out_scale);
}

template<ArithmeticOpType op_type, typename T>
__global__ void ppl_cukernel_arithmetic_one_broadcast(
    const uint64_t num_elems,
    const int outer_stride,
    const int inner_dim,
    const bool first_shorter,
    const T *input0,
    const T* input1,
    T *output) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems) return;
    int inner_idx = index % inner_dim;
    int outer_idx = index / outer_stride;
    uint64_t calc_index = outer_idx * inner_dim + inner_idx;
    uint64_t offset0 = first_shorter ? calc_index : index;
    uint64_t offset1 = first_shorter ? index : calc_index;
    output[index] = ppl_arithmetic_scalar<op_type, T>(input0[offset0], input1[offset1]);
}

#ifdef OPT_ARITHM
template<ArithmeticOpType op_type, typename T, int N>
__global__ void ppl_cukernel_arithmetic_one_broadcast_float4_opt(
    const uint64_t num_elems,
    const int outer_stride,
    const int inner_dim,
    const bool first_shorter,
    const T *input0,
    const T* input1,
    T *output) {
    uint64_t index = (blockIdx.x * blockDim.x + threadIdx.x) * N;
    if (index >= num_elems) return;
    int inner_idx = index % inner_dim;
    int outer_idx = index / outer_stride;
    uint64_t calc_index = outer_idx * inner_dim + inner_idx;

    float4 val0 = *(float4*)(input0 + index);
    float4 val1 = *(float4*)(input1 + calc_index);
    float4 dst;
    T* ptr_val0 = (T*)&val0;
    T* ptr_val1 = (T*)&val1;
    T* ptr_dst  = (T*)&dst;

    #pragma unroll N
    for (int i = 0; i < N; i++) {
        ptr_dst[i] = ppl_arithmetic_scalar<op_type, T>(ptr_val0[i], ptr_val1[i]);
    }
    *(float4*)(output + index) = dst;
}

template<ArithmeticOpType op_type, typename T>
__global__ void ppl_cukernel_arithmetic_one_broadcast_first_shorter_fp16(
    const uint64_t num_elems,
    const int outer_stride,
    const int inner_dim,
    DivModFast broad_mod_fast,
    DivModFast broad_div_fast,
    const T *input0,
    const T* input1,
    T *output) {
    uint64_t index = (blockIdx.x * blockDim.x + threadIdx.x) << 3;
    if (index >= num_elems) return;
    int inner_idx = broad_mod_fast.mod(index);
    int outer_idx = broad_div_fast.div(index);

    uint64_t calc_index = outer_idx * inner_dim + inner_idx;

    float4 vIn0, vIn1, vdst;
    vIn0 = *(float4*)(input0 + calc_index);
    vIn1 = *(float4*)(input1 + index);

    T *ptr_vIn0 = (T *)&vIn0;
    T *ptr_vIn1 = (T *)&vIn1;
    T *ptr_vdst = (T *)&vdst;

    #pragma unroll 8
    for (int i = 0; i < 8; i++)
    {
        ptr_vdst[i] = ppl_arithmetic_scalar<op_type, T>(ptr_vIn0[i], ptr_vIn1[i]);
    }
    *(float4*)(output + index) = vdst;
}

template<ArithmeticOpType op_type, typename T>
__global__ void ppl_cukernel_arithmetic_one_broadcast_not_first_shorter_fp16(
    const uint64_t num_elems,
    const int outer_stride,
    const int inner_dim,
    DivModFast broad_mod_fast,
    DivModFast broad_div_fast,
    const T *input0,
    const T *input1,
    T *output) {
    uint64_t index = (blockIdx.x * blockDim.x + threadIdx.x) << 3;
    if (index >= num_elems) return;
    int inner_idx = broad_mod_fast.mod(index);
    int outer_idx = broad_div_fast.div(index);
    uint64_t calc_index = outer_idx * inner_dim + inner_idx;

    float4 vIn0,vIn1,vdst;
    vIn0 = *(float4*)(input0 + index);
    vIn1 = *(float4*)(input1 + calc_index);

    T *ptr_vIn0 = (T *)&vIn0;
    T *ptr_vIn1 = (T *)&vIn1;
    T *ptr_vdst = (T *)&vdst;

    #pragma unroll 8
    for (int i = 0; i < 8; i++)
    {
        ptr_vdst[i] = ppl_arithmetic_scalar<op_type, T>(ptr_vIn0[i], ptr_vIn1[i]);
    }
    *(float4*)(output + index) = vdst;
}

//inner_dim == 1
template<ArithmeticOpType op_type, typename T>
__global__ void ppl_cukernel_arithmetic_one_broadcast_opt0(
    const uint64_t num_elems,
    const int outer_stride,
    const int inner_dim,
    const bool first_shorter,
    const float r_outer_stride,
    const T *input0,
    const T* input1,
    T *output) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems) return;
    int outer_idx = index / outer_stride;
    uint64_t calc_index = outer_idx * inner_dim;
    output[index] = ppl_arithmetic_scalar<op_type, T>(input0[index], input1[calc_index]);
}

template<ArithmeticOpType op_type, int N, typename T1, typename T2>
__global__ void ppl_cukernel_arithmetic_one_broadcast_opt1(
    const uint64_t num_elems,
    DivModFast outer_stride_modfast,
    const T1 *input0,
    const T2* input1,
    T1 *output) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems) return;

    int outer_idx = outer_stride_modfast.div(index);
    T1 in0_val = input0[index];
    T2 in1_val = input1[outer_idx];
    T1 out_val;

    T2* in0_val_ptr = (T2*)&in0_val;
    T2* out_val_ptr = (T2*)&out_val;

    #pragma unroll N
    for (int i = 0; i < N; i++) {
        out_val_ptr[i] = ppl_arithmetic_scalar<op_type, T2>(in0_val_ptr[i], in1_val);
    }
    output[index] = out_val;
}
#endif//OPT_ARITHM

template<ArithmeticOpType op_type, typename T>
__global__ void ppl_cukernel_arithmetic_one_broadcast_int8(
    const uint64_t num_elems,
    const int outer_stride,
    const int inner_dim,
    const bool first_shorter,
    const T *input0,
    const T* input1,
    T *output,
    float in_scale0 = 0,
    float in_scale1 = 0,
    float out_scale = 0) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems) return;
    int inner_idx = index % inner_dim;
    int outer_idx = index / outer_stride;
    uint64_t calc_index = outer_idx * inner_dim + inner_idx;
    uint64_t offset0 = first_shorter ? calc_index : index;
    uint64_t offset1 = first_shorter ? index : calc_index;
    output[index] = ppl_arithmetic_scalar_int8<op_type, T>(input0[offset0], input1[offset1], in_scale0, in_scale1, out_scale);
}

#ifdef OPT_ARITHM
template<ArithmeticOpType op_type, typename T>
__global__ void ppl_cukernel_arithmetic_one_broadcast_first_shorter_int8_opt(
    const uint64_t num_elems,
    const int outer_stride,
    const int inner_dim,
    const bool first_shorter,
    const T *input0,
    const T* input1,
    T *output,
    DivModFast inner_mod,
    DivModFast stride_mod,
    float in_scale0 = 0,
    float in_scale1 = 0,
    float out_scale = 0) {
    uint64_t index = (blockIdx.x * blockDim.x + threadIdx.x) << 4;
    if (index >= num_elems) return;
    int inner_idx = inner_mod.mod(index);
    int outer_idx = stride_mod.div(index);

    uint64_t calc_index = outer_idx * inner_dim + inner_idx;
    uint64_t offset0 = calc_index;
    uint64_t offset1 = index;

    float4* val0_ptr = (float4*)input0;
    float4* val1_ptr = (float4*)input1;
    float4 val_out;

    float4 val0 = (float4)val0_ptr[offset0 >> 4];
    float4 val1 = (float4)val1_ptr[offset1 >> 4];

    int8_t *val0_ = (int8_t*)&val0;
    int8_t *val1_ = (int8_t*)&val1;
    int8_t *val_out_ = (int8_t*)&val_out;

    #pragma unroll
    for (int i=0; i < 16; i++) {
        val_out_[i] = ppl_arithmetic_scalar_int8<op_type, int8_t>(val0_[i], val1_[i], in_scale0, in_scale1, out_scale);
    }

    float4* output_ptr = (float4*)output;
    output_ptr[index >> 4] = val_out;
}

template<ArithmeticOpType op_type, typename T>
__global__ void ppl_cukernel_arithmetic_one_broadcast_no_first_shorter_int8_opt(
    const uint64_t num_elems,
    const int outer_stride,
    const int inner_dim,
    const bool first_shorter,
    const T *input0,
    const T* input1,
    T *output,
    DivModFast inner_mod,
    DivModFast stride_mod,
    float in_scale0 = 0,
    float in_scale1 = 0,
    float out_scale = 0) {
    uint64_t index = (blockIdx.x * blockDim.x + threadIdx.x) << 4;
    if (index >= num_elems) return;
    int inner_idx = inner_mod.mod(index);
    int outer_idx = stride_mod.div(index);

    uint64_t calc_index = outer_idx * inner_dim + inner_idx;
    uint64_t offset0 = index;
    uint64_t offset1 = calc_index;

    float4* val0_ptr = (float4*)input0;
    float4* val1_ptr = (float4*)input1;
    float4 val_out;

    float4 val0 = (float4)val0_ptr[offset0 >> 4];
    float4 val1 = (float4)val1_ptr[offset1 >> 4];

    int8_t *val0_ = (int8_t*)&val0;
    int8_t *val1_ = (int8_t*)&val1;
    int8_t *val_out_ = (int8_t*)&val_out;

    #pragma unroll
    for (int i=0; i < 16; i++) {
        val_out_[i] = ppl_arithmetic_scalar_int8<op_type, int8_t>(val0_[i], val1_[i], in_scale0, in_scale1, out_scale);
    }

    float4* output_ptr = (float4*)output;
    output_ptr[index >> 4] = val_out;
}
#endif

#ifdef OPT_ARITHM
template<ArithmeticOpType op_type, typename T>
bool do_ppl_arithmetic_xn_opt(uint64_t num_elems, int dim_count, ArithmeticParam param,
    const T* input0, const T* input1, T* output, int grid_size, int block_size, cudaStream_t stream) {
    //Only use xn optimizations so that we can collect a ldg/stg_128, which will reach a maxinum bandwidth
    if (sizeof(T) >= 8) return false;
    uint32_t opt = 16 / sizeof(T);
    if (opt <= 2) return false;
    if (num_elems % (opt*block_size) != 0) return false;
    if (dim_count == 1) return false;

    int idx = dim_count-2;
    //Check the stride_in / stride_out if we can fetch data continuously
    while (idx >= 0) {
        if (param.stride_in0[idx] % opt != 0) {
            return false;
        }
        idx--;
    }
    idx = dim_count - 2;
    while (idx >= 0) {
        if (param.stride_in1[idx] % opt != 0) {
            return false;
        }
        idx--;
    }
    idx = dim_count - 1;
    while (idx >= 0) {
        if (param.stride_out[idx] % opt != 0) {
            return false;
        }
        idx--;
    }

    switch (opt) {
        case 4:
            ppl_cukernel_arithmetic_XN<op_type, T, 4><<<grid_size/4, block_size, 0, stream>>>(num_elems, dim_count, param, input0, input1, output);
            return true;
        case 8:
            ppl_cukernel_arithmetic_XN<op_type, T, 8><<<grid_size/8, block_size, 0, stream>>>(num_elems, dim_count, param, input0, input1, output);
            return true;
        case 16:
            ppl_cukernel_arithmetic_XN<op_type, T, 16><<<grid_size/16, block_size, 0, stream>>>(num_elems, dim_count, param, input0, input1, output);
            return true;
        default:
            return false;
    }

    return false;
}


template<ArithmeticOpType op_type, typename T>
bool do_ppl_arithmetic(uint64_t num_elems, int dim_count, ArithmeticParam param,
    const T* input0, const T* input1, T* output, int grid_size, int block_size, cudaStream_t stream) {
    if (!do_ppl_arithmetic_xn_opt<op_type, T>(num_elems, dim_count, param, input0, input1, output, grid_size, block_size, stream)) {
        DivModMetadata divmodMetaData;
        for(int i = 0; i < dim_count; i++) {
            divmodMetaData.modFast[i] = DivModFast(param.stride_out[i]);
        }
        constexpr int N = 16 / sizeof(T);
        if(((param.lastdim_in0 & (N - 1)) == 0)&&((param.lastdim_in1 & (N - 1)) == 0) && ((param.lastdim_out & (N - 1)) == 0))
        {
            int gridSize = (grid_size + N - 1) / N;
            ppl_cukernel_arithmetic_opt1<op_type,T,float4, 16 / sizeof(T)><<<gridSize, block_size,0,stream>>>
                (num_elems, dim_count, param, input0, input1, output,divmodMetaData);
            return true;
        }
        
        ppl_cukernel_arithmetic_opt<op_type, T, 16 / sizeof(T)><<< (grid_size + N - 1) / N, block_size, 0, stream>>>(num_elems, dim_count, param,
                                                    input0, input1, output, divmodMetaData);
    }
    return true;
}
#endif

template<ArithmeticOpType op_type, typename T>
ppl::common::RetCode PPLCUDAArithMeticForwardImp(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape0,
    const T *input0,
    const ppl::common::TensorShape* input_shape1,
    const T *input1,
    const ppl::common::TensorShape* output_shape,
    T *output) {
    uint64_t num_elems = output_shape->CalcElementsIncludingPadding();
    int dim_count = output_shape->GetDimCount();
    int block_size = 256;
    uint64_t grid_size = (num_elems + block_size - 1) / block_size;
    int axis = 0; bool bidirectional = false;
    int num_broadcast_dims = ppl_get_num_broadcast_dims(input_shape0, input_shape1, axis, bidirectional);
    if (!bidirectional && num_broadcast_dims > 0 && ((input_shape0->GetDimCount() < 2) || (input_shape1->GetDimCount() < 2))) {
        bool first_shorter = false;
        if (input_shape0->CalcElementsIncludingPadding() < input_shape1->CalcElementsIncludingPadding()) {
            first_shorter = true;
        }
        if (input_shape0->CalcElementsIncludingPadding() == 1 || input_shape1->CalcElementsIncludingPadding() == 1) {
#ifdef OPT_ARITHM
            constexpr int N = 16 / sizeof(T);
            const bool use_n_opt = num_elems % N == 0 && grid_size / N > 64;
            if (first_shorter) {
                if (use_n_opt) {
                    grid_size = (grid_size + N - 1) / N;
                    ppl_cukernel_arithmetic_one_scalar_first_shorter_combopt<op_type, T, N><<<grid_size, block_size, 0, stream>>>(
                        num_elems, (const T*)input0, (const T*)input1, (T*)output
                    );
                } else {
                    ppl_cukernel_arithmetic_one_scalar_first_shorter_opt<op_type, T, 1><<<grid_size, block_size, 0, stream>>>(
                        num_elems, (const T*)input0, (const T*)input1, (T*)output
                    );
                }
            } else {
                if (use_n_opt) {
                    grid_size = (grid_size + N - 1) / N;
                    ppl_cukernel_arithmetic_one_scalar_not_first_shorter_combopt<op_type, T, N><<<grid_size, block_size, 0, stream>>>(
                        num_elems, (const T*)input0, (const T*)input1, (T*)output
                    );
                } else {
                    ppl_cukernel_arithmetic_one_scalar_not_first_shorter_opt<op_type, T, 1><<<grid_size, block_size, 0, stream>>>(
                        num_elems, (const T*)input0, (const T*)input1, (T*)output
                    );
                }
            }
#else
            ppl_cukernel_arithmetic_one_scalar<op_type, T><<<grid_size, block_size, 0,
                stream>>>(num_elems, first_shorter, (const T*)input0, (const T*)input1, (T*)output);
#endif
        } else {
            int inner_dim = first_shorter ? input_shape0->GetDim(0) : input_shape1->GetDim(0);
            int channel_dim = 1;
            //TODO(WJF): confused with "channel_dim": according to onnx-broadcast rule, input A(m,n) and B(n), output C(m, n)
            // so there is no need to divide by channel_dim when caculating shorter_index
            // if (output_shape->GetDataFormat() != ppl::common::DATAFORMAT_NDARRAY)
                // channel_dim = output_shape->GetDim(1) + output_shape->GetPadding0(1) + output_shape->GetPadding1(1);
#ifdef OPT_ARITHM
#define CALL_SHARED_X8 \
        if (first_shorter) {\
            if (channel_dim == 1) {\
                ppl_cukernel_arithmetic_one_dimension_s_t2c1x8<op_type, T, true><<<grid_size,block_size, 0, \
                    stream>>>(num_elems, inner_dim, DivModFast(inner_dim), (const T*)input0, (const T*)input1, (T*)output); \
            } else {\
                ppl_cukernel_arithmetic_one_dimension_s_t2cnx8<op_type, T, true><<<grid_size,block_size, 0, \
                    stream>>>(num_elems, inner_dim, DivModFast(inner_dim), DivModFast(channel_dim), (const T*)input0, (const T*)input1, (T*)output); \
            } \
        } else { \
            if (channel_dim == 1) {\
                ppl_cukernel_arithmetic_one_dimension_s_t2c1x8<op_type, T, false><<<grid_size,block_size, 0, \
                    stream>>>(num_elems, inner_dim, DivModFast(inner_dim), (const T*)input0, (const T*)input1, (T*)output); \
            } else {\
                ppl_cukernel_arithmetic_one_dimension_s_t2cnx8<op_type, T, false><<<grid_size,block_size, 0, \
                    stream>>>(num_elems, inner_dim, DivModFast(inner_dim), DivModFast(channel_dim), (const T*)input0, (const T*)input1, (T*)output); \
            } \
        }

        #define CALL_PLAIN_X8 \
        if (first_shorter) {\
            if (channel_dim == 1) {\
                ppl_cukernel_arithmetic_one_dimension_t2c1x8<op_type, T, true><<<grid_size,block_size, 0, \
                    stream>>>(num_elems, inner_dim, DivModFast(inner_dim), (const T*)input0, (const T*)input1, (T*)output); \
            } else {\
                ppl_cukernel_arithmetic_one_dimension_t2cnx8<op_type, T, true><<<grid_size,block_size, 0, \
                    stream>>>(num_elems, inner_dim, DivModFast(inner_dim), DivModFast(channel_dim), (const T*)input0, (const T*)input1, (T*)output); \
            } \
        } else { \
            if (channel_dim == 1) {\
                ppl_cukernel_arithmetic_one_dimension_t2c1x8<op_type, T, false><<<grid_size,block_size, 0, \
                    stream>>>(num_elems, inner_dim, DivModFast(inner_dim), (const T*)input0, (const T*)input1, (T*)output); \
            } else {\
                ppl_cukernel_arithmetic_one_dimension_t2cnx8<op_type, T, false><<<grid_size,block_size, 0, \
                    stream>>>(num_elems, inner_dim, DivModFast(inner_dim), DivModFast(channel_dim), (const T*)input0, (const T*)input1, (T*)output); \
            } \
        }

        if (sizeof(T) == 2) {
            int maybe_shared_size = inner_dim * channel_dim * sizeof(T);
            bool use_shared_buffer = maybe_shared_size < 8192;
            bool use_block_512 = maybe_shared_size >= 4096;
            bool can_use_x8 = use_block_512 ? num_elems % (512 * 8) == 0 : num_elems % (256 * 8) == 0;
            if (can_use_x8) {
                grid_size = use_block_512 ? num_elems/512/8 : num_elems/256/8;
                block_size = use_block_512 ? 512 : 256;
                if (use_shared_buffer) {
                    CALL_SHARED_X8
                } else {
                    CALL_PLAIN_X8
                }
            } else if ((inner_dim % (8 / sizeof(T))) == 0) {
                constexpr int N = 8 / sizeof(T);
                grid_size = (grid_size + N - 1) / N;
                ppl_cukernel_arithmetic_one_dimension_opt<op_type, int64_t, T, N><<<grid_size, block_size, 0,
                    stream>>>(num_elems, inner_dim, channel_dim, first_shorter,
                    first_shorter ? (const T*)input1 : (const T*)input0,
                    first_shorter ? (const T*)input0 : (const T*)input1,
                    (T*)output);
            } else {
                ppl_cukernel_arithmetic_one_dimension<op_type, T><<<grid_size,block_size, 0,
                    stream>>>(num_elems, inner_dim, channel_dim, first_shorter, (const T*)input0, (const T*)input1, (T*)output);
            }
        } else {
            ppl_cukernel_arithmetic_one_dimension<op_type, T><<<grid_size,block_size, 0,
                    stream>>>(num_elems, inner_dim, channel_dim, first_shorter, (const T*)input0, (const T*)input1, (T*)output);
        }
#else
            ppl_cukernel_arithmetic_one_dimension<op_type, T><<<grid_size, block_size, 0,
                stream>>>(num_elems, inner_dim, channel_dim, first_shorter, (const T*)input0, (const T*)input1, (T*)output);
#endif
        }
    } else if (num_broadcast_dims == 0) {
        if ((std::is_same<T, half>::value) && num_elems % 8 == 0) {
            grid_size = ((num_elems >> 3) + block_size - 1) / block_size;
            ppl_cukernel_arithmetic_nobroadcast_float4<op_type, float4><<<grid_size, block_size, 0,
                stream>>>(num_elems >> 3, (const float4*)input0, (const float4*)input1, (float4*)output);
            return ppl::common::RC_SUCCESS;
        }
#ifdef OPT_ARITHM
        if (num_elems % 8 == 0 && grid_size / 8  >= 256 && sizeof(T) < 4) {
            ppl_cukernel_arithmetic_nobroadcast_opt<op_type, T, 8><<<(grid_size+7)/8, block_size, 0,
                stream>>>(num_elems, (const T*)input0, (const T*)input1, (T*)output);
        } else if (num_elems % 4 == 0 && grid_size / 4 > 512 && sizeof(T) < 8) {
            ppl_cukernel_arithmetic_nobroadcast_opt<op_type, T, 4><<<(grid_size+3)/4, block_size, 0,
                stream>>>(num_elems, (const T*)input0, (const T*)input1, (T*)output);
        } else if (num_elems % 2 == 0 && grid_size / 2 > 1024) {
            ppl_cukernel_arithmetic_nobroadcast_opt<op_type, T, 2><<<(grid_size+1)/2, block_size, 0,
                stream>>>(num_elems, (const T*)input0, (const T*)input1, (T*)output);
        } else {
            ppl_cukernel_arithmetic_nobroadcast<op_type, T><<<grid_size, block_size, 0,
            stream>>>(num_elems, (const T*)input0, (const T*)input1, (T*)output);
        }
#else
        ppl_cukernel_arithmetic_nobroadcast<op_type, T><<<grid_size, block_size, 0,
            stream>>>(num_elems, (const T*)input0, (const T*)input1, (T*)output);
#endif
    } else {
        ArithmeticParam param;
        int packed_channel = 1;
        if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8) {
            if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
                // one broadcast (or last dimensions broadcast)
                if (ppl_feature_broadcast(input_shape0, input_shape1, &axis)) {
                    int inner_dim = output_shape->CalcElementsToDimensionIncludingPadding(axis) *
                                    output_shape->CalcElementsFromDimensionIncludingPadding(axis - 1) /
                                    output_shape->CalcElementsIncludingPadding();
                    int outer_stride =  output_shape->CalcElementsFromDimensionIncludingPadding(1);
                    bool first_shorter = false;
                    if (input_shape0->GetRealDimCount() == input_shape1->GetRealDimCount() &&
                        input_shape0->GetDim(axis) < input_shape1->GetDim(axis)) {
                        first_shorter = true;
                    }
                    if (input_shape0->CalcElementsExcludingPadding() < input_shape1->CalcElementsExcludingPadding())  {
                        first_shorter = true;
                    }
                    // ppl_cukernel_arithmetic_one_broadcast<op_type, half><<<grid_size, block_size, 0,
                    //     stream>>>(num_elems, outer_stride, inner_dim, first_shorter, (const half*)input0, (const half*)input1, (half*)output);
                    DivModFast broad_mod_fast(inner_dim);
                    DivModFast broad_div_fast(outer_stride);
                    if(first_shorter) {
                        grid_size = ((num_elems >> 3 ) + block_size - 1) / block_size;
                        ppl_cukernel_arithmetic_one_broadcast_first_shorter_fp16<op_type, T><<<grid_size, block_size, 0,
                            stream>>>(num_elems, outer_stride, inner_dim, broad_mod_fast, broad_div_fast, (const T*)input0, (const T*)input1, (T*)output);
                    }else{
                        grid_size = ((num_elems >> 3 ) + block_size - 1) / block_size;
                        ppl_cukernel_arithmetic_one_broadcast_not_first_shorter_fp16<op_type, T><<<grid_size, block_size, 0,
                            stream>>>(num_elems, outer_stride, inner_dim, broad_mod_fast, broad_div_fast,(const T*)input0, (const T*)input1, (T*)output);
                    }
                // normal case, deal one half once
                } else if ((input_shape0->GetDim(1) & 0x7) || (input_shape1->GetDim(1) & 0x7)) {
                    int channel_shift  = 0;
                    int packed_channel = 8;
                    int suppled_channel = 8;
                    uint64_t grid_size = ((num_elems >> channel_shift) + block_size - 1) / block_size;
                    ppl_arithmetic_prepare_strides_nhwc(input_shape0, input_shape1, output_shape, packed_channel, param.stride_in0, param.stride_in1, param.stride_out, suppled_channel);
#ifdef OPT_ARITHM
                    if(condition_arithmetic(param,input_shape0, input_shape1)){
                        grid_size = ((num_elems >> channel_shift) + (block_size*8) - 1) / (block_size*8);
                        ppl_cukernel_arithmetic_fp16_opt_sc<op_type><<<grid_size,block_size,0, stream>>>(num_elems >> channel_shift, dim_count, param, (const half*)input0, (const half*)input1, (half*)output);
                    }else{
                        ppl_cukernel_arithmetic_fp16<op_type, half, half><<<grid_size, block_size,0, stream>>>(num_elems >> channel_shift, dim_count, param, (const half*)input0, (const half*)input1, (half*)output);
                    }
#else//!OPT_ARITHM
                    ppl_cukernel_arithmetic_fp16<op_type, half, half><<<grid_size, block_size,0, stream>>>(num_elems >> channel_shift, dim_count, param, (const half*)input0, (const half*)input1, (half*)output);
#endif//OPT_ARITHM
                } else { // deal 8 half once
                    int channel_shift  = 3;
                    int packed_channel = 8;
#ifdef OPT_ARITHM
                    int64_t axis_dim = 0;
                    for (int it = 0; it < dim_count; ++it) {
                        if (input_shape0->GetDim(it) == input_shape1->GetDim(it))
                            axis_dim = it;
                    }
                    int32_t axis_width = output_shape->GetDim(axis_dim);
                    if (num_broadcast_dims == 3 && axis_dim == 1 && axis_width < MAX_BLOCK_DIM_YZ) {
                        DivModFast axis_width_mod_fast = DivModFast(axis_width >> channel_shift);
                        grid_size = ((num_elems >> channel_shift) + block_size - 1) / block_size;
                        ppl_cukernel_arithmetic_fp16_nhwc8_opt<op_type, 8, half, float4><<<grid_size, block_size, 0, stream>>>(num_elems >> channel_shift,
                                                                        (const float4 *)input0, (const float4 *)input1, (float4 *)output,
                                                                        axis_width_mod_fast);
                    } else {
#endif//OPT_ARITHM
                    ppl_arithmetic_prepare_strides_nhwc(input_shape0, input_shape1, output_shape, packed_channel, param.stride_in0, param.stride_in1, param.stride_out);
                    ppl_cukernel_arithmetic_fp16<op_type, half8_, float4><<<grid_size, block_size,0, stream>>>(num_elems >> channel_shift, dim_count, param, (const half8_ *)input0, (const half8_ *)input1, (half8_ *)output);
#ifdef OPT_ARITHM
                    }
#endif//OPT_ARITHM
                }
                return ppl::common::RC_SUCCESS;
            }
            packed_channel = 8;
            ppl_arithmetic_prepare_strides_nhwc(input_shape0, input_shape1, output_shape, packed_channel,
                param.stride_in0, param.stride_in1, param.stride_out, packed_channel);
            ppl_cukernel_arithmetic<op_type, T><<<grid_size, block_size, 0,
                stream>>>(num_elems, dim_count, param, (const T*)input0, (const T*)input1, (T*)output);
        } else if(output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC){
            if(ppl_feature_broadcast(input_shape0, input_shape1, &axis)) {
                int inner_dim = output_shape->CalcElementsToDimensionIncludingPadding(axis) *
                                output_shape->CalcElementsFromDimensionIncludingPadding(axis - 1) /
                                output_shape->CalcElementsIncludingPadding();
                int outer_stride =  output_shape->CalcElementsFromDimensionIncludingPadding(1);
                bool first_shorter = false;
                if (input_shape0->GetRealDimCount() == input_shape1->GetRealDimCount() &&
                    input_shape0->GetDim(axis) < input_shape1->GetDim(axis)) {
                    first_shorter = true;
                }
                if (input_shape0->CalcElementsExcludingPadding() < input_shape1->CalcElementsExcludingPadding())  {
                    first_shorter = true;
                }
                ppl_cukernel_arithmetic_one_broadcast<op_type, T><<<grid_size, block_size, 0,
                stream>>>(num_elems, outer_stride, inner_dim, first_shorter, (const T*)input0, (const T*)input1, (T*)output);
            } else {
                ppl_arithmetic_prepare_strides_nhwc(input_shape0, input_shape1, output_shape, packed_channel,
                                                    param.stride_in0, param.stride_in1, param.stride_out);
                ppl_cukernel_arithmetic<op_type, T><<<grid_size, block_size, 0, stream>>>(num_elems, dim_count, param, (const T*)input0, (const T*)input1, (T*)output);
            }
        } else if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
            bool first_shorter = false;
            if (input_shape0->GetRealDimCount() == input_shape1->GetRealDimCount() &&
                input_shape0->GetDim(axis) < input_shape1->GetDim(axis)) {
                first_shorter = true;
            }
            if (input_shape0->CalcElementsExcludingPadding() < input_shape1->CalcElementsExcludingPadding())  {
                first_shorter = true;
            }
            if (num_broadcast_dims == 1) {
                int inner_dim = 1;
                for(int it = axis + 1; it < dim_count; inner_dim *= output_shape->GetDim(it), ++it);
                int outer_stride = inner_dim * output_shape->GetDim(axis);
#ifdef OPT_ARITHM
                if(inner_dim == 1 && first_shorter == 0)
                {
                    if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
                        if (outer_stride % 8 == 0) {
                            const int n = 8;
                            grid_size = ((num_elems / n) + block_size - 1) / block_size;
                            DivModFast outer_stride_mod_fast = DivModFast(outer_stride / n);
                            ppl_cukernel_arithmetic_one_broadcast_opt1<op_type, n><<<grid_size, block_size, 0,
                                stream>>>(num_elems / n, outer_stride_mod_fast, (const float4*)input0, (const half*)input1, (float4*)output);
                        } else if (outer_stride % 4 == 0) {
                            const int n = 4;
                            grid_size = ((num_elems / n) + block_size - 1) / block_size;
                            DivModFast outer_stride_mod_fast = DivModFast(outer_stride / n);
                            ppl_cukernel_arithmetic_one_broadcast_opt1<op_type, n><<<grid_size, block_size, 0,
                                stream>>>(num_elems / n, outer_stride_mod_fast, (const int64_t*)input0, (const half*)input1, (int64_t*)output);
                        } else if (outer_stride % 2 == 0) {
                            const int n = 2;
                            grid_size = ((num_elems / n) + block_size - 1) / block_size;
                            DivModFast outer_stride_mod_fast = DivModFast(outer_stride / n);
                            ppl_cukernel_arithmetic_one_broadcast_opt1<op_type, n><<<grid_size, block_size, 0,
                                stream>>>(num_elems / n, outer_stride_mod_fast, (const float*)input0, (const half*)input1, (float*)output);
                        } else {
                            DivModFast outer_stride_mod_fast = DivModFast(outer_stride);
                            ppl_cukernel_arithmetic_one_broadcast_opt1<op_type, 1><<<grid_size, block_size, 0,
                                stream>>>(num_elems, outer_stride_mod_fast, (const T*)input0, (const T*)input1, (T*)output);
                        }
                    } else {
                        DivModFast outer_stride_mod_fast = DivModFast(outer_stride);
                        ppl_cukernel_arithmetic_one_broadcast_opt1<op_type, 1><<<grid_size, block_size, 0,
                            stream>>>(num_elems, outer_stride_mod_fast, (const T*)input0, (const T*)input1, (T*)output);
                    }
                }else{
                    constexpr int N = 16 / sizeof(T);
                    const bool use_n_opt = (inner_dim % N == 0) && (grid_size / N > 64);
                    if (use_n_opt && op_type != Arithmetic_Sub) {
                        grid_size = (grid_size + N - 1) / N;
                        ppl_cukernel_arithmetic_one_broadcast_float4_opt<op_type, T, N><<<grid_size, block_size, 0,
                            stream>>>(num_elems, outer_stride, inner_dim, first_shorter,
                            first_shorter ? (const T*)input1 : (const T*)input0,
                            first_shorter ? (const T*)input0 : (const T*)input1,
                            (T*)output);
                    } else {
                        ppl_cukernel_arithmetic_one_broadcast<op_type, T><<<grid_size, block_size, 0,
                            stream>>>(num_elems, outer_stride, inner_dim, first_shorter, (const T*)input0, (const T*)input1, (T*)output);
                    }
                }
#else//!OPT_ARITHM
                ppl_cukernel_arithmetic_one_broadcast<op_type, T><<<grid_size, block_size, 0,
                    stream>>>(num_elems, outer_stride, inner_dim, first_shorter, (const T*)input0, (const T*)input1, (T*)output);
#endif//OPT_ARITHM
                return ppl::common::RC_SUCCESS;
            } else if (!bidirectional && ppl_can_one_not_broadcast(input_shape0, input_shape1, axis)) {
                int axis_lgt = first_shorter ? input_shape0->CalcElementsExcludingPadding() : input_shape1->CalcElementsExcludingPadding();
                int inner_dim = first_shorter ? input_shape1->CalcElementsFromDimensionIncludingPadding(axis + 1) : input_shape0->CalcElementsFromDimensionIncludingPadding(axis + 1);
#ifdef OPT_ARITHM
                constexpr int N = 16 / sizeof(T);
                const bool use_n_opt = (inner_dim % N == 0) && (grid_size / N > 64);
                if (use_n_opt) {
                    grid_size = (grid_size + N - 1) / N;
                    ppl_cukernel_arithmetic_one_not_broadcast_float4_opt<op_type, T, N><<<grid_size, block_size, 0,
                        stream>>>(num_elems, axis_lgt, inner_dim, first_shorter,
                        first_shorter ? (const T*)input1 : (const T*)input0,
                        first_shorter ? (const T*)input0 : (const T*)input1,
                        (T*)output);
                } else {
                    ppl_cukernel_arithmetic_one_not_broadcast<op_type, T><<<grid_size, block_size, 0,
                    stream>>>(num_elems, axis_lgt, inner_dim, first_shorter, (const T*)input0, (const T*)input1, (T*)output);
                }
#else//!OPT_ARITHM
                ppl_cukernel_arithmetic_one_not_broadcast<op_type, T><<<grid_size, block_size, 0,
                    stream>>>(num_elems, axis_lgt, inner_dim, first_shorter, (const T*)input0, (const T*)input1, (T*)output);
#endif//OPT_ARITHM
                return ppl::common::RC_SUCCESS;
            }
            ppl_arithmetic_prepare_strides(input_shape0, input_shape1,
                output_shape, packed_channel, param.stride_in0, param.stride_in1, param.stride_out);
#ifdef OPT_ARITHM
            param.lastdim_in0 = input_shape0->GetDim(input_shape0->GetDimCount() - 1);
            param.lastdim_in1 = input_shape1->GetDim(input_shape1->GetDimCount() - 1);
            param.lastdim_out = output_shape->GetDim(output_shape->GetDimCount() - 1);
            do_ppl_arithmetic<op_type, T>(num_elems, dim_count, param, (const T*)input0, (const T*)input1, (T*)output, grid_size, block_size, stream);
#else
            ppl_cukernel_arithmetic<op_type, T><<<grid_size, block_size, 0,
                    stream>>>(num_elems, dim_count, param, (const T*)input0, (const T*)input1, (T*)output);
#endif
        } else {
            return ppl::common::RC_UNSUPPORTED;
        }
    }

    return ppl::common::RC_SUCCESS;
}

template<ArithmeticOpType op_type, typename T>
ppl::common::RetCode PPLCUDAArithMeticForwardImpInt8(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape0,
    const T *input0,
    const ppl::common::TensorShape* input_shape1,
    const T *input1,
    const ppl::common::TensorShape* output_shape,
    T *output,
    float in_scale0 = 0,
    float in_scale1 = 0,
    float out_scale = 0) {
    uint64_t num_elems = output_shape->CalcElementsIncludingPadding();
    int dim_count = output_shape->GetDimCount();
    int block_size = 256;
    uint64_t grid_size = (num_elems + block_size - 1) / block_size;
    int axis = 0; bool bidirectional = false;
    int num_broadcast_dims = ppl_get_num_broadcast_dims(input_shape0, input_shape1, axis, bidirectional);
    if (!bidirectional && ((input_shape0->GetDimCount() < 2) || (input_shape1->GetDimCount() < 2))) {
        bool first_shorter = false;
        if (input_shape0->CalcElementsIncludingPadding() < input_shape1->CalcElementsIncludingPadding()) {
            first_shorter = true;
        }
        if (input_shape0->CalcElementsIncludingPadding() == 1 || input_shape1->CalcElementsIncludingPadding() == 1) {
#ifdef OPT_ARITHM
            if (out_scale == 0) out_scale = 1;
            bool use_16_opt = num_elems % 16 == 0 && grid_size / 16 >= 128;
            if (first_shorter) {
                if (use_16_opt) {
                    if (op_type == Arithmetic_Mul) {
                        ppl_cukernel_arithmetic_one_scalar_int8_first_shorter_opt<op_type, T, 16><<<grid_size, block_size, 0,
                            stream>>>(num_elems, (const T*)input0, (const T*)input1, (T*)output, in_scale0*in_scale1/out_scale, 1, 1);
                    } else if (op_type == Arithmetic_Div) {
                        ppl_cukernel_arithmetic_one_scalar_int8_first_shorter_opt<op_type, T, 16><<<grid_size, block_size, 0,
                            stream>>>(num_elems, (const T*)input0, (const T*)input1, (T*)output, in_scale0/in_scale1/out_scale, 1, 1);
                    } else if (op_type == Arithmetic_PRelu) {
                        ppl_cukernel_arithmetic_one_scalar_int8_first_shorter_opt<op_type, T, 16><<<grid_size, block_size, 0,
                            stream>>>(num_elems, (const T*)input0, (const T*)input1, (T*)output, in_scale0, in_scale1, out_scale);
                    } else {
                        ppl_cukernel_arithmetic_one_scalar_int8_first_shorter_opt<op_type, T, 16><<<grid_size, block_size, 0,
                            stream>>>(num_elems, (const T*)input0, (const T*)input1, (T*)output, in_scale0/out_scale, in_scale1/out_scale, 1);
                    }
                } else {
                    if (op_type == Arithmetic_Mul) {
                        ppl_cukernel_arithmetic_one_scalar_int8_first_shorter_opt<op_type, T, 1><<<grid_size, block_size, 0,
                            stream>>>(num_elems, (const T*)input0, (const T*)input1, (T*)output, in_scale0*in_scale1/out_scale, 1, 1);
                    } else if (op_type == Arithmetic_Div) {
                        ppl_cukernel_arithmetic_one_scalar_int8_first_shorter_opt<op_type, T, 1><<<grid_size, block_size, 0,
                            stream>>>(num_elems, (const T*)input0, (const T*)input1, (T*)output, in_scale0/in_scale1/out_scale, 1, 1);
                    } else if (op_type == Arithmetic_PRelu) {
                        ppl_cukernel_arithmetic_one_scalar_int8_first_shorter_opt<op_type, T, 1><<<grid_size, block_size, 0,
                            stream>>>(num_elems, (const T*)input0, (const T*)input1, (T*)output, in_scale0, in_scale1, out_scale);
                    } else {
                        ppl_cukernel_arithmetic_one_scalar_int8_first_shorter_opt<op_type, T, 1><<<grid_size, block_size, 0,
                            stream>>>(num_elems, (const T*)input0, (const T*)input1, (T*)output, in_scale0/out_scale, in_scale1/out_scale, 1);
                    }
                }
            } else {
                if (use_16_opt) {
                    if (op_type == Arithmetic_Mul)
                        ppl_cukernel_arithmetic_one_scalar_int8_not_first_shorter_opt<op_type, T, 16><<<grid_size, block_size, 0,
                            stream>>>(num_elems, (const T*)input0, (const T*)input1, (T*)output, in_scale0*in_scale1/out_scale, 1, 1);
                    else if (op_type == Arithmetic_Div)
                        ppl_cukernel_arithmetic_one_scalar_int8_not_first_shorter_opt<op_type, T, 16><<<grid_size, block_size, 0,
                            stream>>>(num_elems, (const T*)input0, (const T*)input1, (T*)output, in_scale0/in_scale1/out_scale, 1, 1);
                    else if (op_type == Arithmetic_PRelu)
                        ppl_cukernel_arithmetic_one_scalar_int8_not_first_shorter_opt<op_type, T, 16><<<grid_size, block_size, 0,
                            stream>>>(num_elems, (const T*)input0, (const T*)input1, (T*)output, in_scale0, in_scale1, out_scale);
                    else
                        ppl_cukernel_arithmetic_one_scalar_int8_not_first_shorter_opt<op_type, T, 16><<<grid_size, block_size, 0,
                            stream>>>(num_elems, (const T*)input0, (const T*)input1, (T*)output, in_scale0/out_scale, in_scale1/out_scale, 1);
                } else {
                    if (op_type == Arithmetic_Mul)
                        ppl_cukernel_arithmetic_one_scalar_int8_not_first_shorter_opt<op_type, T, 1><<<grid_size, block_size, 0,
                            stream>>>(num_elems, (const T*)input0, (const T*)input1, (T*)output, in_scale0*in_scale1/out_scale, 1, 1);
                    else if (op_type == Arithmetic_Div)
                        ppl_cukernel_arithmetic_one_scalar_int8_not_first_shorter_opt<op_type, T, 1><<<grid_size, block_size, 0,
                            stream>>>(num_elems, (const T*)input0, (const T*)input1, (T*)output, in_scale0/in_scale1/out_scale, 1, 1);
                    else if (op_type == Arithmetic_PRelu)
                        ppl_cukernel_arithmetic_one_scalar_int8_not_first_shorter_opt<op_type, T, 1><<<grid_size, block_size, 0,
                            stream>>>(num_elems, (const T*)input0, (const T*)input1, (T*)output, in_scale0, in_scale1, out_scale);
                    else
                        ppl_cukernel_arithmetic_one_scalar_int8_not_first_shorter_opt<op_type, T, 1><<<grid_size, block_size, 0,
                            stream>>>(num_elems, (const T*)input0, (const T*)input1, (T*)output, in_scale0/out_scale, in_scale1/out_scale, 1);
                }
            }
#else
            ppl_cukernel_arithmetic_one_scalar_int8<op_type, T><<<grid_size, block_size, 0,
                stream>>>(num_elems, first_shorter, (const T*)input0, (const T*)input1, (T*)output, in_scale0, in_scale1, out_scale);
#endif
        } else {
            int inner_dim = first_shorter ? input_shape0->GetDim(0) : input_shape1->GetDim(0);
#ifdef OPT_ARITHM
            if (out_scale == 0) out_scale = 1;
            if (op_type == Arithmetic_Mul) {
                ppl_cukernel_arithmetic_one_dimension_int8<op_type, T><<<grid_size, block_size, 0,
                    stream>>>(num_elems, inner_dim, first_shorter, (const T*)input0, (const T*)input1, (T*)output, in_scale0*in_scale1/out_scale, 1, 1);
            } else if (op_type == Arithmetic_Div) {
                ppl_cukernel_arithmetic_one_dimension_int8<op_type, T><<<grid_size, block_size, 0,
                    stream>>>(num_elems, inner_dim, first_shorter, (const T*)input0, (const T*)input1, (T*)output, in_scale0/in_scale1/out_scale, 1, 1);
            } else if (op_type == Arithmetic_PRelu) {
                ppl_cukernel_arithmetic_one_dimension_int8<op_type, T><<<grid_size, block_size, 0,
                    stream>>>(num_elems, inner_dim, first_shorter, (const T*)input0, (const T*)input1, (T*)output, in_scale0, in_scale1, out_scale);
            } else {
                ppl_cukernel_arithmetic_one_dimension_int8<op_type, T><<<grid_size, block_size, 0,
                    stream>>>(num_elems, inner_dim, first_shorter, (const T*)input0, (const T*)input1, (T*)output, in_scale0/out_scale, in_scale1/out_scale, 1);
            }
#else
            ppl_cukernel_arithmetic_one_dimension_int8<op_type, T><<<grid_size, block_size, 0,
                stream>>>(num_elems, inner_dim, first_shorter, (const T*)input0, (const T*)input1, (T*)output, in_scale0, in_scale1, out_scale);
#endif
        }
    } else if (num_broadcast_dims == 0) {
#ifdef OPT_ARITHM
        if (out_scale == 0) out_scale = 1;
        if (num_elems % 16 == 0 && grid_size/16 >= 128) {
            bool same_scale = (in_scale0 == in_scale1) && (in_scale0 == out_scale);
            if (op_type == Arithmetic_Mul) {
                ppl_cukernel_arithmetic_nobroadcast_int8_opt_16<op_type, T><<<(grid_size+15)/16, block_size, 0,
                                        stream>>>(num_elems, (const T*)input0, (const T*)input1, (T*)output, in_scale0*in_scale1/out_scale, 1, 1);
            } else if (op_type == Arithmetic_Div) {
                ppl_cukernel_arithmetic_nobroadcast_int8_opt_16<op_type, T><<<(grid_size+15)/16, block_size, 0,
                                        stream>>>(num_elems, (const T*)input0, (const T*)input1, (T*)output, in_scale0/in_scale1/out_scale, 1, 1);
            } else if (op_type == Arithmetic_PRelu) {
                ppl_cukernel_arithmetic_nobroadcast_int8_opt_16<op_type, T><<<(grid_size+15)/16, block_size, 0,
                                        stream>>>(num_elems, (const T*)input0, (const T*)input1, (T*)output, in_scale0, in_scale1, out_scale);
            } else if (op_type == Arithmetic_Add || op_type == Arithmetic_Sub) {
                if (same_scale) {
                    grid_size = ((num_elems >> 4) + block_size - 1) / block_size;
                    ppl_cukernel_arithmetic_nobroadcast_int8_opt_16_no_scale<op_type, float4><<<grid_size, block_size, 0,
                                            stream>>>(num_elems >> 4, (const float4*)input0, (const float4*)input1, (float4*)output);
                } else {
                    ppl_cukernel_arithmetic_nobroadcast_int8_opt_16<op_type, T><<<(grid_size+15)/16, block_size, 0,
                                        stream>>>(num_elems, (const T*)input0, (const T*)input1, (T*)output, in_scale0/out_scale, in_scale1/out_scale, 1);
                }
            } else {
                ppl_cukernel_arithmetic_nobroadcast_int8_opt_16<op_type, T><<<(grid_size+15)/16, block_size, 0,
                                        stream>>>(num_elems, (const T*)input0, (const T*)input1, (T*)output, in_scale0/out_scale, in_scale1/out_scale, 1);
            }
        } else {
            if (op_type == Arithmetic_Mul) {
                ppl_cukernel_arithmetic_nobroadcast_int8<op_type, T><<<grid_size, block_size, 0,
                                        stream>>>(num_elems, (const T*)input0, (const T*)input1, (T*)output, in_scale0*in_scale1/out_scale, 1, 1);
            } else if (op_type == Arithmetic_Div) {
                ppl_cukernel_arithmetic_nobroadcast_int8<op_type, T><<<grid_size, block_size, 0,
                                        stream>>>(num_elems, (const T*)input0, (const T*)input1, (T*)output, in_scale0/in_scale1/out_scale, 1, 1);
            } else if (op_type == Arithmetic_PRelu) {
                ppl_cukernel_arithmetic_nobroadcast_int8<op_type, T><<<grid_size, block_size, 0,
                                        stream>>>(num_elems, (const T*)input0, (const T*)input1, (T*)output, in_scale0, in_scale1, out_scale);
            }
            else
                ppl_cukernel_arithmetic_nobroadcast_int8<op_type, T><<<grid_size, block_size, 0,
                                        stream>>>(num_elems, (const T*)input0, (const T*)input1, (T*)output, in_scale0/out_scale, in_scale1/out_scale, 1);
        }
#else
        ppl_cukernel_arithmetic_nobroadcast_int8<op_type, T><<<grid_size, block_size, 0,
            stream>>>(num_elems, (const T*)input0, (const T*)input1, (T*)output, in_scale0, in_scale1, out_scale);
#endif
    } else {
        ArithmeticParam param;
        int packed_channel = 1;
        if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 ||
            output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16 ||
            output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC) {
                if(ppl_feature_broadcast(input_shape0, input_shape1, &axis)) {
                    int inner_dim = output_shape->CalcElementsToDimensionIncludingPadding(axis) *
                                    output_shape->CalcElementsFromDimensionIncludingPadding(axis - 1) /
                                    output_shape->CalcElementsIncludingPadding();
                    int outer_stride =  output_shape->CalcElementsFromDimensionIncludingPadding(1);
                    bool first_shorter = false;
                    if (input_shape0->GetRealDimCount() == input_shape1->GetRealDimCount() &&
                        input_shape0->GetDim(axis) < input_shape1->GetDim(axis)) {
                        first_shorter = true;
                    }
                    if (input_shape0->CalcElementsExcludingPadding() < input_shape1->CalcElementsExcludingPadding())  {
                        first_shorter = true;
                    }
#ifdef OPT_ARITHM
                    if (out_scale == 0) out_scale = 1;
                    if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16) {
                        grid_size = (num_elems + (block_size << 4) - 1) / (block_size << 4);
                        if (op_type == Arithmetic_Mul) {
                            in_scale0 = in_scale0 * in_scale1 / out_scale;
                            in_scale1 = 1.f;
                            out_scale = 1.f;
                        } else if (op_type == Arithmetic_Div) {
                            in_scale0 = in_scale0 / in_scale1 / out_scale;
                            in_scale1 = 1.f;
                            out_scale = 1.f;
                        }else if (op_type == Arithmetic_PRelu){

                        } else {
                            in_scale0 = in_scale0 / out_scale;
                            in_scale1 = in_scale1 / out_scale;
                            out_scale = 1.f;
                        }
                        if (first_shorter) {
                            ppl_cukernel_arithmetic_one_broadcast_first_shorter_int8_opt<op_type, T><<<grid_size, block_size, 0,
                                    stream>>>(num_elems, outer_stride, inner_dim, first_shorter,
                                            (const T*)input0, (const T*)input1, (T*)output,
                                            DivModFast(inner_dim), DivModFast(outer_stride),
                                            in_scale0, in_scale1, out_scale);
                        } else {
                            ppl_cukernel_arithmetic_one_broadcast_no_first_shorter_int8_opt<op_type, T><<<grid_size, block_size, 0,
                                    stream>>>(num_elems, outer_stride, inner_dim, first_shorter,
                                            (const T*)input0, (const T*)input1, (T*)output,
                                            DivModFast(inner_dim), DivModFast(outer_stride),
                                            in_scale0, in_scale1, out_scale);
                        }
                    } else {
                        if (op_type == Arithmetic_Mul) {
                            ppl_cukernel_arithmetic_one_broadcast_int8<op_type, T><<<grid_size, block_size, 0,
                                    stream>>>(num_elems, outer_stride, inner_dim, first_shorter,
                                            (const T*)input0, (const T*)input1, (T*)output, in_scale0*in_scale1/out_scale, 1, 1);
                        } else if (op_type == Arithmetic_Div) {
                            ppl_cukernel_arithmetic_one_broadcast_int8<op_type, T><<<grid_size, block_size, 0,
                                    stream>>>(num_elems, outer_stride, inner_dim, first_shorter,
                                            (const T*)input0, (const T*)input1, (T*)output, in_scale0/in_scale1/out_scale, 1, 1);
                        } else if (op_type == Arithmetic_PRelu){
                            ppl_cukernel_arithmetic_one_broadcast_int8<op_type, T><<<grid_size, block_size, 0,
                                    stream>>>(num_elems, outer_stride, inner_dim, first_shorter,
                                            (const T*)input0, (const T*)input1, (T*)output, in_scale0, in_scale1, out_scale);
                        } else {
                            ppl_cukernel_arithmetic_one_broadcast_int8<op_type, T><<<grid_size, block_size, 0,
                                    stream>>>(num_elems, outer_stride, inner_dim, first_shorter,
                                            (const T*)input0, (const T*)input1, (T*)output, in_scale0/out_scale, in_scale1/out_scale, 1);
                        }
                    }
#else
                    ppl_cukernel_arithmetic_one_broadcast_int8<op_type, T><<<grid_size, block_size, 0,
                    stream>>>(num_elems, outer_stride, inner_dim, first_shorter, (const T*)input0, (const T*)input1, (T*)output, in_scale0, in_scale1, out_scale);
#endif
                } else {
                    ppl_arithmetic_prepare_strides_nhwc(input_shape0, input_shape1, output_shape, packed_channel,
                    param.stride_in0, param.stride_in1, param.stride_out);
#ifdef OPT_ARITHM
                    if (out_scale == 0) out_scale = 1;
                    if (op_type == Arithmetic_Mul) {
                        ppl_cukernel_arithmetic_int8<op_type, T><<<grid_size, block_size, 0, stream>>>(num_elems, dim_count, param,
                            (const T*)input0, (const T*)input1, (T*)output, in_scale0*in_scale1/out_scale, 1, 1);
                    } else if (op_type == Arithmetic_Div) {
                        ppl_cukernel_arithmetic_int8<op_type, T><<<grid_size, block_size, 0, stream>>>(num_elems, dim_count, param,
                            (const T*)input0, (const T*)input1, (T*)output, in_scale0/in_scale1/out_scale, 1, 1);
                    } else if (op_type == Arithmetic_PRelu){
                        // printf("******************** PRelu Debug 3: s1=%f, s2=%f, s3=%f \n", in_scale0, in_scale1, out_scale);
                        ppl_cukernel_arithmetic_int8<op_type, T><<<grid_size, block_size, 0, stream>>>(num_elems, dim_count, param,
                            (const T*)input0, (const T*)input1, (T*)output, in_scale0, in_scale1, out_scale);
                    } else {
                        ppl_cukernel_arithmetic_int8<op_type, T><<<grid_size, block_size, 0, stream>>>(num_elems, dim_count, param,
                            (const T*)input0, (const T*)input1, (T*)output, in_scale0/out_scale, in_scale1/out_scale, 1);
                    }
#else
                    ppl_cukernel_arithmetic_int8<op_type, T><<<grid_size, block_size, 0, stream>>>(num_elems, dim_count, param,
                        (const T*)input0, (const T*)input1, (T*)output, in_scale0, in_scale1, out_scale);
#endif
                }
        } else if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY) {
            if (num_broadcast_dims == 1) {
                int inner_dim = 1;
                for(int it = axis + 1; it < dim_count; inner_dim *= output_shape->GetDim(it), ++it);
                int outer_stride = inner_dim * output_shape->GetDim(axis);
                bool first_shorter = false;
                if (input_shape0->GetRealDimCount() == input_shape1->GetRealDimCount() &&
                    input_shape0->GetDim(axis) < input_shape1->GetDim(axis)) {
                    first_shorter = true;
                }
                if (input_shape0->CalcElementsExcludingPadding() < input_shape1->CalcElementsExcludingPadding())  {
                    first_shorter = true;
                }
#ifdef OPT_ARITHM
                if (out_scale == 0) out_scale = 1;
                if (op_type == Arithmetic_Mul) {
                    ppl_cukernel_arithmetic_one_broadcast_int8<op_type, T><<<grid_size, block_size, 0,
                    stream>>>(num_elems, outer_stride, inner_dim, first_shorter, (const T*)input0, (const T*)input1, (T*)output, in_scale0*in_scale1/out_scale, 1, 1);
                } else if (op_type == Arithmetic_Div) {
                    ppl_cukernel_arithmetic_one_broadcast_int8<op_type, T><<<grid_size, block_size, 0,
                    stream>>>(num_elems, outer_stride, inner_dim, first_shorter, (const T*)input0, (const T*)input1, (T*)output, in_scale0/in_scale1/out_scale, 1, 1);
                } else if (op_type == Arithmetic_PRelu) {
                    ppl_cukernel_arithmetic_one_broadcast_int8<op_type, T><<<grid_size, block_size, 0,
                    stream>>>(num_elems, outer_stride, inner_dim, first_shorter, (const T*)input0, (const T*)input1, (T*)output, in_scale0, in_scale1, out_scale);
                } else {
                    ppl_cukernel_arithmetic_one_broadcast_int8<op_type, T><<<grid_size, block_size, 0,
                    stream>>>(num_elems, outer_stride, inner_dim, first_shorter, (const T*)input0, (const T*)input1, (T*)output, in_scale0/out_scale, in_scale1/out_scale, 1);
                }
#else
                ppl_cukernel_arithmetic_one_broadcast_int8<op_type, T><<<grid_size, block_size, 0,
                    stream>>>(num_elems, outer_stride, inner_dim, first_shorter, (const T*)input0, (const T*)input1, (T*)output, in_scale0, in_scale1, out_scale);
#endif

                return ppl::common::RC_SUCCESS;
            }
            ppl_arithmetic_prepare_strides(input_shape0, input_shape1,
                output_shape, packed_channel, param.stride_in0, param.stride_in1, param.stride_out);
#ifdef OPT_ARITHM
            if (out_scale == 0) out_scale = 1;
            if (op_type == Arithmetic_Mul) {
                ppl_cukernel_arithmetic_int8<op_type, T><<<grid_size, block_size, 0, stream>>>(num_elems, dim_count, param,
                    (const T*)input0, (const T*)input1, (T*)output, in_scale0*in_scale1/out_scale, 1, 1);
            } else if (op_type == Arithmetic_Div) {
                ppl_cukernel_arithmetic_int8<op_type, T><<<grid_size, block_size, 0, stream>>>(num_elems, dim_count, param,
                    (const T*)input0, (const T*)input1, (T*)output, in_scale0/in_scale1/out_scale, 1, 1);
            } else if (op_type == Arithmetic_PRelu) {
                ppl_cukernel_arithmetic_int8<op_type, T><<<grid_size, block_size, 0, stream>>>(num_elems, dim_count, param,
                    (const T*)input0, (const T*)input1, (T*)output, in_scale0, in_scale1, out_scale);
            } else {
                ppl_cukernel_arithmetic_int8<op_type, T><<<grid_size, block_size, 0, stream>>>(num_elems, dim_count, param,
                    (const T*)input0, (const T*)input1, (T*)output, in_scale0/out_scale, in_scale1/out_scale, 1);
            }
#else
            ppl_cukernel_arithmetic_int8<op_type, T><<<grid_size, block_size, 0, stream>>>(num_elems, dim_count, param, (const T*)input0, (const T*)input1, (T*)output, in_scale0, in_scale1, out_scale);
#endif
        } else {
            return ppl::common::RC_UNSUPPORTED;
        }
    }

    return ppl::common::RC_SUCCESS;
}


#define INSTANT(OPTYPE) \
ppl::common::RetCode PPLCUDAArithMetic##OPTYPE##ForwardImp( \
    cudaStream_t stream, \
    const ppl::common::TensorShape* input_shape0_ref, \
    const void *input0, \
    const ppl::common::TensorShape* input_shape1_ref, \
    const void *input1, \
    const ppl::common::TensorShape* output_shape_ref, \
    void *output, \
    float in_scale0, \
    float in_scale1, \
    float out_scale) { \
    ppl::common::TensorShape input_shape0_obj = *input_shape0_ref; \
    ppl::common::TensorShape input_shape1_obj = *input_shape1_ref; \
    ppl::common::TensorShape output_shape_obj = *output_shape_ref; \
    ppl::common::TensorShape* input_shape0 = &input_shape0_obj; \
    ppl::common::TensorShape* input_shape1 = &input_shape1_obj; \
    ppl::common::TensorShape* output_shape = &output_shape_obj; \
    if (input_shape0->GetDimCount() == input_shape1->GetDimCount() && input_shape0->GetDimCount() > 3 \
        && (input_shape0->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY)) { \
        ppl_refine_tensor_shape(input_shape0, input_shape1, output_shape); } \
    if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) { \
        return PPLCUDAArithMeticForwardImp<Arithmetic_##OPTYPE, half>(stream, \
            input_shape0, (const half*)input0, input_shape1, \
            (const half*)input1, output_shape, (half*)output); \
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) { \
        return PPLCUDAArithMeticForwardImp<Arithmetic_##OPTYPE, float>(stream, \
            input_shape0, (const float*)input0, input_shape1, \
            (const float*)input1, output_shape, (float*)output); \
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT64) { \
        return PPLCUDAArithMeticForwardImp<Arithmetic_##OPTYPE, double>(stream, \
            input_shape0, (const double*)input0, input_shape1, \
            (const double*)input1, output_shape, (double*)output); \
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_INT64) { \
        return PPLCUDAArithMeticForwardImp<Arithmetic_##OPTYPE, int64_t>(stream, \
            input_shape0, (const int64_t*)input0, input_shape1, \
            (const int64_t*)input1, output_shape, (int64_t*)output); \
    } else if(output_shape->GetDataType() == ppl::common::DATATYPE_INT32) { \
        return PPLCUDAArithMeticForwardImp<Arithmetic_##OPTYPE, int32_t>(stream, \
            input_shape0, (const int32_t*)input0, input_shape1, \
            (const int32_t*)input1, output_shape, (int32_t*)output); \
    } else if(output_shape->GetDataType() == ppl::common::DATATYPE_INT16) { \
        return PPLCUDAArithMeticForwardImp<Arithmetic_##OPTYPE, int16_t>(stream, \
            input_shape0, (const int16_t*)input0, input_shape1, \
            (const int16_t*)input1, output_shape, (int16_t*)output); \
    } else if(output_shape->GetDataType() == ppl::common::DATATYPE_INT8) { \
        return PPLCUDAArithMeticForwardImpInt8<Arithmetic_##OPTYPE, int8_t>(stream, \
            input_shape0, (const int8_t*)input0, input_shape1, \
            (const int8_t*)input1, output_shape, (int8_t*)output, in_scale0, in_scale1, out_scale); \
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_UINT64) { \
        return PPLCUDAArithMeticForwardImp<Arithmetic_##OPTYPE, uint64_t>(stream, \
            input_shape0, (const uint64_t*)input0, input_shape1, \
            (const uint64_t*)input1, output_shape, (uint64_t*)output); \
    } else if(output_shape->GetDataType() == ppl::common::DATATYPE_UINT32) { \
        return PPLCUDAArithMeticForwardImp<Arithmetic_##OPTYPE, uint32_t>(stream, \
            input_shape0, (const uint32_t*)input0, input_shape1, \
            (const uint32_t*)input1, output_shape, (uint32_t*)output); \
    } else if(output_shape->GetDataType() == ppl::common::DATATYPE_UINT16) { \
        return PPLCUDAArithMeticForwardImp<Arithmetic_##OPTYPE, uint16_t>(stream, \
            input_shape0, (const uint16_t*)input0, input_shape1, \
            (const uint16_t*)input1, output_shape, (uint16_t*)output); \
    } else { \
        return ppl::common::RC_UNSUPPORTED; \
    } \
}

template<ArithmeticOpType op_type, typename T>
ppl::common::RetCode PPLCUDAArithMeticForwardImpLimitNhwc(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape0,
    const T *input0,
    const ppl::common::TensorShape* input_shape1,
    const T *input1,
    const ppl::common::TensorShape* output_shape,
    T *output) {
    uint64_t num_elems = output_shape->CalcElementsExcludingPadding(); // only effective value calculated
    int dim_count = output_shape->GetDimCount();
    int block_size = 256;
    uint64_t grid_size = (num_elems + block_size - 1) / block_size;
    ArithmeticParam param_ndarray, param_nhwc;
    int packed_channel = 1;
    ppl_arithmetic_prepare_strides(input_shape0, input_shape1,
        output_shape, packed_channel, param_ndarray.stride_in0, param_ndarray.stride_in1, param_ndarray.stride_out);
    packed_channel = 8;
    ppl_arithmetic_prepare_strides_limit_nhwc(input_shape0, input_shape1,
        output_shape, packed_channel, param_nhwc.stride_in0, param_nhwc.stride_in1, param_nhwc.stride_out);

    int axis = 0; bool bidirectional = false;
    int num_broadcast_dims = ppl_get_num_broadcast_dims(input_shape0, input_shape1, axis, bidirectional);
    bool first_shorter = false;
    if (input_shape0->CalcElementsExcludingPadding() < input_shape1->CalcElementsExcludingPadding())  {
        first_shorter = true;
    }
    if (num_broadcast_dims == 1) {
        if (first_shorter) param_nhwc.stride_in0[axis] = 0;
        else param_nhwc.stride_in1[axis] = 0;
    }
    if (!bidirectional) {
#ifdef OPT_ARITHM
        constexpr int N = 16 / sizeof(T);
        int64_t channels = first_shorter ? input_shape1->GetDim(1) : input_shape0->GetDim(1);
        const bool use_n_opt = (channels % N == 0);

        if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 &&
                output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16 &&
                use_n_opt && dim_count == 4 && num_broadcast_dims == 1 && axis == 1 &&
                (input_shape0->GetDim(1) == 1 || input_shape1->GetDim(1) == 1)) {
            grid_size = (grid_size + N - 1) / N;
            ppl_cukernel_arithmetic_limit_nhwc8_broadcast_c_float4_opt<op_type, T, N><<<grid_size, block_size, 0,
                stream>>>(num_elems, channels,
                    first_shorter ? (const T*)input1 : (const T*)input0,
                    first_shorter ? (const T*)input0 : (const T*)input1,
                    (T*)output);
            return ppl::common::RC_SUCCESS;
        }
#endif
        ppl_cukernel_arithmetic_limit_nhwc<op_type, T><<<grid_size, block_size, 0,
                stream>>>(num_elems, dim_count, param_ndarray, param_nhwc, (const T*)input0, (const T*)input1, (T*)output);
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }

    return ppl::common::RC_SUCCESS;
}

template<ArithmeticOpType op_type, typename T>
ppl::common::RetCode PPLCUDAArithMeticForwardImpLimitNhwcInt8(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape0,
    const T *input0,
    const ppl::common::TensorShape* input_shape1,
    const T *input1,
    const ppl::common::TensorShape* output_shape,
    T *output,
    float in_scale0,
    float in_scale1,
    float out_scale) {
    uint64_t num_elems = output_shape->CalcElementsExcludingPadding(); // only effective value calculated
    int dim_count = output_shape->GetDimCount();
    int block_size = 256;
    uint64_t grid_size = (num_elems + block_size - 1) / block_size;
    ArithmeticParam param_ndarray, param_nhwc;
    int packed_channel = 1;
    ppl_arithmetic_prepare_strides(input_shape0, input_shape1,
        output_shape, packed_channel, param_ndarray.stride_in0, param_ndarray.stride_in1, param_ndarray.stride_out);
    packed_channel = 8;
    if (input_shape0->GetDataFormat() == ppl::common::DATAFORMAT_NHWC16) packed_channel = 16;
    ppl_arithmetic_prepare_strides_limit_nhwc(input_shape0, input_shape1,
        output_shape, packed_channel, param_nhwc.stride_in0, param_nhwc.stride_in1, param_nhwc.stride_out);

    int axis = 0; bool bidirectional = false;
    int num_broadcast_dims = ppl_get_num_broadcast_dims(input_shape0, input_shape1, axis, bidirectional);
    bool first_shorter = false;
    if (input_shape0->CalcElementsExcludingPadding() < input_shape1->CalcElementsExcludingPadding())  {
        first_shorter = true;
    }
    if (num_broadcast_dims == 1) {
        if (first_shorter) param_nhwc.stride_in0[axis] = 0;
        else param_nhwc.stride_in1[axis] = 0;
    }
    if (!bidirectional) {
#ifdef OPT_ARITHM
        if (out_scale == 0) out_scale = 1;
        if (op_type == Arithmetic_Mul) {
            ppl_cukernel_arithmetic_limit_nhwc_int8<op_type, T><<<grid_size, block_size, 0,
                stream>>>(num_elems, dim_count, param_ndarray, param_nhwc, (const T*)input0, (const T*)input1, (T*)output,
                in_scale0*in_scale1/out_scale, 1, 1);
        } else if (op_type == Arithmetic_Div) {
            ppl_cukernel_arithmetic_limit_nhwc_int8<op_type, T><<<grid_size, block_size, 0,
                stream>>>(num_elems, dim_count, param_ndarray, param_nhwc, (const T*)input0, (const T*)input1, (T*)output,
                in_scale0/in_scale1/out_scale, 1, 1);
        } else {
            ppl_cukernel_arithmetic_limit_nhwc_int8<op_type, T><<<grid_size, block_size, 0,
                    stream>>>(num_elems, dim_count, param_ndarray, param_nhwc, (const T*)input0, (const T*)input1, (T*)output,
                    in_scale0/out_scale, in_scale1/out_scale, 1);
        }
#else
        ppl_cukernel_arithmetic_limit_nhwc_int8<op_type, T><<<grid_size, block_size, 0,
                stream>>>(num_elems, dim_count, param_ndarray, param_nhwc, (const T*)input0, (const T*)input1, (T*)output,
                in_scale0, in_scale1, out_scale);
#endif
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }

    return ppl::common::RC_SUCCESS;
}

#define INSTANT_LIMNHWC(OPTYPE) \
ppl::common::RetCode PPLCUDAArithMetic##OPTYPE##ForwardImp( \
    cudaStream_t stream, \
    const ppl::common::TensorShape* input_shape0_ref, \
    const void *input0, \
    const ppl::common::TensorShape* input_shape1_ref, \
    const void *input1, \
    const ppl::common::TensorShape* output_shape_ref, \
    void *output, \
    float in_scale0, \
    float in_scale1, \
    float out_scale) { \
    ppl::common::TensorShape input_shape0_obj = *input_shape0_ref; \
    ppl::common::TensorShape input_shape1_obj = *input_shape1_ref; \
    ppl::common::TensorShape output_shape_obj = *output_shape_ref; \
    ppl::common::TensorShape* input_shape0 = &input_shape0_obj; \
    ppl::common::TensorShape* input_shape1 = &input_shape1_obj; \
    ppl::common::TensorShape* output_shape = &output_shape_obj; \
    if (input_shape0->GetDimCount() == input_shape1->GetDimCount() && input_shape0->GetDimCount() > 3 \
        && (input_shape0->GetDataFormat() == ppl::common::DATAFORMAT_NDARRAY)) { \
        ppl_refine_tensor_shape(input_shape0, input_shape1, output_shape); } \
    if (output_shape->GetDataFormat() == ppl::common::DATAFORMAT_NHWC8 && \
        ((input_shape0->GetDimCount() >= 2 && (input_shape0->GetDim(1) & 0x7)) || \
        (input_shape1->GetDimCount() >= 2 && (input_shape1->GetDim(1) & 0x7)))) { \
        if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) { \
            return PPLCUDAArithMeticForwardImpLimitNhwc<Arithmetic_##OPTYPE, half>(stream, \
                input_shape0, (const half*)input0, input_shape1, \
                (const half*)input1, output_shape, (half*)output); \
        } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) { \
            return PPLCUDAArithMeticForwardImpLimitNhwc<Arithmetic_##OPTYPE, float>(stream, \
                input_shape0, (const float*)input0, input_shape1, \
                (const float*)input1, output_shape, (float*)output); \
        } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT64) { \
            return PPLCUDAArithMeticForwardImpLimitNhwc<Arithmetic_##OPTYPE, double>(stream, \
                input_shape0, (const double*)input0, input_shape1, \
                (const double*)input1, output_shape, (double*)output); \
        } else if (output_shape->GetDataType() == ppl::common::DATATYPE_INT64) { \
            return PPLCUDAArithMeticForwardImpLimitNhwc<Arithmetic_##OPTYPE, int64_t>(stream, \
                input_shape0, (const int64_t*)input0, input_shape1, \
                (const int64_t*)input1, output_shape, (int64_t*)output); \
        } else if(output_shape->GetDataType() == ppl::common::DATATYPE_INT32) { \
            return PPLCUDAArithMeticForwardImpLimitNhwc<Arithmetic_##OPTYPE, int32_t>(stream, \
                input_shape0, (const int32_t*)input0, input_shape1, \
                (const int32_t*)input1, output_shape, (int32_t*)output); \
        } else if(output_shape->GetDataType() == ppl::common::DATATYPE_INT16) { \
            return PPLCUDAArithMeticForwardImpLimitNhwc<Arithmetic_##OPTYPE, int16_t>(stream, \
                input_shape0, (const int16_t*)input0, input_shape1, \
                (const int16_t*)input1, output_shape, (int16_t*)output); \
        } else if(output_shape->GetDataType() == ppl::common::DATATYPE_INT8) { \
            return PPLCUDAArithMeticForwardImpLimitNhwcInt8<Arithmetic_##OPTYPE, int8_t>(stream, \
                input_shape0, (const int8_t*)input0, input_shape1, \
                (const int8_t*)input1, output_shape, (int8_t*)output, in_scale0, in_scale1, out_scale); \
        } else if (output_shape->GetDataType() == ppl::common::DATATYPE_UINT64) { \
            return PPLCUDAArithMeticForwardImpLimitNhwc<Arithmetic_##OPTYPE, uint64_t>(stream, \
                input_shape0, (const uint64_t*)input0, input_shape1, \
                (const uint64_t*)input1, output_shape, (uint64_t*)output); \
        } else if(output_shape->GetDataType() == ppl::common::DATATYPE_UINT32) { \
            return PPLCUDAArithMeticForwardImpLimitNhwc<Arithmetic_##OPTYPE, uint32_t>(stream, \
                input_shape0, (const uint32_t*)input0, input_shape1, \
                (const uint32_t*)input1, output_shape, (uint32_t*)output); \
        } else if(output_shape->GetDataType() == ppl::common::DATATYPE_UINT16) { \
            return PPLCUDAArithMeticForwardImpLimitNhwc<Arithmetic_##OPTYPE, uint16_t>(stream, \
                input_shape0, (const uint16_t*)input0, input_shape1, \
                (const uint16_t*)input1, output_shape, (uint16_t*)output); \
        } else { \
            return ppl::common::RC_UNSUPPORTED; \
        } \
    } \
    if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) { \
        return PPLCUDAArithMeticForwardImp<Arithmetic_##OPTYPE, half>(stream, \
            input_shape0, (const half*)input0, input_shape1, \
            (const half*)input1, output_shape, (half*)output); \
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) { \
        return PPLCUDAArithMeticForwardImp<Arithmetic_##OPTYPE, float>(stream, \
            input_shape0, (const float*)input0, input_shape1, \
            (const float*)input1, output_shape, (float*)output); \
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_FLOAT64) { \
        return PPLCUDAArithMeticForwardImp<Arithmetic_##OPTYPE, double>(stream, \
            input_shape0, (const double*)input0, input_shape1, \
            (const double*)input1, output_shape, (double*)output); \
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_INT64) { \
        return PPLCUDAArithMeticForwardImp<Arithmetic_##OPTYPE, int64_t>(stream, \
            input_shape0, (const int64_t*)input0, input_shape1, \
            (const int64_t*)input1, output_shape, (int64_t*)output); \
    } else if(output_shape->GetDataType() == ppl::common::DATATYPE_INT32) { \
        return PPLCUDAArithMeticForwardImp<Arithmetic_##OPTYPE, int32_t>(stream, \
            input_shape0, (const int32_t*)input0, input_shape1, \
            (const int32_t*)input1, output_shape, (int32_t*)output); \
    } else if(output_shape->GetDataType() == ppl::common::DATATYPE_INT16) { \
        return PPLCUDAArithMeticForwardImp<Arithmetic_##OPTYPE, int16_t>(stream, \
            input_shape0, (const int16_t*)input0, input_shape1, \
            (const int16_t*)input1, output_shape, (int16_t*)output); \
    } else if(output_shape->GetDataType() == ppl::common::DATATYPE_INT8) { \
        return PPLCUDAArithMeticForwardImpInt8<Arithmetic_##OPTYPE, int8_t>(stream, \
            input_shape0, (const int8_t*)input0, input_shape1, \
            (const int8_t*)input1, output_shape, (int8_t*)output, in_scale0, in_scale1, out_scale); \
    } else if (output_shape->GetDataType() == ppl::common::DATATYPE_UINT64) { \
        return PPLCUDAArithMeticForwardImp<Arithmetic_##OPTYPE, uint64_t>(stream, \
            input_shape0, (const uint64_t*)input0, input_shape1, \
            (const uint64_t*)input1, output_shape, (uint64_t*)output); \
    } else if(output_shape->GetDataType() == ppl::common::DATATYPE_UINT32) { \
        return PPLCUDAArithMeticForwardImp<Arithmetic_##OPTYPE, uint32_t>(stream, \
            input_shape0, (const uint32_t*)input0, input_shape1, \
            (const uint32_t*)input1, output_shape, (uint32_t*)output); \
    } else if(output_shape->GetDataType() == ppl::common::DATATYPE_UINT16) { \
        return PPLCUDAArithMeticForwardImp<Arithmetic_##OPTYPE, uint16_t>(stream, \
            input_shape0, (const uint16_t*)input0, input_shape1, \
            (const uint16_t*)input1, output_shape, (uint16_t*)output); \
    } else if(output_shape->GetDataType() == ppl::common::DATATYPE_UINT8) { \
        return PPLCUDAArithMeticForwardImp<Arithmetic_##OPTYPE, uint8_t>(stream, \
            input_shape0, (const uint8_t*)input0, input_shape1, \
            (const uint8_t*)input1, output_shape, (uint8_t*)output); \
    } else { \
        return ppl::common::RC_UNSUPPORTED; \
    } \
}

INSTANT(Add);
INSTANT(Sub);
INSTANT(Mul);
INSTANT_LIMNHWC(Div);
INSTANT_LIMNHWC(Max);
INSTANT_LIMNHWC(Min);
INSTANT_LIMNHWC(Pow);
INSTANT_LIMNHWC(PRelu);
INSTANT_LIMNHWC(Mod);
INSTANT_LIMNHWC(FMod);

#undef INSTANT
