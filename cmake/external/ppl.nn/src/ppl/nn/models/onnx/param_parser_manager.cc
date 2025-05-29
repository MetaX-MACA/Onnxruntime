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

#include "ppl/nn/models/onnx/param_parser_manager.h"

// NOTE: sorted in alphabet order
#include "ppl/nn/models/onnx/parsers/onnx/parse_argmax_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_argmin_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_batchnormalization_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_bitwise_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_cast_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_celu_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_clip_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_concat_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_constant_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_constant_of_shape_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_conv_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_convtranspose_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_cumsum_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_deformconv_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_depth_to_space_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_einsum_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_elu_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_eyelike_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_flatten_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_gather_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_gather_nd_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_gemm_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_gru_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_hard_sigmoid_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_if_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_instancenormalization_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_isinf_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_leaky_relu_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_loop_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_lrn_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_lstm_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_maxunpool_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_mod_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_non_max_suppression_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_one_hot_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_pad_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_pooling_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_random_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_random_uniform_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_reduce_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_reshape_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_resize_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_roialign_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_scatter_elements_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_selu_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_shrink_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_slice_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_softmax_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_split_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_split_to_sequence_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_space_to_depth_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_squeeze_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_swish_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_thresholded_relu_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_topk_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_transpose_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_trilu_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_unsqueeze_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_logsoftmax_param.h"

#include "ppl/nn/models/onnx/parsers/mmcv/parse_mmcv_gridsample_param.h"
#include "ppl/nn/models/onnx/parsers/mmcv/parse_mmcv_modulated_deform_conv2d_param.h"
#include "ppl/nn/models/onnx/parsers/mmcv/parse_mmcv_nonmaxsupression_param.h"
#include "ppl/nn/models/onnx/parsers/mmcv/parse_mmcv_roialign_param.h"

#include "ppl/nn/models/onnx/parsers/pmx/parse_group_norm_param.h"
#include "ppl/nn/models/onnx/parsers/pmx/parse_layer_norm_param.h"
#include "ppl/nn/models/onnx/parsers/pmx/parse_ms_deformable_attention_param.h"
#include "ppl/nn/models/onnx/parsers/pmx/parse_multi_head_attn_v1_param.h"
#include "ppl/nn/models/onnx/parsers/pmx/parse_ppl_channel_shuffle_param.h"
#include "ppl/nn/models/onnx/parsers/pmx/parse_split_activation_param.h"
#include "ppl/nn/models/onnx/parsers/pmx/parse_quick_gelu_param.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx {

template <typename T>
shared_ptr<ir::Attr> CreateParam() {
    return make_shared<T>();
}

#define PPL_REGISTER_OP_WITH_PARAM(domain, type, first_version, last_version, param_type, parse_param_func,    \
                                   pack_param_func)                                                            \
    do {                                                                                                       \
        if (last_version < first_version) {                                                                    \
            LOG(ERROR) << "register op[" << domain << ":" << type << "] failed: last_version[" << last_version \
                       << "] < first_version[" << first_version << "]";                                        \
            exit(-1);                                                                                          \
        }                                                                                                      \
                                                                                                               \
        ParserInfo parse_info;                                                                                 \
        parse_info.create_param = CreateParam<param_type>;                                                     \
        parse_info.parse_param = parse_param_func;                                                             \
        parse_info.pack_param = pack_param_func;                                                               \
        auto status = Register(domain, type, utils::VersionRange(first_version, last_version), parse_info);    \
        if (status != RC_SUCCESS) {                                                                            \
            exit(-1);                                                                                          \
        }                                                                                                      \
    } while (0)

#define PPL_REGISTER_OP_WITHOUT_PARAM(domain, type, first_version, last_version, parse_param_func) \
    do {                                                                                           \
        ParserInfo parse_info;                                                                     \
        parse_info.create_param = nullptr;                                                         \
        parse_info.parse_param = parse_param_func;                                                 \
        parse_info.pack_param = nullptr;                                                           \
        Register(domain, type, utils::VersionRange(first_version, last_version), parse_info);      \
    } while (0)

// NOTE: sorted in alphabet order
ParamParserManager::ParamParserManager() {
    // A
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Abs", 6, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Acos", 7, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Acosh", 7, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Add", 7, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "And", 7, 16, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("", "ArgMax", 1, 16, ArgMaxParam, ParseArgMaxParam, PackArgMaxParam);
    PPL_REGISTER_OP_WITH_PARAM("", "ArgMin", 1, 16, ArgMinParam, ParseArgMinParam, PackArgMinParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Asin", 7, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Asinh", 7, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Atan", 7, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Atanh", 7, 16, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("", "AveragePool", 1, 16, PoolingParam, ParsePoolingParam, PackPoolingParam);
    // B
    PPL_REGISTER_OP_WITH_PARAM("", "BatchNormalization", 9, 13, BatchNormalizationParam, ParseBatchNormalizationParam,
                               PackBatchNormalizationParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("","BitwiseAnd", 6, 18, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("","BitwiseNot", 6, 18, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("","BitwiseOr", 6, 18, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("","BitwiseXor", 6, 18, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("","BitShift", 6, 18, BitShiftParam,ParseBitShiftParam,PackBitShiftParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("com.metax-tech","BitwiseAnd", 6, 18, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("com.metax-tech","BitwiseNot", 6, 18, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("com.metax-tech","BitwiseOr", 6, 18, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("com.metax-tech","BitwiseXor", 6, 18, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("com.metax-tech","BitShift", 6, 18, BitShiftParam,ParseBitShiftParam,PackBitShiftParam);
    // C
    PPL_REGISTER_OP_WITH_PARAM("", "Cast", 9, 16, CastParam, ParseCastParam, PackCastParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Ceil", 6, 16, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("", "Celu", 1, 16, CeluParam, ParseCeluParam, PackCeluParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Clip", 6, 16, ParseClipParam);
    PPL_REGISTER_OP_WITH_PARAM("", "Concat", 4, 16, ConcatParam, ParseConcatParam, PackConcatParam);
    PPL_REGISTER_OP_WITH_PARAM("", "Constant", 9, 16, ConstantParam, ParseConstantParam, PackConstantParam);
    PPL_REGISTER_OP_WITH_PARAM("", "ConstantOfShape", 9, 16, ConstantOfShapeParam, ParseConstantOfShapeParam,
                               PackConstantOfShapeParam);
    PPL_REGISTER_OP_WITH_PARAM("", "Conv", 1, 16, ConvParam, ParseConvParam, PackConvParam);
    PPL_REGISTER_OP_WITH_PARAM("", "ConvTranspose", 1, 16, ConvTransposeParam, ParseConvTransposeParam,
                               PackConvTransposeParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Cos", 7, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Cosh", 7, 16, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("", "CumSum", 11, 16, CumSumParam, ParseCumSumParam, PackCumSumParam);
    // D
    PPL_REGISTER_OP_WITH_PARAM("", "DeformConv", 1, 19, DeformConvParam, ParseDeformConvParam, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("", "DepthToSpace", 1, 16, DepthToSpaceParam, ParseDepthToSpaceParam,
                               PackDepthToSpaceParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Div", 7, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Dropout", 1, 16, nullptr); // will be skip
    // E
    PPL_REGISTER_OP_WITH_PARAM("", "Einsum", 12, 16, EinSumParam, ParseEinSumParam, PackEinSumParam);
    PPL_REGISTER_OP_WITH_PARAM("", "Elu", 1, 16, EluParam, ParseEluParam, PackEluParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Equal", 7, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Erf", 9, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Exp", 6, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Expand", 8, 16, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("", "EyeLike", 9, 16, EyeLikeParam, ParseEyeLikeParam, PackEyeLikeParam);
    // F
    PPL_REGISTER_OP_WITH_PARAM("", "Flatten", 1, 16, FlattenParam, ParseFlattenParam, PackFlattenParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Floor", 6, 16, nullptr);
    // G
    PPL_REGISTER_OP_WITH_PARAM("", "Gather", 1, 16, GatherParam, ParseGatherParam, PackGatherParam);
    PPL_REGISTER_OP_WITH_PARAM("", "GatherElements", 1, 16, GatherParam, ParseGatherParam, PackGatherParam);
    PPL_REGISTER_OP_WITH_PARAM("", "GatherND", 11, 16, GatherNDParam, ParseGatherNDParam, PackGatherNDParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("com.microsoft", "Gelu", 1, 16, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("", "Gemm", 9, 16, GemmParam, ParseGemmParam, PackGemmParam);
    PPL_REGISTER_OP_WITH_PARAM("", "GlobalAveragePool", 1, 16, PoolingParam, ParsePoolingParam, PackPoolingParam);
    PPL_REGISTER_OP_WITH_PARAM("", "GlobalMaxPool", 1, 16, PoolingParam, ParsePoolingParam, PackPoolingParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Greater", 7, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "GreaterOrEqual", 7, 16, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("", "GRU", 1, 14, GRUParam, ParseGRUParam, PackGRUParam);
    // H
    PPL_REGISTER_OP_WITH_PARAM("", "HardSigmoid", 6, 16, HardSigmoidParam, ParseHardSigmoidParam, PackHardSigmoidParam);
    //PPL_REGISTER_OP_WITHOUT_PARAM("", "HardSwish", 1, 16, nullptr);
    //PPL_REGISTER_OP_WITHOUT_PARAM("com.metax-tech", "HardSwish", 1, 16, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("", "HardSwish", 1, 16, HardSigmoidParam, ParseHardSigmoidParam, PackHardSigmoidParam);
    PPL_REGISTER_OP_WITH_PARAM("com.metax-tech", "HardSwish", 1, 16, HardSigmoidParam, ParseHardSigmoidParam, PackHardSigmoidParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("com.metax-tech", "HardMish", 1, 16, nullptr);
    // I
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Identity", 1, 16, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("", "If", 1, 13, IfParam, ParseIfParam, PackIfParam);
    PPL_REGISTER_OP_WITH_PARAM("", "InstanceNormalization", 6, 16, InstanceNormalizationParam,
                               ParseInstanceNormalizationParam, PackInstanceNormalizationParam);
    PPL_REGISTER_OP_WITH_PARAM("", "IsInf", 1, 16, IsinfParam, ParseIsinfParam, PackIsinfParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "IsNaN", 1, 16, nullptr);
    // L
    PPL_REGISTER_OP_WITH_PARAM("com.metax-tech", "LayerNorm", 1, 16, ppl::nn::pmx::LayerNormParam, ParseLayerNormParam, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("", "LayerNormalization", 1, 18, ppl::nn::pmx::LayerNormParam, ParseLayerNormParam, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("", "LeakyRelu", 6, 16, LeakyReluParam, ParseLeakyReluParam, PackLeakyReluParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Less", 7, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Log", 6, 16, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("", "Loop", 1, 13, LoopParam, ParseLoopParam, PackLoopParam);
    PPL_REGISTER_OP_WITH_PARAM("", "LogSoftmax", 1, 13, LogSoftmaxParam, ParseLogSoftmaxParam, PackLogSoftmaxParam);
    PPL_REGISTER_OP_WITH_PARAM("", "LRN", 1, 16, LRNParam, ParseLRNParam, PackLRNParam);
    PPL_REGISTER_OP_WITH_PARAM("", "LSTM", 7, 13, LSTMParam, ParseLSTMParam, PackLSTMParam);
    // M
    PPL_REGISTER_OP_WITHOUT_PARAM("", "MatMul", 1, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Max", 6, 16, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("", "MaxPool", 1, 16, PoolingParam, ParsePoolingParam, PackPoolingParam);
    PPL_REGISTER_OP_WITH_PARAM("", "MaxUnpool", 9, 16, MaxUnpoolParam, ParseMaxUnpoolParam, PackMaxUnpoolParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Min", 6, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("com.metax-tech", "Mish", 1, 16, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("", "Mod", 10, 13, ModParam, ParseModParam, PackModParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Mul", 7, 16, nullptr);
    // N
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Neg", 1, 16, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("", "NonMaxSuppression", 10, 16, NonMaxSuppressionParam, ParseNonMaxSuppressionParam,
                               PackNonMaxSuppressionParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "NonZero", 9, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Not", 1, 16, nullptr);
    // O
    PPL_REGISTER_OP_WITH_PARAM("", "OneHot", 9, 16, OneHotParam, ParseOneHotParam, PackOneHotParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Or", 7, 16, nullptr);
    // P
    PPL_REGISTER_OP_WITH_PARAM("", "Pad", 2, 16, PadParam, ParsePadParam, PackPadParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Pow", 7, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "PRelu", 6, 16, nullptr);
    // R
    PPL_REGISTER_OP_WITH_PARAM("", "RandomNormal", 1, 16, RandomNormalParam, ParseRandomNormalParam, PackRandomNormalParam);
    PPL_REGISTER_OP_WITH_PARAM("", "RandomNormalLike", 1, 16, RandomNormalLikeParam, ParseRandomNormalLikeParam, PackRandomNormalLikeParam);
    PPL_REGISTER_OP_WITH_PARAM("", "RandomUniform", 1, 16, RandomUniformParam, ParseRandomUniformParam, PackRandomUniformParam);
    PPL_REGISTER_OP_WITH_PARAM("", "RandomUniformLike", 1, 16, RandomUniformLikeParam, ParseRandomUniformLikeParam, PackRandomUniformLikeParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Range", 11, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Reciprocal", 6, 16, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("", "ReduceL2", 1, 16, ReduceParam, ParseReduceParam, PackReduceParam);
    PPL_REGISTER_OP_WITH_PARAM("", "ReduceMax", 1, 16, ReduceParam, ParseReduceParam, PackReduceParam);
    PPL_REGISTER_OP_WITH_PARAM("", "ReduceMean", 1, 16, ReduceParam, ParseReduceParam, PackReduceParam);
    PPL_REGISTER_OP_WITH_PARAM("", "ReduceMin", 1, 16, ReduceParam, ParseReduceParam, PackReduceParam);
    PPL_REGISTER_OP_WITH_PARAM("", "ReduceProd", 1, 16, ReduceParam, ParseReduceParam, PackReduceParam);
    PPL_REGISTER_OP_WITH_PARAM("", "ReduceSum", 1, 16, ReduceParam, ParseReduceParam, PackReduceParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Relu", 6, 16, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("", "Reshape", 5, 16, ReshapeParam, ParseReshapeParam, PackReshapeParam);
    PPL_REGISTER_OP_WITH_PARAM("", "Resize", 11, 16, ResizeParam, ParseResizeParam, PackResizeParam);
    PPL_REGISTER_OP_WITH_PARAM("", "RoiAlign", 10, 15, RoiAlignParam, ParseRoiAlignParam, PackRoiAlignParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Round", 11, 16, nullptr);
    // // S
    PPL_REGISTER_OP_WITH_PARAM("", "ScatterElements", 11, 15, ScatterElementsParam, ParseScatterElementsParam,
                               PackScatterElementsParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "ScatterND", 11, 15, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("", "Selu", 1, 16, SeluParam, ParseSeluParam, PackSeluParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "SequenceAt", 11, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Shape", 1, 16, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("", "Shrink", 1, 16, ShrinkParam, ParseShrinkParam, PackShrinkParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Sigmoid", 6, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Sign", 9, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Sin", 7, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Sinh", 7, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Size", 1, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Slice", 1, 16, ParseSliceParam);
    PPL_REGISTER_OP_WITH_PARAM("", "Softmax", 1, 16, SoftmaxParam, ParseSoftmaxParam, PackSoftmaxParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Softplus", 1, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Softsign", 1, 14, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("", "Split", 2, 17, SplitParam, ParseSplitParam, PackSplitParam);
    PPL_REGISTER_OP_WITH_PARAM("", "SplitToSequence", 11, 16, SplitToSequenceParam, ParseSplitToSequenceParam,
                               PackSplitToSequenceParam);
    PPL_REGISTER_OP_WITH_PARAM("", "SpaceToDepth", 1, 16, SpaceToDepthParam, ParseSpaceToDepthParam,
                               PackSpaceToDepthParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Sqrt", 6, 16, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("", "Squeeze", 1, 16, SqueezeParam, ParseSqueezeParam, PackSqueezeParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Sub", 7, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Sum", 6, 16, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("", "Swish", 1, 16, SwishParam, ParseSwishParam, PackSwishParam);
    PPL_REGISTER_OP_WITH_PARAM("com.metax-tech", "Swish", 1, 16, SwishParam, ParseSwishParam, PackSwishParam);
    // T
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Tanh", 6, 16, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("", "ThresholdedRelu", 1, 16, ThresholdedReluParam, ParseThresholdedReluParam, PackThresholdedReluParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Tile", 6, 16, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("", "TopK", 1, 16, TopKParam, ParseTopKParam, PackTopKParam);
    PPL_REGISTER_OP_WITH_PARAM("", "Transpose", 1, 16, TransposeParam, ParseTransposeParam, PackTransposeParam);
    PPL_REGISTER_OP_WITH_PARAM("", "Trilu", 14, 16, TriluParam, ParseTriluParam, PackTriluParam);
    // U
    PPL_REGISTER_OP_WITH_PARAM("", "Unsqueeze", 1, 16, UnsqueezeParam, ParseUnsqueezeParam, PackUnsqueezeParam);
    // W
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Where", 9, 16, nullptr);
    // X
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Xor", 7, 16, nullptr);

    // mmcv op param parser
    PPL_REGISTER_OP_WITH_PARAM("mmcv", "grid_sampler", 1, 1, ppl::nn::mmcv::MMCVGridSampleParam,
                               ParseMMCVGridSampleParam, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("mmcv", "MMCVRoiAlign", 1, 1, ppl::nn::mmcv::MMCVRoiAlignParam, ParseMMCVRoiAlignParam,
                               nullptr);
    PPL_REGISTER_OP_WITH_PARAM("mmcv", "MMCVModulatedDeformConv2d", 1, 1, ppl::nn::mmcv::MMCVModulatedDeformConv2dParam,
                               ParseMMCVModulatedDeformConv2dParam, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("mmcv", "NonMaxSuppression", 1, 1, ppl::nn::mmcv::MMCVNMSParam, ParseMMCVNMSParam,
                               nullptr);

    // ppl op param parser
    PPL_REGISTER_OP_WITHOUT_PARAM("pmx", "GELU", 1, 16, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("pmx", "ChannelShuffle", 1, 1, ppl::nn::pmx::ChannelShuffleParam,
                               ParseChannelShuffleParam, nullptr);

    PPL_REGISTER_OP_WITH_PARAM("mmdeploy", "MSDeformAttn", 1, 1, ppl::nn::pmx::MSDeformAttnParam, ParseMSDeformAttnParam, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("com.metax-tech", "MultiHeadAttentionV1", 1, 1, ppl::nn::pmx::MultiHeadAttnV1Param, ParseMultiHeadAttnV1Param, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("com.metax-tech", "GroupNormalization", 1, 1, ppl::nn::pmx::GroupNormalizationParam, ParseGroupNormParam, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("com.metax-tech", "SplitActivation", 1, 1, ppl::nn::pmx::SplitActivationParam, ParseSplitActivationParam, nullptr);

    PPL_REGISTER_OP_WITH_PARAM("pmx", "LayerNorm", 1, 1, ppl::nn::pmx::LayerNormParam, ParseLayerNormParam, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("com.microsoft", "QuickGelu", 1, 16, ppl::nn::pmx::QuickGeluParam, ParseQuickGeluParam, nullptr);

}

}}} // namespace ppl::nn::onnx
