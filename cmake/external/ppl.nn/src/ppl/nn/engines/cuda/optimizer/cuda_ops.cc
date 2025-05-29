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

#include "ppl/nn/engines/cuda/optimizer/opt_kernel_creator_manager.h"
using namespace std;
using namespace ppl::common;

#include "ppl/nn/engines/cuda/optimizer/ops/pmx/bridge_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/pmx/batch_gemm_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/pmx/channel_shuffle_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/pmx/fused_add_layernorm_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/pmx/fused_gathernd_reshape_concat_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/pmx/fused_gather_reshape_add_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/pmx/fused_mul_add_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/pmx/multi_head_attn_v1_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/pmx/ms_deformable_attention_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/pmx/layernorm_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/pmx/shape_operation_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/pmx/gelu_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/pmx/quick_gelu_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/pmx/group_normalization.h"
#include "ppl/nn/engines/cuda/optimizer/ops/pmx/reduce_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/pmx/split_activation_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/conv_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/abs_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/add_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/and_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/argmax_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/argmin_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/average_pool_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/batch_normalization_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/bitwise_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/cast_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/ceil_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/celu_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/convtranspose_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/clip_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/concat_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/constant_of_shape_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/cos_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/cumsum_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/deformconv_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/depth_to_space_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/div_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/einsum_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/elu_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/equal_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/erf_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/exp_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/expand_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/eyelike_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/flatten_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/floor_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/gather_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/gather_elements_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/gather_nd_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/gemm_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/global_average_pool_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/global_max_pool_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/greater_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/greater_or_equal_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/hard_sigmoid_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/hard_mish_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/hard_swish_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/identity_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/if_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/instance_normalization_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/isinf_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/isnan_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/leaky_relu_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/less_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/log_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/loop_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/logsoftmax_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/lstm_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/matmul_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/max_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/max_pool_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/max_unpool_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/mod_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/min_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/mish_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/mul_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/neg_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/non_max_suppression_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/non_zero_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/not_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/roialign_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/one_hot_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/or_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/pad_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/pow_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/prelu_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/random_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/range_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/reciprocal_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/reduce_l2_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/relu_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/reshape_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/resize_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/round_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/scatter_elements_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/scatter_nd_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/sequence_at_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/selu_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/shape_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/shrink_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/sigmoid_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/sign_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/sin_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/size_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/slice_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/softmax_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/softplus_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/softsign_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/split_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/split_to_sequence_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/space_to_depth_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/sqrt_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/squeeze_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/sub_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/sum_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/swish_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/tanh_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/thresholded_relu_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/tile_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/topk_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/transpose_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/trigonometric_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/trilu_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/unsqueeze_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/where_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/xor_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/onnx/lstm_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/mmcv/mmcv_non_max_suppression_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/mmcv/mmcv_roialign_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/mmcv/mmcv_gridsample_op.h"
#include "ppl/nn/engines/cuda/optimizer/ops/mmcv/mmcv_modulated_deform_conv2d_op.h"

namespace ppl { namespace nn { namespace cuda {

template <typename T>
static CudaOptKernel* GenericCreateOptKernel(const ir::Node* node) {
    return new T(node);
}

template <typename T>
static void RegisterOptKernelCreator(const string& domain, const string& type, uint64_t first_version,
                                     uint64_t last_version) {
    if (last_version < first_version) {
        LOG(ERROR) << "register op[" << domain << ":" << type << "] failed: last_version[" << last_version
                   << "] < first_version[" << first_version << "]";
        exit(-1);
    }
    OptKernelCreatorManager::GetInstance()->Register(domain, type, utils::VersionRange(first_version, last_version),
                                                     GenericCreateOptKernel<T>);
}

// NOTE: sorted in alphabet order
void RegisterBuiltinOpImpls() {
    static bool ops_are_registered = false;
    if (ops_are_registered) {
        return;
    }
    ops_are_registered = true;

    // onnx op's default domain is ""
    // A
    RegisterOptKernelCreator<AbsOp>("", "Abs", 6, 16);
    RegisterOptKernelCreator<AcosOp>("", "Acos", 7, 16);
    RegisterOptKernelCreator<AcoshOp>("", "Acosh", 7, 16);
    RegisterOptKernelCreator<AddOp>("", "Add", 7, 16);
    RegisterOptKernelCreator<AndOp>("", "And", 7, 16);
    RegisterOptKernelCreator<ArgmaxOp>("", "ArgMax", 1, 16);
    RegisterOptKernelCreator<ArgminOp>("", "ArgMin", 1, 16);
    RegisterOptKernelCreator<AsinOp>("", "Asin", 7, 16);
    RegisterOptKernelCreator<AsinhOp>("", "Asinh", 7, 16);
    RegisterOptKernelCreator<AtanOp>("", "Atan", 7, 16);
    RegisterOptKernelCreator<AtanhOp>("", "Atanh", 7, 16);
    RegisterOptKernelCreator<AveragePoolOp>("", "AveragePool", 1, 16);
    // B
    RegisterOptKernelCreator<BatchNormalizationOp>("", "BatchNormalization", 9, 13);
    RegisterOptKernelCreator<BitShiftOp>("", "BitShift", 9, 18);
    RegisterOptKernelCreator<BitwiseAndOp>("", "BitwiseAnd", 9, 18);
    RegisterOptKernelCreator<BitwiseNotOp>("", "BitwiseNot", 9, 18);
    RegisterOptKernelCreator<BitwiseOrOp>("", "BitwiseOr", 9, 18);
    RegisterOptKernelCreator<BitwiseXorOp>("", "BitwiseXor", 9, 18);
    RegisterOptKernelCreator<BitShiftOp>("com.metax-tech", "BitShift", 9, 18);
    RegisterOptKernelCreator<BitwiseAndOp>("com.metax-tech", "BitwiseAnd", 9, 18);
    RegisterOptKernelCreator<BitwiseNotOp>("com.metax-tech", "BitwiseNot", 9, 18);
    RegisterOptKernelCreator<BitwiseOrOp>("com.metax-tech", "BitwiseOr", 9, 18);
    RegisterOptKernelCreator<BitwiseXorOp>("com.metax-tech", "BitwiseXor", 9, 18);
    // C
    RegisterOptKernelCreator<CastOp>("", "Cast", 9, 16);
    RegisterOptKernelCreator<CeilOp>("", "Ceil", 6, 18);
    RegisterOptKernelCreator<CeluOp>("", "Celu", 1, 16);
    RegisterOptKernelCreator<ClipOp>("", "Clip", 6, 16);
    RegisterOptKernelCreator<ConcatOp>("", "Concat", 4, 16);
    RegisterOptKernelCreator<ConstantOfShapeOp>("", "ConstantOfShape", 9, 16);
    RegisterOptKernelCreator<ConvOp>("", "Conv", 1, 16);
    RegisterOptKernelCreator<ConvTransposeOp>("", "ConvTranspose", 1, 16);
    RegisterOptKernelCreator<CosOp>("", "Cos", 7, 16);
    RegisterOptKernelCreator<CoshOp>("", "Cosh", 7, 16);
    RegisterOptKernelCreator<CumSumOp>("", "CumSum", 11, 16);
    // D
    RegisterOptKernelCreator<DepthToSpaceOp>("", "DepthToSpace", 1, 16);
    RegisterOptKernelCreator<DeformConvOp>("", "DeformConv", 1, 19);
    RegisterOptKernelCreator<DivOp>("", "Div", 7, 16);

    // E
    RegisterOptKernelCreator<EinSumOp>("", "Einsum", 12, 16);
    RegisterOptKernelCreator<EluOp>("", "Elu", 1, 16);
    RegisterOptKernelCreator<EqualOp>("", "Equal", 7, 16);
    RegisterOptKernelCreator<ErfOp>("", "Erf", 9, 16);
    RegisterOptKernelCreator<ExpOp>("", "Exp", 6, 16);
    RegisterOptKernelCreator<ExpandOp>("", "Expand", 8, 16);
    RegisterOptKernelCreator<EyeLikeOp>("", "EyeLike", 9, 16);
    // F
    RegisterOptKernelCreator<FlattenOp>("", "Flatten", 1, 16);
    RegisterOptKernelCreator<FloorOp>("", "Floor", 6, 16);
    // G
    RegisterOptKernelCreator<GatherOp>("", "Gather", 1, 16);
    RegisterOptKernelCreator<GatherElementsOp>("", "GatherElements", 1, 16);
    RegisterOptKernelCreator<GatherNDOp>("", "GatherND", 11, 16);
    RegisterOptKernelCreator<GeluOp>("com.microsoft", "Gelu", 1, 1);
    RegisterOptKernelCreator<GemmOp>("", "Gemm", 7, 16);
    RegisterOptKernelCreator<GlobalAveragePoolOp>("", "GlobalAveragePool", 1, 16);
    RegisterOptKernelCreator<GlobalMaxPoolOp>("", "GlobalMaxPool", 1, 16);
    RegisterOptKernelCreator<GreaterOp>("", "Greater", 9, 16);
    RegisterOptKernelCreator<GreaterOrEqualOp>("", "GreaterOrEqual", 9, 16);
    RegisterOptKernelCreator<GroupNormalizationOp>("com.metax-tech", "GroupNormalization", 1, 1);
    // H
    RegisterOptKernelCreator<HardSigmoidOp>("", "HardSigmoid", 6, 16);
    RegisterOptKernelCreator<HardMishOp>("com.metax-tech", "HardMish", 1, 16);
    RegisterOptKernelCreator<HardSwishOp>("com.metax-tech", "HardSwish", 1, 16);
    RegisterOptKernelCreator<HardSwishOp>("", "HardSwish", 1, 16);
    // I
    RegisterOptKernelCreator<IdentityOp>("", "Identity", 1, 16);
    RegisterOptKernelCreator<IfOp>("", "If", 1, 13);
    RegisterOptKernelCreator<InstanceNormalizationOp>("", "InstanceNormalization", 6, 16);
    RegisterOptKernelCreator<IsinfOp>("", "IsInf", 1, 16);
    RegisterOptKernelCreator<IsnanOp>("", "IsNaN", 1, 16);
    // L
    RegisterOptKernelCreator<LayerNormOp>("com.metax-tech", "LayerNorm", 1, 16);
    RegisterOptKernelCreator<LayerNormOp>("", "LayerNormalization", 1, 18);
    RegisterOptKernelCreator<LeakyReluOp>("", "LeakyRelu", 6, 16);
    RegisterOptKernelCreator<LessOp>("", "Less", 9, 16);
    RegisterOptKernelCreator<LogOp>("", "Log", 6, 18);
    RegisterOptKernelCreator<LoopOp>("", "Loop", 1, 13);
    RegisterOptKernelCreator<LogSoftmaxOp>("", "LogSoftmax", 1, 13);
    RegisterOptKernelCreator<LstmOp>("", "LSTM", 7, 13);
    // M
    RegisterOptKernelCreator<MatMulOp>("", "MatMul", 9, 16);
    RegisterOptKernelCreator<MaxOp>("", "Max", 6, 16);
    RegisterOptKernelCreator<MaxPoolOp>("", "MaxPool", 1, 16);
    RegisterOptKernelCreator<MaxUnPoolOp>("", "MaxUnpool", 9, 16);
    RegisterOptKernelCreator<ModOp>("", "Mod", 10, 13);
    RegisterOptKernelCreator<MinOp>("", "Min", 6, 16);
    RegisterOptKernelCreator<MishOp>("com.metax-tech", "Mish", 1, 16);
    RegisterOptKernelCreator<MulOp>("", "Mul", 7, 16);
    // N
    RegisterOptKernelCreator<NegOp>("", "Neg", 1, 16);
    RegisterOptKernelCreator<NonMaxSupressionOp>("", "NonMaxSuppression", 10, 16);
    RegisterOptKernelCreator<NonZeroOp>("", "NonZero", 9, 12);
    RegisterOptKernelCreator<NotOp>("", "Not", 1, 16);
    // O
    RegisterOptKernelCreator<OneHotOp>("", "OneHot", 9, 16);
    RegisterOptKernelCreator<OrOp>("", "Or", 7, 16);

    // P
    RegisterOptKernelCreator<PadOp>("", "Pad", 2, 16);
    RegisterOptKernelCreator<PowOp>("", "Pow", 7, 16);
    RegisterOptKernelCreator<PReluOp>("", "PRelu", 6, 16);

    // Q
    RegisterOptKernelCreator<QuickGeluOp>("com.microsoft", "QuickGelu", 1, 1);

    // R
    RegisterOptKernelCreator<RandomNormalOp>("", "RandomNormal", 1, 16);
    RegisterOptKernelCreator<RandomNormalLikeOp>("", "RandomNormalLike", 1, 16);
    RegisterOptKernelCreator<RandomUniformOp>("", "RandomUniform", 1, 16);
    RegisterOptKernelCreator<RandomUniformLikeOp>("", "RandomUniformLike", 1, 16);
    RegisterOptKernelCreator<RangeOp>("", "Range", 11, 16);
    RegisterOptKernelCreator<ReciprocalOp>("", "Reciprocal", 6, 16);
    RegisterOptKernelCreator<ReduceL2Op>("", "ReduceL2", 1, 16);
    RegisterOptKernelCreator<ReduceOp>("", "ReduceMax", 1, 16);
    RegisterOptKernelCreator<ReduceOp>("", "ReduceMean", 1, 16);
    RegisterOptKernelCreator<ReduceOp>("", "ReduceMin", 1, 16);
    RegisterOptKernelCreator<ReduceOp>("", "ReduceProd", 1, 16);
    RegisterOptKernelCreator<ReduceOp>("", "ReduceSum", 1, 16);
    RegisterOptKernelCreator<ReluOp>("", "Relu", 6, 14);
    RegisterOptKernelCreator<ReshapeOp>("", "Reshape", 5, 16);
    RegisterOptKernelCreator<ResizeOp>("", "Resize", 11, 16); // resize 13 input size from 1-4 not 3-4
    RegisterOptKernelCreator<ROIAlignOp>("", "RoiAlign", 10, 15);
    RegisterOptKernelCreator<RoundOp>("", "Round", 11, 16);
    // S
    RegisterOptKernelCreator<ScatterElementsOp>("", "ScatterElements", 11, 13);
    RegisterOptKernelCreator<ScatterNDOp>("", "ScatterND", 11, 15);
    RegisterOptKernelCreator<SequenceAtOp>("", "SequenceAt", 11, 16);
    RegisterOptKernelCreator<SeluOp>("", "Selu", 1, 16);
    RegisterOptKernelCreator<ShapeOp>("", "Shape", 1, 16);
    RegisterOptKernelCreator<ShrinkOp>("", "Shrink", 1, 16);
    RegisterOptKernelCreator<SigmoidOp>("", "Sigmoid", 6, 16);
    RegisterOptKernelCreator<SignOp>("", "Sign", 9, 16);
    RegisterOptKernelCreator<SinOp>("", "Sin", 1, 16);
    RegisterOptKernelCreator<SinhOp>("","Sinh",1, 16);
    RegisterOptKernelCreator<SizeOp>("","Size",1, 16);
    RegisterOptKernelCreator<SliceOp>("", "Slice", 1, 16);
    RegisterOptKernelCreator<SoftmaxOp>("", "Softmax", 1, 16);
    RegisterOptKernelCreator<SoftplusOp>("", "Softplus", 1, 16);
    RegisterOptKernelCreator<SoftsignOp>("", "Softsign", 1, 16);
    RegisterOptKernelCreator<SplitOp>("", "Split", 2, 16);
    RegisterOptKernelCreator<SplitActivationOp>("com.metax-tech", "SplitActivation", 1, 1);
    RegisterOptKernelCreator<SplitToSequenceOp>("", "SplitToSequence", 11, 16);
    RegisterOptKernelCreator<SpaceToDepthOp>("", "SpaceToDepth", 1, 16);
    RegisterOptKernelCreator<SqrtOp>("", "Sqrt", 6, 16);
    RegisterOptKernelCreator<SqueezeOp>("", "Squeeze", 1, 16);
    RegisterOptKernelCreator<SubOp>("", "Sub", 7, 16);
    RegisterOptKernelCreator<SumOp>("", "Sum", 6, 16);
    RegisterOptKernelCreator<SwishOp>("", "Swish", 1, 16);
    RegisterOptKernelCreator<SwishOp>("com.metax-tech", "Swish", 1, 16);
    // T
    RegisterOptKernelCreator<TanhOp>("", "Tanh", 6, 16);
    RegisterOptKernelCreator<ThresholdedReluOp>("", "ThresholdedRelu", 1, 16);
    RegisterOptKernelCreator<TileOp>("", "Tile", 6, 16);
    RegisterOptKernelCreator<TopKOp>("", "TopK", 11, 16);
    RegisterOptKernelCreator<TransposeOp>("", "Transpose", 1, 16);
    RegisterOptKernelCreator<TriluOp>("", "Trilu", 14, 16);
    // U
    RegisterOptKernelCreator<UnsqueezeOp>("", "Unsqueeze", 1, 16);
    // W
    RegisterOptKernelCreator<WhereOp>("", "Where", 9, 15);
    // X
    RegisterOptKernelCreator<XorOp>("", "Xor", 9, 15);

    // mmcv op domain is "mmcv"
    RegisterOptKernelCreator<MMCVGridSampleOp>("mmcv", "grid_sampler", 1, 1);
    RegisterOptKernelCreator<MMCVROIAlignOp>("mmcv", "MMCVRoiAlign", 1, 1);
    RegisterOptKernelCreator<MMCVModulatedDeformConv2dOp>("mmcv", "MMCVModulatedDeformConv2d", 1, 1);
    RegisterOptKernelCreator<MMCVNonMaxSupressionOp>("mmcv", "NonMaxSuppression", 1, 1);

    // ppl customize op domain is "pmx"
    RegisterOptKernelCreator<BridgeOp>("pmx", "Bridge", 1, 1);
    RegisterOptKernelCreator<ChannelShuffleOp>("pmx", "ChannelShuffle", 1, 1);
    RegisterOptKernelCreator<GeluOp>("pmx", "GELU", 1, 1);
    RegisterOptKernelCreator<LayerNormOp>("pmx", "LayerNorm", 1, 1);
    RegisterOptKernelCreator<MultiHeadAttnV1Op>("com.metax-tech", "MultiHeadAttentionV1", 1, 1);
    RegisterOptKernelCreator<PPLShapeOperationOp>("pmx", "Shape", 1, 1);
    RegisterOptKernelCreator<MSDeformAttnOp>("mmdeploy", "MSDeformAttn", 1, 1);
    RegisterOptKernelCreator<FusedMulAddOp>("pmx", "FusedMulAdd", 1, 1);
    RegisterOptKernelCreator<FusedAddLayerNormOp>("pmx", "FusedAddLayerNorm", 1, 1);
    RegisterOptKernelCreator<FusedGatherNDReshapeConcatOp>("pmx", "FusedGatherNDReshapeConcat", 1, 1);
    RegisterOptKernelCreator<BatchGemmOp>("pmx", "BatchGemm", 1, 1);
    RegisterOptKernelCreator<FusedGatherReshapeAddOp>("pmx", "FusedGatherReshapeAdd", 1, 1);

}

}}} // namespace ppl::nn::cuda
