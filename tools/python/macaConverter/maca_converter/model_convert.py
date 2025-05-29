# -*- coding: utf-8 -*-

import os
import onnx
from onnx import version_converter
import copy
import numpy as np
import logging
import log
import onnxruntime
import sys, getopt
import json
import argparse
import h5py
import time
import fuse
from swish_convert import merge_swish_and_hard_swish
from mish_convert import merge_mish
from hard_sigmoid_convert import merge_hard_sigmiod
from gelu_fuse import merge_gelu
import bn2conv
import values
import version_check
import operation
import copy
import tempfile

from caffe2onnx.src.load_save_model import loadcaffemodel, saveonnxmodel
from caffe2onnx.src.caffe2onnx import Caffe2Onnx

from onnxsim.onnx_simplifier import simplify
from onnx_sim.onnx_simplifier import simplify_onnx

from float16 import convert_float_to_float16
from correct_batch import correct_batch_for_opset_convert, convert_ort_type_2_np, get_data_list
from resize_convert import merge_resize
from ln_convert import merge_layernorm
from input_fp32_to_uint8 import fp32_to_uint8
from matmul2gemm import matmul_add_to_gemm
from common import MXC_CONFIG
from reducel2_convert import merge_reducel2
from fuse_mha import mha_fuse
from fuse_convtrans_add_bn import ct_add_bn_fuse
from fuse_matmul_bn import matmul_bn_to_gemm
from fuse_conv_asmd_v2 import conv_asmd_to_conv_v2
from fuse_mul_add_conv import fuse_mac
from fuse_gn import gn_fuse
from fuse_split_activation import sa_fuse
from node_naming import naming_onnx_node
from fuse_gemm_bn import gemm_bn_fuse

using_wheel = False
inputs_as_nchw = ''

support_mish = 0

#logging.basicConfig(level=logging.INFO, filename='./convert.log', filemode='w')
#logger = logging.getLogger("[MacaConverter]")

logger = log.getLogger('[MacaConverter]', log.INFO)
# file_handler = logging.FileHandler('./convert.log')
# file_handler.setLevel(logging.CRITICAL)
# logger.addHandler(file_handler)

from onnx import shape_inference, TensorProto, version_converter, numpy_helper

import argparse

valid_model_type = ['caffe', 'pytorch', 'tf-h5', 'tf-ckpt', 'tf-sm', 'tf-graph', 'darknet', 'onnx', 'paddle']

optimization_op_list = ['Max', 'Min', 'Sum', 'Mean']

############ Error Code Define #################
exit_code_normal = 0
exit_code_no_caffe_cfg_file = -1
exit_code_sm2onnx = -2
exit_code_h52onnx = -3
exit_code_ckpt2onnx = -4
exit_code_pb2onnx = -5
exit_code_no_darknet_cfg_or_weights = -6
exit_code_convert_darknet2onnx = -7
exit_code_fuse_mish = -8
exit_code_check_optimization_op = -9
exit_code_check_modify_onnx2dynamic = -10
exit_code_check_convert_gap_2_ap = -11
exit_code_model_not_exist = -12
exit_code_invalid_model_type = -13
exit_code_pytorch_no_input_shape = -14
exit_code_tensorflow_no_inputs_or_outputs = -15
exit_code_extract_sub_no_inputs_or_outputs = -16
exit_code_model_type_not_onnx = -17
#########################################################

def set_using_wheel():
   global using_wheel
   using_wheel = True

def parse_args():
   parser = argparse.ArgumentParser(description='Convert caffe/tensorflow/torch/paddle/darknet model to ONNX.')

   parser.add_argument("--model_path",
                        type=str,
                        help="Input path(model file or folder)")

   parser.add_argument("--model_type",
                        type=str,
                        help="Input model type(ex: caffe/pytorch/tf-h5/...)")

   parser.add_argument("--output",
                        type=str,
                        help="Output path(ex: ./output.onnx)")

   parser.add_argument("--op_set",
                        type=int, required=False,
                        help="Set op_set version(default: 13)")

   #for simplify
   parser.add_argument("--simplify",
                        type=int, required=False,
                        choices=[0, 1, 2],
                        default=1,
                        help="Simplify the model(0:no simplify;1:do simplify; 2:for dynamic model)")

   parser.add_argument("--simplify_hw",
                        type=str,
                        required=False,
                        default='',
                        help="When h/w is -1, you can specify h/w as you expected(together with --simplify 2)")

   #for pytorch/dynamic_paddle
   parser.add_argument("--input_shape",
                        type=str,
                        required=False,
                        default='',
                        help="Input shape for pytorch/paddle(ex: [1,3,224,224] or [1,3,224,224]/[1,3,56,56])")

   #######
   parser.add_argument("--inputs",
                        type=str,
                        required=False,
                        default='',
                        help="When do checkpoint2ONNX/graph2ONNX/onnx_sub_graph, you should specify inputs(ex: --inputs image:0)")

   parser.add_argument("--outputs",
                        type=str,
                        required=False,
                        default='',
                        help="When do checkpoint2ONNX/graph2ONNX/onnx_sub_graph, you should specify outputs(ex: --outputs predict:0)")

   #for extract sub graph
   parser.add_argument("--extract_sub",
                        type=int,
                        required=False,
                        choices=[0, 1],
                        default=0,
                        help="If set 1, the tool will extract sub graph by specify inputs/outputs")

   #for dynamic batch size
   parser.add_argument("--dynamic_batch",
                        type=int,
                        required=False,
                        default=1,
                        choices=[0, 1],
                        help="If set 1, the tool will convert batch size to -1")

   #for fp32-->fp16
   parser.add_argument("--fp32_to_fp16",
                        type=int,
                        required=False,
                        default=0,
                        choices=[0, 1],
                        help="If set 1, the tool will convert fp32 to fp16 in the model")

   #for paddle dynamic model or pytorch
   parser.add_argument("--model_def_file",
                        type=str,
                        required=False,
                        default='',
                        help="Paddle/pytorch model definition file location(ex: --model_def_file ./cnn.py)")

   parser.add_argument("--model_weights_file",
                        type=str,
                        required=False,
                        default='',
                        help="Paddle/pytorch model weights file location(ex: --model_weights_file ./0.99667.pth)")

   parser.add_argument("--model_class_name",
                        type=str,
                        required=False,
                        default='',
                        help="Paddle/pytorch model calss name(ex: --model_class_name CNN)")

   parser.add_argument("--model_input_type",
                        type=str,
                        required=False,
                        #choices=['float', 'float32', 'float16', 'uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32', 'uint64', 'int64', 'bool'],
                        default='',
                        help="Paddle/pytorch input type(default float, choice is ['float', 'float32', 'float16', 'uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32', 'uint64', 'int64', 'bool'])")

   parser.add_argument("--params_file",
                        type=str,
                        required=False,
                        default='',
                        help="Paddle/pytorch params declaration file location(ex: --params_file ./params.py)")

   #for tensorflow
   parser.add_argument("--inputs_as_nchw",
                        type=str,
                        required=False,
                        default='',
                        help="When some input of tensorflow model is nhwc, you can use it(ex: --inputs_as_nchw image:0) to convert to nchw")

   #for pad+pool fuse
   parser.add_argument("--fuse_pad_pool",
                        type=int,
                        required=False,
                        default=1,
                        choices=[0, 1],
                        help="If set 1, the tool will fuse Pad into Pool")

   #for convert BN to GroupConv(1x1)
   parser.add_argument("--bn_to_conv",
                        type=int,
                        required=False,
                        choices=[0, 1],
                        default=1,
                        help="If set 1, the tool will convert BN to group 1x1_Conv")

   #for pytorch/paddle
   parser.add_argument("--output_num",
                        type=int,
                        required=False,
                        default=1,
                        help="If output num of pytorch model > 1, you can specify it by --output_num")

   #for pytorch
   parser.add_argument("--keep_batch",
                        type=int,
                        choices=[0, 1],
                        required=False,
                        default=1,
                        help="For pytorch, if set 1, the tool will keep model batch size(if 0, set it to dynamic(-1))")

   #for convert Reshape+Expand+Reshape to Resize
   parser.add_argument("--expand_to_resize",
                        type=int,
                        required=False,
                        choices=[0, 1],
                        default=1,
                        help="If set 1, the tool will convert Reshape+Expand+Reshape to Resize")

   #reset model value_info(some model(batch=-1) may have wrong value info for middle node))
   parser.add_argument("--reset_value_info",
                        type=int,
                        required=False,
                        choices=[0, 1],
                        default=0,
                        help="If set 1, the tool will try correct wrong value info")

   #fuse match ops to LayerNorm
   parser.add_argument("--fuse_layernorm",
                        type=int,
                        required=False,
                        choices=[0, 1],
                        default=1,
                        help="If set 1, the tool will fuse match ops to LayerNorm")

   #reset model value_info(some model(batch=-1) may have wrong value info for middle node))
   parser.add_argument("--reset_batch",
                        type=str,
                        required=False,
                        nargs='*',
                        default='', #should be 'input_batch,output_batch'
                        help="If set 1, the tool will try reset model batch_size")

   #fuse Gelu
   parser.add_argument("--fuse_gelu",
                        type=int,
                        required=False,
                        default=1,
                        choices=[0, 1],
                        help="If set 1, the tool will fuse Gelu")

   #Disable all optimization
   parser.add_argument("--disable_all_optimizer",
                        type=int,
                        required=False,
                        default=0,
                        choices=[0, 1],
                        help="If set 1, the tool will force all optimization value to 0")

   #fp32-->u8(for input type)
   parser.add_argument("--fp32_to_u8",
                        type=int,
                        required=False,
                        default=0,
                        help="If set 1, the tool will change input type from float to uint8")

   #for merge mish
   parser.add_argument("--support_mish",
                        type=int,
                        required=False,
                        default=1,
                        choices=[0, 1],
                        help="If set 1, the tool will fuse Softplus+Tanh+Mul to Mish")

   #for merge swish
   parser.add_argument("--support_swish",
                        type=int,
                        required=False,
                        choices=[0, 1],
                        default=1,
                        help="If set 1, the tool will convert Sigmoid+Mul to Swish; HardSigmoid+Mul to HardSwish")

   #fuse match ops to ReduceL2
   parser.add_argument("--fuse_reducel2",
                        type=int,
                        required=False,
                        choices=[0, 1],
                        default=1,
                        help="If set 1, the tool will fuse match ops to ReduceL2")

   #convert Add+Clip+Div to HardSigmoid
   parser.add_argument("--fuse_hard_sigmoid",
                        type=int,
                        required=False,
                        default=1,
                        choices=[0, 1],
                        help="If set 1, the tool will merge Add+Cilp+Div to HardSigmoid")


   #matmul+add-->gemm
   parser.add_argument("--matmul_to_gemm",
                        type=int,
                        required=False,
                        default=1,
                        choices=[0, 1],
                        help="If set 1, the tool will fuse Matmul+Add to Gemm")

   #fuse mha
   parser.add_argument("--fuse_mha",
                        type=int,
                        required=False,
                        choices=[0, 1],
                        default=1,
                        help="If set 1, the tool will fuse mha")

   #fuse convtranspose+add+bn
   parser.add_argument("--fuse_ct_add_bn",
                        type=int,
                        required=False,
                        choices=[0, 1],
                        default=1,
                        help="If set 1, the tool will fuse ConvTranspose+Add+BN to ConvTranspose")

   #fuse matmul+reshape+bn+reshape
   parser.add_argument("--fuse_matmul_bn",
                        type=int,
                        required=False,
                        choices=[0, 1],
                        default=1,
                        help="If set 1, the tool will fuse MatMul+Reshape+BN+Reshape to Gemm")

   #fuse conv+add/sub/mul/div
   parser.add_argument("--fuse_conv_asmd",
                        type=int,
                        required=False,
                        choices=[0, 1],
                        default=1,
                        help="If set 1, the tool will fuse Conv+Add/Sub/Mul/Div to Conv(when Add/Sub/Mul/Div value is constant)")

   #fuse mul+add+conv
   parser.add_argument("--fuse_mul_add_conv",
                        type=int,
                        required=False,
                        choices=[0, 1],
                        default=0,
                        help="If set 1, the tool will fuse Mul+Add+Conv to Conv(when Mul/Add value is constant(scalar or shape is [1]) and conv_weight.shape is NxCx1x1)")

   #for caffe pooling
   parser.add_argument("--ceil_floor_reverse",
                        type=int,
                        required=False,
                        choices=[0, 1],
                        default=0,
                        help="If set 1, the tool will set ceil=1 in caffe pooling")

   #fuse Reshape+InstanceNormalization+Reshape
   parser.add_argument("--fuse_gn",
                        type=int,
                        required=False,
                        choices=[0, 1],
                        default=1,
                        help="If set 1, the tool will fuse Reshape+InstanceNormalization+Reshape to GroupNormalization")

   #fuse Slice+Gelu+Mul
   parser.add_argument("--fuse_sa",
                        type=int,
                        required=False,
                        choices=[0, 1],
                        default=1,
                        help="If set 1, the tool will fuse Slice+Gelu+Mul to SplitActivation")

   #fuse Gemm+BatchNormalization
   parser.add_argument("--fuse_gemm_bn",
                        type=int,
                        required=False,
                        choices=[0, 1],
                        default=1,
                        help="If set 1, the tool will fuse Gemm+BatchNormalization to Gemm")

   #add name for node if node name is null
   parser.add_argument("--add_node_name",
                        type=int,
                        required=False,
                        choices=[0, 1],
                        default=1,
                        help="If set 1, the tool will add name for node if node name is null")

   # simplify is to skip constant folding
   parser.add_argument("--simplify_skip_constant_folding",
                        type=int,
                        required=False,
                        choices=[0, 1],
                        default=0,
                        help="If set 1, the tool will skip constant folding when simplify")

   # simplify is to skip optimization
   parser.add_argument("--simplify_skip_optimization",
                        type=int,
                        required=False,
                        choices=[0, 1],
                        default=0,
                        help="If set 1, the tool will skip optimization when simplify")

   #show version
   parser.add_argument('--version', '-v',
                        action='store_true',
                        default=False,
                        help='Show current version')

   args = parser.parse_args()

   return args

def get_caffe_files(model_path):
   items = os.listdir(model_path)
   prototxt_cnt = 0
   caffemodel_cnt = 0
   prototxt_file = ''
   caffemodel_file = ''

   for f in items:
      if f.endswith(".prototxt"):
         prototxt_cnt = prototxt_cnt + 1
         prototxt_file = f
      elif f.endswith(".caffemodel"):
         caffemodel_cnt = caffemodel_cnt + 1
         caffemodel_file = f

   if prototxt_cnt == 1 and caffemodel_cnt == 1:
      if model_path.endswith("/"):
         prototxt_file = model_path + prototxt_file
         caffemodel_file = model_path + caffemodel_file
      else:
         prototxt_file = model_path + '/' + prototxt_file
         caffemodel_file = model_path + '/' + caffemodel_file
      logger.info('got prototxt_file:{}, caffemodel_file:{}'.format(prototxt_file, caffemodel_file))
   elif prototxt_cnt > 1 or caffemodel_cnt > 1:
      prototxt_file = ''
      caffemodel_file = ''
      logger.error('ERROR: prototxt_cnt > 1 or caffemodel_cnt > 1')
   elif prototxt_cnt == 0 or caffemodel_cnt == 0:
      prototxt_file = ''
      caffemodel_file = ''
      logger.error('ERROR: No .prototxt file or no .caffemodel file')

   return prototxt_file, caffemodel_file

def convert_caffe2onnx(model_path, output, op_set, ceil_floor_reverse):
      logger.info('Begin converting caffe to onnx...')
      prototxt_file, caffemodel_file = get_caffe_files(model_path)

      if prototxt_file == '' or caffemodel_file == '':
         sys.exit(exit_code_no_caffe_cfg_file)

      onnxmodel_path = output

      graph, params = loadcaffemodel(prototxt_file, caffemodel_file)
      c2o = Caffe2Onnx(graph, params, onnxmodel_path, op_set, ceil_floor_reverse)
      onnxmodel = c2o.createOnnxModel(op_set) #qiuzy debug

      saveonnxmodel(onnxmodel, onnxmodel_path)

def convert_sm2onnx(model_path, output, op_set, **kwargs):
      logger.info('Begin converting tf-savemodel to onnx...')

      try:
         import tensorflow
      except Exception as e:
         print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
         print(e)
         print('Please install tensorflow(pip install tensorflow==2.4.0)')
         print('If numpy version > 1.19.5, tensorflow version should be 2.7.4')
         print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
         sys.exit(exit_code_sm2onnx)

      version_check.check('tensorflow', tensorflow.__version__)

      if using_wheel == False:
         cmd = 'python -m tf2onnx.convert --saved-model ' + model_path + ' --opset ' + str(op_set) + ' --output ' + output
      else:
         cmd = 'python -m maca_converter.tf2onnx.convert --saved-model ' + model_path + ' --opset ' + str(op_set) + ' --output ' + output

      if inputs_as_nchw != '':
         cmd += ' --inputs-as-nchw ' + inputs_as_nchw
      if kwargs.get('inputs', ''):
         cmd += f" --inputs {kwargs.get('inputs', '')}"
      if kwargs.get('outputs', ''):
         cmd += f" --outputs {kwargs.get('outputs', '')}"

      logger.info('convert_tfsm2onnx: {}'.format(cmd))
      r = os.system(cmd)
      if r != 0:
         logger.error('ERROR: convert_sm2onnx failed')
         sys.exit(exit_code_sm2onnx)

def convert_h52onnx(model_path, output, op_set):
      logger.info('Begin converting tf-savemodel to onnx...')

      try:
         import tensorflow
      except Exception as e:
         print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
         print(e)
         print('Please install tensorflow(pip install tensorflow==2.4.0)')
         print('If numpy version > 1.19.5, tensorflow version should be 2.7.4')
         print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
         sys.exit(exit_code_h52onnx)

      version_check.check('tensorflow', tensorflow.__version__)

      if using_wheel == False:
         cmd = 'python -m tf2onnx.convert --keras ' + model_path + ' --opset ' + str(op_set) + ' --output ' + output
      else:
         cmd = 'python -m maca_converter.tf2onnx.convert --keras ' + model_path + ' --opset ' + str(op_set) + ' --output ' + output

      if inputs_as_nchw != '':
         cmd += ' --inputs-as-nchw ' + inputs_as_nchw
      logger.info('convert_tfh52onnx: {}'.format(cmd))
      r = os.system(cmd)
      if r != 0:
         logger.error('ERROR: convert_h52onnx failed')
         sys.exit(exit_code_h52onnx)

def convert_ckpt2onnx(model_path, output, op_set, inputs, outputs):
      logger.info('Begin converting tf-ckpt to onnx...')

      try:
         import tensorflow
      except Exception as e:
         print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
         print(e)
         print('Please install tensorflow(pip install tensorflow==2.4.0)')
         print('If numpy version > 1.19.5, tensorflow version should be 2.7.4')
         print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
         sys.exit(exit_code_ckpt2onnx)

      version_check.check('tensorflow', tensorflow.__version__)

      if using_wheel == False:
         cmd = 'python -m tf2onnx.convert --checkpoint ' + model_path + ' --opset ' + str(op_set) + ' --output ' + output \
               + ' --inputs '  + inputs + ' --outputs ' + outputs
      else:
         cmd = 'python -m maca_converter.tf2onnx.convert --checkpoint ' + model_path + ' --opset ' + str(op_set) + ' --output ' + output \
               + ' --inputs '  + inputs + ' --outputs ' + outputs
      if inputs_as_nchw != '':
         cmd += ' --inputs-as-nchw ' + inputs_as_nchw

      logger.info('convert_ckpt2onnx: {}'.format(cmd))

      r = os.system(cmd)
      if r != 0:
         logger.error('ERROR: convert_ckpt2onnx failed')
         sys.exit(exit_code_ckpt2onnx)

def convert_graph2onnx(model_path, output, op_set, inputs, outputs):
      logger.info('Begin converting tf-graph to onnx...')

      try:
         import tensorflow
      except Exception as e:
         print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
         print(e)
         print('Please install tensorflow(pip install tensorflow==2.4.0)')
         print('If numpy version > 1.19.5, tensorflow version should be 2.7.4')
         print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
         sys.exit(exit_code_pb2onnx)

      version_check.check('tensorflow', tensorflow.__version__)

      if using_wheel == False:
         cmd = 'python -m tf2onnx.convert --graphdef ' + model_path + ' --opset ' + str(op_set) + ' --output ' + output \
              + ' --inputs '  + inputs + ' --outputs ' + outputs
      else:
         cmd = 'python -m maca_converter.tf2onnx.convert --graphdef ' + model_path + ' --opset ' + str(op_set) + ' --output ' + output \
              + ' --inputs '  + inputs + ' --outputs ' + outputs

      if inputs_as_nchw != '':
         cmd += ' --inputs-as-nchw ' + inputs_as_nchw

      logger.info('convert_graph2onnx: {}'.format(cmd))

      r = os.system(cmd)
      if r != 0:
         logger.error('ERROR: convert_graph2onnx failed')
         sys.exit(exit_code_pb2onnx)

def get_darknet_files(model_path):
   items = os.listdir(model_path)
   cfg_cnt = 0
   weights_cnt = 0
   cfg_file = ''
   weights_file = ''

   for f in items:
      if f.endswith(".cfg"):
         cfg_cnt = cfg_cnt + 1
         cfg_file = f
      elif f.endswith(".weights"):
         weights_cnt = weights_cnt + 1
         weights_file = f

   if cfg_cnt == 1 and weights_cnt == 1:
      if model_path.endswith("/"):
         cfg_file = model_path + cfg_file
         weights_file = model_path + weights_file
      else:
         cfg_file = model_path + '/' + cfg_file
         weights_file = model_path + '/' + weights_file

      logger.info('got cfg_file:{}, weights_file:{}'.format(cfg_file, weights_file))
   elif cfg_cnt > 1 or weights_cnt > 1:
      cfg_file = ''
      weights_file = ''
      logger.error('ERROR: cfg_cnt > 1 or weights_cnt > 1')
   elif cfg_cnt == 0 or weights_cnt == 0:
      cfg_file = ''
      weights_file = ''
      logger.error('ERROR: No .cfg file or no .weights file')

   return cfg_file, weights_file

def convert_dn2onnx(model_path, output, op_set):
   global support_mish
   logger.info('Begin converting darknet to onnx...... support_mish: {}'.format(support_mish))
   cfg_file, weights_file = get_darknet_files(model_path)
   if cfg_file == '' or weights_file == '':
      sys.exit(exit_code_no_darknet_cfg_or_weights)

   if using_wheel == False:
      cmd = 'python ./darknet2onnx.py --cfg_file ' + cfg_file + ' --weights_file ' + weights_file + ' --output_file ' + output + ' --support_mish ' + str(support_mish)
   else:
      cmd = 'python -m maca_converter.darknet2onnx --cfg_file ' + cfg_file + ' --weights_file ' + weights_file + ' --output_file ' + output + ' --support_mish ' + str(support_mish)

   if '-tiny' in cfg_file or '-tiny' in weights_file:
      if using_wheel == False:
         cmd = 'python ./darknet2onnx.py --cfg_file ' + cfg_file + ' --weights_file ' + weights_file + ' --strides 32 16 8 ' + ' --neck FPN ' + ' --output_file ' + output + ' --support_mish ' + str(support_mish)
      else:
         cmd = 'python -m maca_converter.darknet2onnx --cfg_file ' + cfg_file + ' --weights_file ' + weights_file + ' --strides 32 16 8 ' + ' --neck FPN ' + ' --output_file ' + output + ' --support_mish ' + str(support_mish)
   elif 'yolov3' in cfg_file or 'yolov3' in weights_file:
      if using_wheel == False:
         cmd = 'python ./darknet2onnx.py --cfg_file ' + cfg_file + ' --weights_file ' + weights_file + ' --strides 32 16 8 ' + ' --neck FPN ' + ' --output_file ' + output
      else:
         cmd = 'python -m maca_converter.darknet2onnx --cfg_file ' + cfg_file + ' --weights_file ' + weights_file + ' --strides 32 16 8 ' + ' --neck FPN ' + ' --output_file ' + output
   logger.info('convert_dn2onnx: {}'.format(cmd))

   r = os.system(cmd)
   if r != 0:
      logger.error('ERROR: convert_dn2onnx failed')
      sys.exit(exit_code_convert_darknet2onnx)

def convert_mish(model_path, output, op_set):
   global support_mish
   logger.info('Begin converting mish')

   if using_wheel == False:
      cmd = 'python ./mish_convert.py --onnx_file ' + model_path + ' --output_file ' + output
   else:
      cmd = 'python -m maca_converter.mish_convert --onnx_file ' + model_path + ' --output_file ' + output

   logger.info('convert_mish: {}'.format(cmd))
   r = os.system(cmd)
   if r != 0:
      logger.error('ERROR: convert_mish failed')
      sys.exit(exit_code_fuse_mish)

def convert(model_path, model_type, output, op_set, input_shape_list, inputs, outputs,
               model_def_file,
               model_class_name,
               model_input_type,
               model_weights_file,
               output_num,
               keep_batch,
               params_file,
               ceil_floor_reverse):

   if model_type == 'caffe':
      convert_caffe2onnx(model_path, output, op_set, ceil_floor_reverse)

   if model_type == 'tf-sm':
      convert_sm2onnx(model_path, output, op_set, inputs=inputs, outputs=outputs)

   if model_type == 'tf-h5':
      convert_h52onnx(model_path, output, op_set)

   if model_type == 'tf-ckpt':
      convert_ckpt2onnx(model_path, output, op_set, inputs, outputs)

   if model_type == 'tf-graph':
      convert_graph2onnx(model_path, output, op_set, inputs, outputs)

   if model_type == 'darknet':
      convert_dn2onnx(model_path, output, op_set)

   if model_type == 'pytorch':
      #convert_pt2onnx(model_path, output, op_set, input_shape)
      from pt2onnx import convert_pt2onnx
      convert_pt2onnx(model_path, output, op_set, input_shape_list,
                           model_def_file, model_class_name, model_weights_file, output_num, model_input_type, keep_batch, params_file)

   if model_type == 'paddle':
      from pd2onnx import convert_pd2onnx
      convert_pd2onnx(model_path, output, op_set, input_shape_list, model_def_file, model_class_name, model_input_type, model_weights_file)

def optimization_op(model):
   #model = onnx.load(onnxfile)

   delete_node_id = 0
   delete = False
   #export_onnx = onnxfile

   for node_id, node in enumerate(model.graph.node):
      #print(node_id, ", name:", node.name, ", input:", node.input, ", output:", node.output,  \
      #         ", op:", node.op_type, ', len(input):', len(node.input))

      if node.op_type in optimization_op_list and len(node.input) == 1:
         delete_node_id = node_id
         delete = True
         break

   if delete == True:
      log.debug('delete: {}'.format(delete_node_id))
      delete_node = model.graph.node[delete_node_id]
      log.debug('delete node op: {}'.format(delete_node.op_type))
      next_node = model.graph.node[delete_node_id+1]

      for i, n_ in enumerate(next_node.input):
         #print('next input:', n_)
         if n_ == delete_node.output[0]:
               log.debug('got it: {]}'.format(n_))
               next_node.input[i] = delete_node.input[0]

      model.graph.node.remove(delete_node)
      export_onnx = onnxfile

      try:
         onnx.checker.check_model(model)
      except onnx.checker.ValidationError as e:
         logger.warning('The model cannot be saved for: {}'.format(e))
         if 'No Op registered for Mish' in str(e):
               logger.warning('ignore mish warning, continue saving~')
         else:
               logger.error('ERROR: check model failed')
               sys.exit(exit_code_check_optimization_op)
      else:
         logger.info('---Begin saving model...')

      ###################
      #onnx.checker.check_model(model)
      #onnx.save(model, export_onnx)

   return delete

def correct_output_shape(model):
   for output in model.graph.output:
      if len(output.type.tensor_type.shape.dim) > 0:
         output_shape = output.type.tensor_type.shape.dim
         output_shape = [x.dim_value for x in output_shape]

         dynamic_output_shape_ = any(d==-1 or d==0 for d in output_shape)
         if dynamic_output_shape_ == True:
            logger.info('The model output is dynamic, output: {} {}'.format(output.name, output_shape))
            output.type.tensor_type.shape.dim[0].dim_value = 1

   for output in model.graph.output:
      if len(output.type.tensor_type.shape.dim) > 0:
         output_shape = output.type.tensor_type.shape.dim
         output_shape = [x.dim_value for x in output_shape]
         logger.debug('The model output is dynamic, output_shape: {}'.format(output_shape))

def reset_model_value_info(model):
   model_bak = copy.deepcopy(model)

   del model_bak.graph.value_info[:]

   try:
      new_model = onnx.shape_inference.infer_shapes(model_bak)
   except BaseException as e:
      logger.warning('reset_model_value_info, the model cannot be inferenced for: {}'.format(e))
      return model
   else:
      new_model = onnx.shape_inference.infer_shapes(model_bak)
      new_model = onnx.shape_inference.infer_shapes(new_model)
      return new_model

def reset_batch_size(model, input_batch, output_batch):
   for input_ in model.graph.input:
      if len(input_.type.tensor_type.shape.dim) > 0:
         dim_proto = input_.type.tensor_type.shape.dim[0]
         dim_proto.dim_value = input_batch

   for output_ in model.graph.output:
      if len(output_.type.tensor_type.shape.dim) > 0:
         dim_proto = output_.type.tensor_type.shape.dim[0]
         dim_proto.dim_value = output_batch

   del model.graph.value_info[:]

   try:
      new_model = onnx.shape_inference.infer_shapes(model)
   except BaseException as e:
      logger.warning('reset_batch_size, the model cannot be inferenced for: {}'.format(e))
      new_model = model
   else:
      new_model = onnx.shape_inference.infer_shapes(new_model)

   return new_model

def model_simplify(onnx_model, simplify_model, simplify_hw, skip_constant_folding, perform_optimization):
   #onnx_model = onnx.load(model_path)
   is_dynamic_input_shape = False

   init_list = []
   input_shapes_ = {}
   model_proto = onnx.load(onnx_model) if isinstance(onnx_model, str) else onnx_model
   for init in model_proto.graph.initializer:
      init_list.append(init.name)

   for input_ in model_proto.graph.input:
      #print('graph_input_name:', input_.name)
      if input_.name not in init_list:
         if len(input_.type.tensor_type.shape.dim) > 0:
            input_shape = input_.type.tensor_type.shape.dim
            input_shape = [x.dim_value for x in input_shape]

            if len(input_shape) < 2:
               if input_shape[0] <= 0:
                  is_dynamic_input_shape = True
                  input_shapes_[input_.name] = [1]
               continue

            dynamic_input_shape_ = any(d==-1 or d==0 for d in input_shape)
            if dynamic_input_shape_ == True:
               is_dynamic_input_shape = True
               logger.info('The model input is dynamic, input: {} {}'.format(input_.name, input_shape))
               #input_shape[0] = 1
               in_shape = [s if s > 0 else 1 for s in input_shape]
               if simplify_hw != '':
                  hw_list = simplify_hw.split(',')
                  in_shape[-1] = int(hw_list[1])
                  in_shape[-2] = int(hw_list[0])
                  logger.debug('in_shape: {}'.format(in_shape))

               input_shapes_[input_.name] = in_shape #input_shape
               #break

            '''
            dim_proto_input = input_.type.tensor_type.shape.dim[0]
            if dim_proto_input.dim_value == -1 or dim_proto_input.dim_value == 0:
               print('The model input is dynamic~~~~~~')
               dynamic_input_shape_ = True
               input_shape[0] = 1
               input_shapes_[input_.name] = input_shape
               break
            '''

   if simplify_model == 2:
      if is_dynamic_input_shape == True:
         try:
            model_simp, check = simplify(onnx_model, input_shapes=input_shapes_, skip_constant_folding=skip_constant_folding, perform_optimization=perform_optimization)
         except Exception as e:
            logger.warning('using new onnxsim failed(--simplify 2), try using old version now...')
            model_simp, check = simplify_onnx(onnx_model, input_shapes=input_shapes_, skip_constant_folding=skip_constant_folding, perform_optimization=perform_optimization)

         if simplify_hw == '':
            #correct_batch_for_opset_convert(model_simp)
            correct_output_shape(model_simp)
            try:
               model_simp = reset_model_value_info(model_simp)
            except Exception as e:
               logger.warning('Cannot do reset_value operation~')
      else:
         try:
            model_simp, check = simplify(onnx_model, dynamic_input_shape=False, perform_optimization=perform_optimization)
         except Exception as e:
            logger.warning('using new onnxsim failed(--simplify 2, dynamic_input_shape is false), try using old version now...')
            model_simp, check = simplify_onnx(onnx_model, dynamic_input_shape=False, perform_optimization=perform_optimization)
   else:
      try:
         model_simp, check = simplify(onnx_model, dynamic_input_shape=is_dynamic_input_shape, skip_constant_folding=skip_constant_folding, perform_optimization=perform_optimization)
      except Exception as e:
         logger.warning('using new onnxsim failed, try using old version now...dynamic_input_shape:{}'.format(is_dynamic_input_shape))
         model_simp, check = simplify_onnx(onnx_model, input_shapes=input_shapes_, dynamic_input_shape=is_dynamic_input_shape, skip_constant_folding=skip_constant_folding, perform_optimization=perform_optimization)

   #onnx.save(model_simp, model_path)

   return model_simp

def modify_onnx2dynamic(onnx_model):
   for idx in range(len(onnx_model.graph.input)):
      if len(onnx_model.graph.input[idx].type.tensor_type.shape.dim) > 0:
         dim_proto_input = onnx_model.graph.input[idx].type.tensor_type.shape.dim[0]
         # dim_proto_input.dim_param = 'bs'
         dim_proto_input.dim_value = -1

   for idx in range(len(onnx_model.graph.value_info)):
      if len(onnx_model.graph.value_info[idx].type.tensor_type.shape.dim) > 0:
         logger.debug('modify_onnx2dynamic, value info name: {}'.format(onnx_model.graph.value_info[idx].name))
         dim_proto_input = onnx_model.graph.value_info[idx].type.tensor_type.shape.dim[0]
         # dim_proto_input.dim_param = 'bs'
         dim_proto_input.dim_value = -1

   for idx in range(len(onnx_model.graph.output)):
      if len(onnx_model.graph.output[idx].type.tensor_type.shape.dim):
         dim_proto_output = onnx_model.graph.output[idx].type.tensor_type.shape.dim[0]
         # dim_proto_output.dim_param = 'bs'
         dim_proto_output.dim_value = -1

   ### for Reshape
   reshape_param = []
   for node_id, node in enumerate(onnx_model.graph.node):
      #print(node_id, ", name:", node.name, ", input:", node.input, ", output:", node.output,  \
      #         ", op:", node.op_type, ', len(input):', len(node.input))
      if node.op_type == 'Reshape':
         logger.debug('Reshape, input: {}'.format(node.input))
         if node.input[1] not in reshape_param:
            reshape_param.append(node.input[1])

   for n in reshape_param:
      for init in onnx_model.graph.initializer:
         logger.debug('loop init.name: {}'.format(init.name))
         if n == init.name:
            logger.debug('got it in initializer: {} {}'.format(n, init.int64_data))
            #init.int64_data[0] = -1
            dtype = init.data_type
            np_dtype = convert_ort_type_2_np(dtype)
            if init.raw_data:
               params_list = np.fromstring(init.raw_data, dtype=np_dtype)
               logger.debug('len(params_list): {}'.format(len(params_list)))
               adjust = True
               for val in params_list:
                  if val == -1:
                     adjust = False

               if adjust == True and params_list[0] != -1:
                     params_list[0] = -1
                     init.raw_data = params_list.tostring()
            else:
               data_list = get_data_list(dtype, init)
               adjust = True
               logger.debug('len(data_list): {}'.format(len(data_list)))

               for val in data_list:
                  if val == -1:
                     adjust = False

               if adjust == True and len(data_list) > 0 and data_list[0] != -1:
                     data_list[0] = -1

############# for constant node
   for n in reshape_param:
      for node in onnx_model.graph.node:
         if node.op_type == 'Constant':
            if node.output[0] == n:
               logger.info('got constant output: {}'.format(node.output))
               attributes = node.attribute
               for attr in attributes:
                     if attr.name == 'value':
                        v = values.get_tensor_value(attr.t)
                        #print('got type v:', type(v))
                        adjust = True
                        for val in v:
                           if val == -1:
                              adjust = False
                              break

                        if adjust == True:
                           v[0] = -1
                           vv = [v_ for v_ in v]
                           #print('-----new vv:', vv, type(vv))
                           if isinstance(v, np.ndarray) == True:
                              values.set_tensor_value(attr.t, v)
                           else:
                              values.set_tensor_value(attr.t, vv)
########################

   operation.adjust_resize2dynamic(onnx_model)

   #onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
   try:
      onnx.checker.check_model(onnx_model)
   except onnx.checker.ValidationError as e:
      logger.warning('{}'.format(e))
   except ValueError as e:
      logger.warning(e)
   else:
      logger.info('*** The model is modified!')

   return onnx_model

def post_process(new_model, inference_success):
   start_time = time.time()

   debug_print = False

   delete = optimization_op(new_model)
   while delete == True:
      debug_print = True
      delete = optimization_op(new_model)

   end_time1 = time.time()

   if debug_print == True:
      logger.info('optimization_op cost {} seconds'.format(end_time1 - start_time))

   noused_node_list = operation.find_noused_op(new_model)
   for noused_node in noused_node_list:
      operation.del_node_by_nodeName(new_model, noused_node)
   operation.swap_operation(new_model)


def my_extract_model(
        input_path,  # type: Text
        output_path,  # type: Text
        input_names,  # type: List[Text]
        output_names  # type: List[Text]
):  # type: (...) -> None
   """Extracts sub-model from an ONNX model.

   The sub-model is defined by the names of the input and output tensors *exactly*.

   Note: For control-flow operators, e.g. If and Loop, the _boundary of sub-model_,
   which is defined by the input and output tensors, should not _cut through_ the
   subgraph that is connected to the _main graph_ as attributes of these operators.

   Arguments:
   input_path (string): The path to original ONNX model.
   output_path (string): The path to save the extracted ONNX model.
   input_names (list of string): The names of the input tensors that to be extracted.
   output_names (list of string): The names of the output tensors that to be extracted.
   """
   if not os.path.exists(input_path):
      raise ValueError("Invalid input model path: %s" % input_path)
   if not output_path:
      raise ValueError("Output model path shall not be empty!")
   if not output_names:
      raise ValueError("Output tensor names shall not be empty!")

   #onnx.checker.check_model(input_path)
   try:
      onnx.checker.check_model(input_path)
   except onnx.checker.ValidationError as e:
      print('Extract warning:: %s' % e)
   else:
      logger.info('~~~~ Begin extracting model...')

   model = onnx.load(input_path)

   e = onnx.utils.Extractor(model)
   extracted = e.extract_model(input_names, output_names)

   onnx.save(extracted, output_path)
   #onnx.checker.check_model(output_path)
   try:
      onnx.checker.check_model(output_path)
   except onnx.checker.ValidationError as e:
      print('Extracted warning: %s' % e)
   else:
      logger.info('^^^^ Finish extracting model...')

   return True

def extract_sub_graph(input_path, output_path, input_names, output_names):
   logger.info('input_names: {}, output_names: {}'.format(input_names, output_names))
   input_list = input_names.split(',')
   output_list = output_names.split(',')
   #onnx.utils.extract_model(input_path, output_path, input_list, output_list)
   return my_extract_model(input_path, output_path, input_list, output_list)

def process(args):
   global support_mish
   global inputs_as_nchw

   model_path = args.model_path
   model_type = args.model_type
   output = args.output
   op_set = args.op_set
   input_shape = args.input_shape
   inputs = args.inputs
   outputs = args.outputs
   simplify_model = args.simplify
   extract_sub = args.extract_sub
   dynamic_batch = args.dynamic_batch
   fp32_to_fp16 = args.fp32_to_fp16
   support_mish = args.support_mish
   model_def_file = args.model_def_file
   model_class_name = args.model_class_name
   model_input_type = args.model_input_type
   model_weights_file = args.model_weights_file
   inputs_as_nchw = args.inputs_as_nchw
   fuse_pad_pool = args.fuse_pad_pool
   bn_to_conv = args.bn_to_conv
   output_num = args.output_num
   keep_batch = args.keep_batch
   params_file = args.params_file
   simplify_hw = args.simplify_hw
   expand_to_resize = args.expand_to_resize
   reset_value_info = args.reset_value_info
   fuse_layernorm = args.fuse_layernorm
   reset_batch = args.reset_batch
   fuse_gelu = args.fuse_gelu
   disable_all_optimizer = args.disable_all_optimizer
   fp32_to_u8 = args.fp32_to_u8
   matmul_to_gemm = args.matmul_to_gemm
   support_swish = args.support_swish
   fuse_hard_sigmoid = args.fuse_hard_sigmoid
   fuse_reducel2 = args.fuse_reducel2
   fuse_mha = args.fuse_mha
   fuse_ct_add_bn = args.fuse_ct_add_bn
   ceil_floor_reverse = args.ceil_floor_reverse
   fuse_matmul_bn = args.fuse_matmul_bn
   fuse_conv_asmd = args.fuse_conv_asmd
   fuse_mul_add_conv = args.fuse_mul_add_conv
   fuse_gn = args.fuse_gn
   fuse_sa = args.fuse_sa
   add_node_name = args.add_node_name
   fuse_gemm_bn = args.fuse_gemm_bn
   simplify_skip_constant_folding = args.simplify_skip_constant_folding
   simplify_skip_optimization = args.simplify_skip_optimization

   if args.version:
      print('maca_converter version:', MXC_CONFIG.VERSION)
      print('last modified:', MXC_CONFIG.LAST_MODIFIED)
      exit(0)

   if model_path == None or model_type == None or output == None:
      print('WARNING: model_path/model_type/output COULD NOT be null')
      exit(-1)

   if disable_all_optimizer == 1:
      print('------- disable all optimazation')
      support_mish = 0
      fuse_pad_pool = 0
      bn_to_conv = 0
      expand_to_resize = 0
      fuse_gelu = 0
      fuse_layernorm = 0
      matmul_to_gemm = 0
      support_swish = 0
      fuse_hard_sigmoid = 0
      fuse_reducel2 = 0
      fuse_mha = 0
      fuse_ct_add_bn = 0
      ceil_floor_reverse = 0
      fuse_matmul_bn = 0
      fuse_conv_asmd = 0
      fuse_mul_add_conv = 0
      fuse_gn = 0
      fuse_sa = 0
      fuse_gemm_bn = 0
      #add_node_name = 0

   logger.info('model_path:{}, model_type:{}, output:{}'.format(model_path, model_type, output))

   if model_type == 'tf-ckpt' or model_type == 'tf-graph' :
      logger.debug('checkpoint: {} {}'.format(inputs, outputs))

   logger.info('---input_shape: {}'.format(input_shape))

   input_shape_list = input_shape.split('/')
   logger.info('---input_shape_list: {}'.format(input_shape_list))

   #input_shape = input_shape_list[0]

   dynamic_paddle = False
   if model_type == 'paddle':
         from pd2onnx import is_dynamic_paddle
         dynamic_paddle = is_dynamic_paddle(input_shape_list, model_def_file, model_class_name, model_weights_file)
         #if dynamic_paddle == True and model_input_type == '':
         #   model_input_type = 'float32'

   can_ignore_model_path = False
   if model_type == 'pytorch':
      #if '.' in model_class_name:
      if model_weights_file != '':
         can_ignore_model_path = True

   if dynamic_paddle == False and can_ignore_model_path == False and not os.path.exists(model_path):
      logger.error('ERROR: {} is not exist'.format(model_path))
      sys.exit(exit_code_model_not_exist)

   if model_type not in valid_model_type:
      logger.error('Valid mode type is {}'.format(valid_model_type))
      logger.error('ERROR: {} is not valid mode type'.format(model_type))
      sys.exit(exit_code_invalid_model_type)

   op_set_default = 13

   if op_set != None and op_set < op_set_default:
      op_set_default = op_set

   if model_type == 'pytorch' and args.input_shape == '':
      logger.warning('WARNNIG: when converting pytorch model, you must tell the input shape(ex: --input_shape [1, 3, 32, 32])')
      logger.warning('WARNNIG: also, you should provide model definition file')
      sys.exit(exit_code_pytorch_no_input_shape)

   if (model_type == 'tf-ckpt' or model_type == 'tf-graph') and (args.inputs == '' or args.outputs == ''):
      logger.warning('WARNNIG: When converting checkpoint/graph, you must tell the inputs(ex: --inputs input0:0,input1:0) and outputs(ex: --outputs output0:0)')
      sys.exit(exit_code_tensorflow_no_inputs_or_outputs)

   if extract_sub == 1:
      if args.inputs == '' or args.outputs == '':
         logger.warning('WARNNIG: When extract sub graph, you must tell the inputs(ex: --inputs input0:0,input1:0) and outputs(ex: --outputs output0:0)')
         sys.exit(exit_code_extract_sub_no_inputs_or_outputs)

      if model_type != 'onnx':
         logger.warning('WARNNING: only onnx model supports extracting...')
         sys.exit(exit_code_model_type_not_onnx)

      r = extract_sub_graph(model_path, output, inputs, outputs)
      if r == True:
         logger.critical('Convert Success!')

      sys.exit(exit_code_normal)

   logger.info('begin convert..')

   begin_time = time.time()

   if model_type != 'onnx':
      convert(model_path,
               model_type,
               output,
               op_set_default,
               input_shape_list,
               inputs,
               outputs,
               model_def_file,
               model_class_name,
               model_input_type,
               model_weights_file,
               output_num,
               keep_batch,
               params_file,
               ceil_floor_reverse)

   end_time1 = time.time()

   logger.info('finish convert, it cost {} seconds'.format(end_time1 - begin_time))

   if model_type != 'onnx':
      model = onnx.load(output)
   else:
      model = onnx.load(model_path)

   # from onnx.external_data_helper import convert_model_from_external_data
   # convert_model_from_external_data(model)

   if op_set != None :
      if model_type == 'onnx':
         logger.info('ONNX, add_value_info_for_constants...')
         correct_batch_for_opset_convert(model)
         operation.add_value_info_for_constants(model)
         model = version_converter.convert_version(model, op_set)
      elif op_set != op_set_default:
         correct_batch_for_opset_convert(model)
         operation.add_value_info_for_constants(model)
         model = version_converter.convert_version(model, op_set)

      operation.eliminate_unused_input_initializer(model)

   inference_success = False
   new_model = model

   #new_model = onnx.shape_inference.infer_shapes(model)
   try:
      new_model = onnx.shape_inference.infer_shapes(model)
   except BaseException as e:
      print('The model cannot be inferenced for: %s' % e)
      new_model = model
   else:
      logger.info('Inference success---')
      inference_success = True

   #onnx.checker.check_model(new_model)
   tmp_model_file = None
   try:
      onnx.checker.check_model(new_model)
   except BaseException as e: #onnx.checker.ValidationError as e:
      if "maximum protobuf size of 2GB" in e.args[0]:
         _, tmp_model_file = tempfile.mkstemp(suffix='.onnx', prefix='tmp_', dir='./')
         tmp_model_file = os.path.basename(tmp_model_file)
         external_data = tmp_model_file + "_data"
         onnx.save(new_model, tmp_model_file, save_as_external_data=True, all_tensors_to_one_file=True, location=external_data, size_threshold=1024, convert_attribute=False)
      logger.warning('ignore warning(check_model), continue saving~')
   else:
      logger.info('### Begin saving model...')

   #if dynamic_batch == 1:
   #   logger.info('modify model to dynamic batch...')
   #   new_model = modify_onnx2dynamic(new_model)

   end_time2 = time.time()

   logger.info('generate inference shape model, it cost {} seconds'.format(end_time2 - end_time1))

   if tmp_model_file is not None:
      new_model = onnx.load(tmp_model_file)

   if simplify_model == 1 or simplify_model == 2:
      logger.info('begin doing simplify...')
      # new_model = new_model if tmp_model_file is None else tmp_model_file
      new_model = model_simplify(new_model, simplify_model, simplify_hw, simplify_skip_constant_folding, not simplify_skip_optimization)

   if tmp_model_file is not None:
      [os.remove(filename) for filename in [tmp_model_file, tmp_model_file+"_data"] if os.path.exists(filename)]

   post_process(new_model, inference_success)

   if add_node_name == 1:
      new_model = naming_onnx_node(new_model)

   if reset_batch != '':
      batchs = reset_batch #.split(' ')
      input_batch = int(batchs[0])
      output_batch = input_batch
      if len(batchs) >= 2:
         output_batch = int(batchs[1])

      if input_batch == 0:
         input_batch = -1

      if output_batch == 0:
         output_batch = -1

      logger.info('got batchs: {}'.format(batchs))

      new_model = reset_batch_size(new_model, input_batch, output_batch)

   if fuse_pad_pool == 1:
      logger.info('begin doing fuse_pad_to_pool...')
      new_model = fuse.fuse_pad_to_pool(new_model)

   if support_mish == 1:
      new_model = merge_mish(new_model)

   if support_swish == 1:
      new_model = merge_swish_and_hard_swish(new_model)

   if fuse_hard_sigmoid == 1:
      new_model = merge_hard_sigmiod(new_model)

   if expand_to_resize == 1:
      new_model = merge_resize(new_model)

   if model_type == 'onnx' and reset_value_info == 1:
      new_model = reset_model_value_info(new_model)

   if fuse_layernorm == 1:
      new_model = merge_layernorm(new_model)

   if fuse_gelu== 1:
      new_model = merge_gelu(new_model)

   if fuse_reducel2 == 1:
      new_model = merge_reducel2(new_model)

   if fp32_to_u8 == 1:
      new_model = fp32_to_uint8(new_model)

   if matmul_to_gemm == 1:
      new_model = matmul_add_to_gemm(new_model)

   #if fuse_mha == 1:
   #   new_model = mha_fuse(new_model)

   if fuse_ct_add_bn == 1:
      new_model = ct_add_bn_fuse(new_model)

   if fuse_matmul_bn == 1:
      new_model = matmul_bn_to_gemm(new_model)

   if fuse_conv_asmd == 1:
      new_model = conv_asmd_to_conv_v2(new_model)

   if fuse_mul_add_conv == 1:
      new_model = fuse_mac(new_model)

   if bn_to_conv == 1 and simplify_model != 0:
      new_model = bn2conv.bn2conv(new_model)

   if fp32_to_fp16 == 1:
      logger.info('begin doing fp32-->fp16...')
      new_model = convert_float_to_float16(new_model, keep_io_types=True)

   if fuse_mha == 1:
      new_model = mha_fuse(new_model)

   if fuse_gn == 1:
      new_model = gn_fuse(new_model)

   if fuse_sa == 1:
      new_model = sa_fuse(new_model)

   if fuse_gemm_bn == 1:
      new_model = gemm_bn_fuse(new_model)

   delete = operation.eliminate_redundant_reshape(new_model)
   while delete == True:
      delete = operation.eliminate_redundant_reshape(new_model)

   operation.eliminate_unused_input_initializer(new_model)
   operation.eliminate_unused_constant_node(new_model)
   operation.remove_unused_initializer(new_model)
   operation.remove_unused_castNode(new_model)
   new_model = fuse.fuse_transpose_relu_transpose(new_model)

   if dynamic_batch == 1:
      logger.info('---modify model to dynamic batch...')
      new_model = modify_onnx2dynamic(new_model)

   try:
      onnx.save(new_model, output)
   except:
      location_data = os.path.basename(output) + "_data"
      onnx.save(new_model, output, save_as_external_data=True, all_tensors_to_one_file=True, location=location_data, size_threshold=1024, convert_attribute=False)

   end_time3 = time.time()

   logger.info('The whole progress cost {} seconds'.format(end_time3 - begin_time))

   logger.critical('Convert Success!')

def main(args):
   #clear log file
   # with open("./convert.log", 'r+') as file:
   #    file.truncate(0)

   process(args)

if __name__ == "__main__":
   args = parse_args()
   main(args)
