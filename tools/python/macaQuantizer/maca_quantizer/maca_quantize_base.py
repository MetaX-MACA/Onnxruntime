# coding=utf-8
"""
-------------------------------------------------------------------
   Copyright (c) 2021-2023 Metax Inc. All rights reserved.

   Description :
   create date:  2023/4/13 15:54
-------------------------------------------------------------------
"""


import os
import sys
import torch
import json

from ppq import *
from ppq.api import *

import ppq.lib as PFL
from ppq.lib.common import *
from ppq.executor.torch import OPERATION_FORWARD_TABLE
from ppq.IR.morph import GraphDeviceSwitcher, GraphFormatter

from maca_quantizer.core.config import REGISTER_TABLE                                   

from maca_quantizer.utils.utils import maca_error,maca_warning, maca_info
from maca_quantizer.utils.io_utils import empty_maca_cache
from maca_quantizer.version import MXQ_CONFIG

sys.setrecursionlimit(100000)


class MacaQuantizeBase(object):
    """
        Maca automtic quantization tool
        Args:
            import_model_file  (str)         : 被量化的 onnx 模型文件路径 onnx model location (目前只支持 onnx 模型)
            export_model_file  (str)         : 量化后导出的 onnx 模型文件路径 
            calibrate_dir  (str)             ：校准数据集 data 目录 (支持.npy 格式的数据，以及 .bin 或 .raw 的二进制数据, 数据需在目录下的data文件夹下) 
            input_shape  (List[int])         ：被量化模型的输入shape
            batchsize  (int)                 : batchsize 大小

            optimize_output_level            : 

    """
    def __init__(self, 
                import_model_file, 
                export_model_file, 
                calib_dataloader, 
                input_shape, 
                input_type,
                batch_size = 16, 
                device = 'cuda', 
                platform = TargetPlatform.PPL_CUDA_INT8, 
                input_layout = 'chw', 
                optimize_output_level = 1, 
                force_advance_quant = False,
                collecting_device = 'cuda',
                output_threshold = 0.1,
                optimize_config={}
                ):
        self.import_model_file = import_model_file
        self.calib_dataloader = calib_dataloader
        self.input_shape = input_shape
        self.input_type = input_type
        self.batch_size = batch_size
        self._executing_device = device
        self._target_platform = platform
        self._optimize_output_level = optimize_output_level
        self._force_advance_quant = force_advance_quant
        self._optimize_config = optimize_config

        self._collecting_device = collecting_device
        self._output_threshold = output_threshold
        self.export_model_file = os.path.abspath(export_model_file)
        self.export_cfg_file   = os.path.splitext(self.export_model_file)[0] + "_cfg.json"
        self._hybrid_op_list = []

        self._calib_steps= min(len(self.calib_dataloader), 512)
        self.__version = MXQ_CONFIG.VERSION


    @property
    def version(self):
        return self.__version

    def collate_fn(self, batch):
        if isinstance(batch, list):
            return [x.to(self._executing_device) for x in batch]
        else:
            return batch.to(self._executing_device)


    def _register_handler(self, register_type):
        self.print_register(register_type)
        for key, value in REGISTER_TABLE[register_type].items():
            if value is None: continue        
            # register quantizer
            if key=="parser":
                register_network_parser(parser=value, framework=NetworkFramework.ONNX)
            if key=="operation":  
                for op_type, op_forwad in value.items():
                    register_operation_handler(handler=op_forwad, operation_type=op_type, platform=self._target_platform)
                    register_operation_handler(handler=op_forwad, operation_type=op_type, platform=TargetPlatform.UNSPECIFIED)
                    register_operation_handler(handler=op_forwad, operation_type=op_type, platform=TargetPlatform.FP32)
                    maca_info(f"Register operstion: {op_type:10} --> {op_forwad}")
            # register quantizer
            elif key=="quantizer":
                register_network_quantizer(value, platform=self._target_platform)
            # register exporter
            elif key=="exporter":
                register_network_exporter(exporter=value, platform=self._target_platform)

    def print_register(self, register_type):
        assert register_type in REGISTER_TABLE, f"{register_type} not in REGISTER_TABLE"
        for key, value in REGISTER_TABLE[register_type].items():
            if key=="operation":  continue
            if value is None:
                if key=="parser":
                    value = __PARSERS__[self._target_platform]
                elif key=="quantizer":
                    value = __QUANTIZER_COLLECTION__[self._target_platform]
                elif key=="exporter":
                    value = __EXPORTERS__[self._target_platform]
            maca_info(f"{key:10}: {value}")


    def _load_maca_graph(self, model_path, **kwargs):
        graph = load_onnx_graph(onnx_import_file=model_path)
        return graph


    @empty_maca_cache
    def manually_optimize(self, graph, passes, trace_meta=False, **kwargs):
        torch_executor = TorchExecutor(graph, device=self._executing_device)

        if trace_meta:    
            try:
                dummy_input = next(iter(self.calib_dataloader))
            except:
                shape = list(self.calib_dataloader.dataset[0].shape)
                input_shape = self.calib_dataloader.batch_size + shape
                input_device = self.calib_dataloader.dataset[0].device
                input_dtype = self.calib_dataloader.dataset[0].dtype
                dummy_input = torch.ones(size=input_shape, device=input_device, dtype=input_dtype)
            torch_executor.tracing_operation_meta(inputs=dummy_input)
    
        opt_pass =  PFL.Pipeline(passes)
        opt_pass.optimize(
                graph=graph, dataloader=self.calib_dataloader, verbose=True, 
                calib_steps=32, collate_fn=self.collate_fn, 
                executor=torch_executor)
        return graph

    @empty_maca_cache
    def analyse_error(self, graph, method='snr', layerwise=False, graphwise=True):
        # TODO: specify threshold for different method
        error_threshold = self._output_threshold
        graph_reports, layer_reports = {}, {}
        output_error = {}
        layer_flag, graph_flag, output_flag = True, True, True
        error_op = set()

        if graphwise:
            maca_info('Computing Graphwise Quantization SNR')
            graph_reports = graphwise_error_analyse(
                graph=graph, running_device=self._executing_device, steps=4,
                dataloader=self.calib_dataloader, collate_fn=self.collate_fn,
                method = method)
            if graph_reports is not None: 
                for op, value in graph_reports.items():
                    if method.lower()=='snr':
                        if value > error_threshold: 
                            # maca_warning(f'{op} graphwise error significatly')
                            error_op.add(op)
                            graph_flag=False
                            # break
                    else:
                        if value < error_threshold: 
                            # maca_warning(f'{op} graphwise error significatly')
                            error_op.add(op)
                            graph_flag=False
                            # break


        if layerwise:
            maca_info('Computing Layerwise Quantization SNR')
            layer_reports = layerwise_error_analyse(
                graph=graph, running_device=self._executing_device, steps=4,
                dataloader=self.calib_dataloader, collate_fn=self.collate_fn,
                method = method)
            for op, value in layer_reports.items():
                if method.lower()=='snr':
                    if value > error_threshold: 
                        # maca_warning(f'{op} layerwise error significatly')
                        error_op.add(op)
                        layer_flag=False
                        # break
                else:
                    if value < error_threshold: 
                        # maca_warning(f'{op} layerwise error significatly')
                        error_op.add(op)
                        layer_flag=False
                        # break

        output_report_name = self.get_output_compute_op(graph, graph_reports)
        for name in set(output_report_name):
            if graph_reports[name] > error_threshold:
                output_flag = False
                maca_warning(f'{name} output error significatly, Please optimize')
            output_error[name] = graph_reports[name]

        return output_flag, output_error, error_op


    @empty_maca_cache
    def statistical_analyse_graph(self, graph):
        records = statistical_analyse(graph, 
                    running_device=self._executing_device,
                    dataloader=self.calib_dataloader, 
                    collate_fn=self.collate_fn,
                    steps=4)
        # ready to records config to json.
        quant_info_recorder = {}
        for info in records:
            op_name = info['Op name']
            quant_info_recorder[op_name] = info

        exports = {'quant_info': quant_info_recorder}
        file_path = os.path.splitext(self.export_model_file)[0] + '_static_analyze.cfg'
        with open(file=file_path, mode='w') as file:
            json.dump(exports, file, indent=4)
        return records


    @empty_maca_cache
    def _quantize_model(self, quant_setting):
        maca_info('Network Start Quantize')
        model_graph = self._load_maca_graph(model_path=self.import_model_file)
        if len(self.input_shape) > 1: # multiply input
            dummy_input = [torch.zeros(size=shape_, device=self._executing_device, \
                dtype=type_) for shape_, type_ in zip(self.input_shape, self.input_type)]
            quantized = quantize_native_model(
                setting=quant_setting,
                model=model_graph,
                calib_dataloader=self.calib_dataloader,
                calib_steps=self._calib_steps, 
                input_shape=None,
                input_dtype=None,
                # inputs=self.calib_dataloader[0],
                inputs=dummy_input,
                collate_fn=self.collate_fn, 
                platform=self._target_platform,
                device=self._executing_device
                )
        else:
            quantized = quantize_native_model(
                setting=quant_setting,
                model=model_graph,
                calib_dataloader=self.calib_dataloader,
                calib_steps=self._calib_steps, 
                input_shape=self.input_shape[0],
                input_dtype = self.input_type[0],
                inputs=None,
                collate_fn=lambda x: x.to(self._executing_device), 
                platform=self._target_platform,
                device=self._executing_device
                )

        return quantized


    def export_quantization(self, graph, export_type, copy_graph=True):
        maca_info(f'Generating Quantization target file:{self.export_model_file}')
        assert isinstance(graph, BaseGraph)
        external_data = True if export_type in {"external"} else False
        extra_param = {
            "save_as_external_data": external_data
        }
        export_ppq_graph(
                graph=graph, platform=self._target_platform,
                graph_save_to=self.export_model_file, 
                config_save_to=self.export_cfg_file,
                copy_graph=copy_graph, **extra_param)




        
