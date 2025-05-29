# coding=utf-8
"""
-------------------------------------------------------------------
   Copyright (c) 2021-2023 Metax Inc. All rights reserved.

   Description :
   create date:  2023/4/14 15:54
-------------------------------------------------------------------
"""


import os
import yaml
import torch
import tqdm
import importlib
from addict import Addict
from torch.utils.data import DataLoader


from maca_quantizer.version import MXQ_CONFIG

from maca_quantizer.ppq_.ppq import BaseGraph, TorchExecutor, PPQ_CONFIG
from maca_quantizer.core.config import MXPPL_ALGORITHM_PLATFORM, EXPORTER_INSTANCE
from maca_quantizer.mxppl_quantize_tool import MACAPPLQuantize

from maca_quantizer.utils.utils import (register_module_from_file, maca_error, maca_warning, maca_info, 
                                        check_path, get_config, get_input_info)
from maca_quantizer.utils.io_utils import gen_random_dataloader, load_dataset




__all__ = ["MacaQuantizeRunner"]



class MacaQuantizeRunner():
    __EXPORT_TYPE = [e.name for e in EXPORTER_INSTANCE]
    __DEVICE_ENVIRONMENT_NAME = "MACA_QUANTIZER_USING_MXGPU"    # 0:Disable 1:Enable using cuda
    _PREPROC_MODULE_NAME = "MacaPreProc"
    _VERSION = MXQ_CONFIG.VERSION
    

    def __init__(self, config_file, ep='mxcuda',mode='auto', analyse=False, random=False):
        self.config_content = self.read_config(config_file)
        self._mode = mode
        self._analyse = analyse
        self._random = random
        self._inference_ep = ep

        self.check_device()
        self._init_config()
        self._check_config()

    @classmethod
    def version(cls):
        return cls._VERSION

    @classmethod
    def check_device(cls):
        enable_cuda = bool(int(os.environ.get(cls.__DEVICE_ENVIRONMENT_NAME,  1)))
        # PPQ_CONFIG.USING_CUDA_KERNEL = enable_cuda
        if enable_cuda and not torch.cuda.is_available():
            maca_error("Please set environment variables \"MACA_QUANTIZER_USING_MXGPU=0\" switch cpu version.")
        device =  'cuda' if enable_cuda  else  'cpu'
        return device


    @staticmethod
    def read_config(config_file):
        content = None
        if not os.path.exists(config_file):
            maca_error(f'{config_file} is not exist the file')
        try:
            with open(config_file, "r") as f:
                content = yaml.load(f, Loader=yaml.Loader)
        except Exception as e:
            maca_error(f'read yaml config error, error msg is: {e}')

        return Addict(content)

    @staticmethod
    def _soft_link(src, dst):
        os.symlink(os.path.abspath(src), os.path.abspath(dst))


    def _init_config(self) -> None:
        input_shape_list, input_type_list = get_input_info(self.config_content["import_model"])
        if 'device' not in self.config_content:
            self.config_content.device = self.check_device()
        if 'collecting_device' not in self.config_content:
            self.config_content.collecting_device = self.config_content.device

        self.config_content.without_bs   = get_config(self.config_content, 'without_bs',  default=False)
        self.config_content.input_type   = get_config(self.config_content, 'input_type', default=input_type_list)

        self.config_content.quant_dispatch  = get_config(self.config_content, 'dispatch', default='conservative')
        self.config_content.optimize_level  = get_config(self.config_content, 'optimize_level', default=2)
        self.config_content.export_type     = get_config(self.config_content, 'export_type', default='onnx')
        self.config_content.export_batch    = get_config(self.config_content, 'export_batch',  default=1)

        # Only support perchannel for weight 
        self.config_content.quant_type      = get_config(self.config_content, 'quant_type', default='perchannel')
        self.config_content.quant_algorithm = get_config(self.config_content, 'quant_algorithm', default=None)

        self.config_content.output_threshold      = get_config(self.config_content, 'output_threshold', default=0.1)
        self.config_content.force_advance_quant   = get_config(self.config_content, 'force_advance_quant', default=False)
        

        if 'input_shape' not in self.config_content:
            self.config_content.input_shape = input_shape_list
        else:
            self.set_input_shape(input_shape_list) 


    def set_input_shape(self, shape_list):
        # Todo：dynamic height and width must specify input_shape
        checked_shapes = []
        if len(shape_list)==1:
            assert -1 not in self.config_content.input_shape, "Please specify input_shape"
            if not self.config_content.without_bs:
                checked_shapes.append([1] + [i for i in self.config_content.input_shape])
            else:
                checked_shapes.append([i for i in self.config_content.input_shape])
        else:
            assert all(-1 not in shape for shape in self.config_content.input_shape), \
                 "Please specify input_shape"
            if not self.config_content.without_bs:
                checked_shapes = [[1] + shape for shape in self.config_content.input_shape]
            else:
                checked_shapes = [shape for shape in self.config_content.input_shape]
        self.config_content.input_shape = checked_shapes


    # TODO :register preprocess module
    def _register_preproc(self):
        default_preproc_file = os.path.join(os.path.dirname(__file__), 'preprocess/preprocess_v1.py')
        preproc_type = get_config(self.config_content.dataset.preprocessing, key='type', default='PreprocessV1')
        file_path = get_config(self.config_content.dataset.preprocessing, key='path', default=default_preproc_file)
        register_module_from_file(module_name=self._PREPROC_MODULE_NAME, file_path=file_path)
        self.config_content.dataset.preprocessing.type = preproc_type


    def _check_config(self):
        check_path(self.config_content.import_model)
        check_path(os.path.abspath(self.config_content.export_model), check_dir=True)
        if not self._random:
            check_path(self.config_content.dataset.calib_dir)
        # if not os.path.exists(os.path.dirname(os.path.abspath(self.config_content.export_model))):
        #     os.makedirs(os.path.dirname(os.path.abspath(self.config_content.export_model)))
        
        if self.config_content.dataset.preprocessing.enable:
            self._register_preproc()

        if self.config_content.export_type not in self.__EXPORT_TYPE:
            maca_error(f"Export type \"{self.config_content.export_type}\" is not supported, Please choice in {self.__EXPORT_TYPE}")

        if len(self.config_content.input_type) != len(self.config_content.input_shape):
            maca_error(f"input_shape number {len(self.config_content.input_shape)}, input_type number {len(self.config_content.input_type)}")
        if self.config_content.device=='cuda':
            maca_info(f"{MXQ_CONFIG.NAME} will using GPU to accelerate.")
            assert torch.cuda.is_available(), "Pytorch cannot use cuda, Please switch cpu version."
        return


    def preprocess_data(self, calib_num=None, device='cpu'):
        preprocess_cfg = self.config_content.dataset.preprocessing
        input_shape = self.config_content.input_shape
        calib_dir = self.config_content.dataset.calib_dir
        without_bs = self.config_content.without_bs
        batch_size = self.config_content.dataset.batch_size
        if not preprocess_cfg.enable:
            dataloader =  load_dataset(calib_dir, input_shape, batch_size, count=calib_num, device=device, without_bs=without_bs)
            return dataloader

        preprocess_type = preprocess_cfg.type
        preprocess_attr = preprocess_cfg.attributes

        # TODO multiply and dynamic
        input_shape_ = input_shape[0] if self.config_content.without_bs else input_shape[0][1:]  # 预处理不需要bs

        module = importlib.import_module(self._PREPROC_MODULE_NAME)
        # TODO: init parameter not common
        proprecess_obj = module.__dict__[preprocess_type](calib_dir, input_shape_, batch_size, calib_num)
        data_tensor = proprecess_obj.preprocess(preprocess_attr, device=device)

        # file_name = os.path.join("./data/input_data/", os.path.basename(self.config_content.import_model).replace('onnx','npy'))
        # np.save(file_name, data_tensor[0].cpu().numpy())
        calibration_dataloader = DataLoader(dataset=data_tensor, batch_size=batch_size, shuffle=False)

        return calibration_dataloader

    def run(self):
        dynamic_flag = self.config_content.export_batch==-1 and not self.config_content.without_bs
        type_platform = MXPPL_ALGORITHM_PLATFORM[self.config_content.quant_type]

        if self.config_content.without_bs:
            input_shape_list = self.config_content.input_shape
        else:
            bs= self.config_content.export_batch if not dynamic_flag else 1
            input_shape_list = [[bs]+ shape[1:] for shape in self.config_content.input_shape]

        if self._random:
            calibrate_dataloader = gen_random_dataloader(input_shape_list, self.config_content.without_bs, device=self.config_content.device)
        else:
            calibrate_dataloader = self.preprocess_data(self.config_content.dataset.calib_num, device=self.config_content.device)

        opt_config = get_config(self.config_content, "optimize", default={})
        # quantize_tool = MACAQUANTIZATIONTOOL[self._inference_ep.upper()].value
        quantization_obj = MACAPPLQuantize(
                                    import_model_file      = self.config_content.import_model, 
                                    export_model_file      = self.config_content.export_model,
                                    calib_dataloader       = calibrate_dataloader, 
                                    input_shape            = input_shape_list,
                                    input_type             = self.config_content.input_type, 
                                    batch_size             = self.config_content.dataset.batch_size,
                                    device                 = self.config_content.device,
                                    platform               = type_platform,
                                    dispatch_type          = self.config_content.quant_dispatch,
                                    quant_algorithm        = self.config_content.quant_algorithm,
                                    optimize_output_level  = self.config_content.optimize_level,
                                    force_advance_quant    = self.config_content.force_advance_quant,
                                    collecting_device      = self.config_content.collecting_device,
                                    output_threshold       = self.config_content.output_threshold,
                                    optimize_config        = opt_config
                                    )
        quantized_graph = quantization_obj.autoQuantization(self._mode)
        if self._analyse:
            quantization_obj.statistical_analyse_graph(quantized_graph)

        dynamic_shape = get_config(self.config_content, "dynamic_shape", default=False)
        quantization_obj.post_quant_refine(graph=quantized_graph, dynamic_shape=dynamic_shape)

        if not dynamic_flag and not self.config_content.without_bs:
            quantization_obj.covert_specify_batchsize(graph=quantized_graph, batch_size=self.config_content.export_batch)

        quantization_obj.export_quantization(graph=quantized_graph, export_type=self.config_content.export_type, dynamic_batch=dynamic_flag, dynamic_shape=dynamic_shape)
        # quantization_obj.convert_model_form_native(model=quantized_graph, dynamic_batch=dynamic_flag)

        return quantized_graph

    def run_naive(self, dataloader):
        dynamic_flag = self.config_content.export_batch==-1 and not self.config_content.without_bs
        type_platform = MXPPL_ALGORITHM_PLATFORM[self.config_content.quant_type]
        if not isinstance(dataloader, DataLoader):
            dataloader = DataLoader(dataloader, batch_size=self.config_content.dataset.batch_size)

        data_sample = next(iter(dataloader))

        input_shape_list = []
        for key, var in data_sample.items():
            input_shape_list.append(var.shape)

        opt_config = get_config(self.config_content, "optimize", default={})
        # quantize_tool = MACAQUANTIZATIONTOOL[self._inference_ep.upper()].value
        quantization_obj = MACAPPLQuantize(
                                    import_model_file      = self.config_content.import_model, 
                                    export_model_file      = self.config_content.export_model,
                                    calib_dataloader       = dataloader, 
                                    input_shape            = input_shape_list,
                                    input_type             = self.config_content.input_type, 
                                    batch_size             = self.config_content.dataset.batch_size,
                                    device                 = self.config_content.device,
                                    platform               = type_platform,
                                    dispatch_type          = self.config_content.quant_dispatch,
                                    quant_algorithm        = self.config_content.quant_algorithm,
                                    optimize_output_level  = self.config_content.optimize_level,
                                    force_advance_quant    = self.config_content.force_advance_quant,
                                    collecting_device      = self.config_content.collecting_device,
                                    output_threshold       = self.config_content.output_threshold,
                                    optimize_config        = opt_config
                                    )
        quantized_graph = quantization_obj.autoQuantization(self._mode)
        if self._analyse:
            quantization_obj.statistical_analyse_graph(quantized_graph)
        quantization_obj.post_quant_refine(graph=quantized_graph)

        if not dynamic_flag and not self.config_content.without_bs:
            quantization_obj.covert_specify_batchsize(graph=quantized_graph, batch_size=self.config_content.export_batch)

        quantization_obj.export_quantization(graph=quantized_graph, export_type=self.config_content.export_type, dynamic_batch=dynamic_flag)

        return quantized_graph


    def execute(self, graph):
        # only supported export graph TargetPlatform.NATIVE
        execution_dataloader = self.preprocess_data(calib_num=50)
        assert isinstance(graph, BaseGraph)
        # build an executor:
        results = []
        executor = TorchExecutor(graph=graph, device='cuda')
        for data in tqdm(execution_dataloader, desc='Running with executor.'):
            tensor = executor.forward(inputs=data.to('cuda'))
            results.append(tensor)
        return results 

