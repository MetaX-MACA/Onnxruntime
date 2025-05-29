# coding=utf-8
"""
-------------------------------------------------------------------
   Copyright (c) 2021-2023 Metax Inc. All rights reserved.

   Description :
   create date:  2023/4/13 15:54
-------------------------------------------------------------------
"""


import sys
import copy
import onnx
import torch
import numpy as np

from ppq import *
from ppq.api import *
from ppq.quantization.optim import (LearnedStepSizePass, AdaroundPass, PassiveParameterQuantizePass, 
                                    ParameterBakingPass)
from maca_quantizer.utils import (maca_error, maca_warning, maca_info, get_config, empty_maca_cache, 
                                  modify_onnx2dynamic, get_hybrid_path_op)

from .maca_quantize_base import MacaQuantizeBase
from .core.optim import MetaxDispatchPass, MetaxMixturePass, MetaxFP16Pass, MetaxFormatGemmPass, MetaxRemoveUselessPass, MetaxTransposeBetweenPass
from .core.config import  EXPORTER_INSTANCE, FUSE_OPERATION_TABLE, FUSE_OPERATION_TYPE, ACTIVATIONS_TYPE, DISPATCH_FP32_ATTRIB_KEY
from .version import MXQ_CONFIG  

sys.setrecursionlimit(100000)



class MACAPPLQuantize(MacaQuantizeBase):
    """
        Maca automtic quantization tool
        Args:
            import_model_file  (str)         : 被量化的 onnx 模型文件路径 onnx model location (目前只支持 onnx 模型)
            export_model_file  (str)         : 量化后导出的 onnx 模型文件路径 
            calibrate_dir  (str)             ：校准数据集 data 目录 (支持.npy 格式的数据，以及 .bin 或 .raw 的二进制数据, 数据需在目录下的data文件夹下) 
            input_shape  (List[int])         ：被量化模型的输入shape
            batchsize  (int)                 : batchsize 大小

            optimize_output_level            : 0:disable     1: int8 + fp32     2: int8 + fp16

    """
    __OP_FUSION_ENVIRONMENT_NAME = 'MX_ENABLE_OPERATOR_FUSION'
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
                dispatch_type='conservative',
                # quant_algorithm='percentile', 
                quant_algorithm=None, 
                optimize_config={}
                ):
        super().__init__(
                import_model_file, 
                export_model_file, 
                calib_dataloader, 
                input_shape, 
                input_type,
                batch_size, 
                device, 
                platform, 
                input_layout, 
                optimize_output_level, 
                force_advance_quant,
                collecting_device,
                output_threshold,
                optimize_config
                )
        self._activation_algo = quant_algorithm
        self._dispatch_type = dispatch_type
        self._register_handler('mxc_pplcuda')

    def _load_maca_graph(self, model_path):
        graph = super()._load_maca_graph(model_path)
        self.pre_quant_refine(graph)
        return graph

    def default_config(self) -> QuantizationSetting:
        quant_setting = QuantizationSettingFactory.pplcuda_setting()
        # quant_setting.fusion_setting.fuse_activation = False
        # choice in {'pointwise', 'conservative', 'pursus', 'allin', 'pplnn'}
        align_element_config = get_config(self._optimize_config,'align_elementwise', default=True)
        align_concat_config  = get_config(self._optimize_config,'align_concat', default=True)
        quant_setting.dispatcher = self._dispatch_type
        quant_setting.quantize_parameter_setting.baking_parameter = False

        if not align_element_config:
            quant_setting.fusion_setting.align_elementwise_to = 'None'
        if not align_concat_config:
            quant_setting.fusion_setting.align_concat_to = 'None'
        # quant_setting.fusion_setting.align_avgpooling_to = "Align to Output"

        equalization_config = get_config(self._optimize_config,'equalization', default=None)
        equalization_enable = get_config(equalization_config,'enable', False)
        if equalization_enable:
            equalization_type = get_config(equalization_config,'type', compulsive=True)
            if equalization_type=='layerwise':
                quant_setting.equalization = True
            elif equalization_type=='ssd':
                quant_setting.ssd_equalization = True
            else:
                raise ValueError(f"Unknown equalization type \'{equalization_type}\'")

        return quant_setting

    def init_basic_configs(self, config: QuantizationSetting) -> List[QuantizationSetting]:
        basic_config = config
        calib_algo_list = [None, 'percentile','minmax', 'kl']
        config_list = [copy.deepcopy(basic_config) for i in range(len(calib_algo_list))]
        # config-0
        config_list[0].quantize_activation_setting.calib_algorithm = self._activation_algo

        if self._activation_algo in calib_algo_list:
            calib_algo_list.pop(calib_algo_list.index(self._activation_algo))
        else:
            raise ValueError(f"Not Implement activation algorithm \'{self._activation_algo}\'")

        for i in range(len(config_list[1:])):
            config_list[i+1].quantize_activation_setting.calib_algorithm = calib_algo_list[i]

        if self._activation_algo is not None:
            config_list = config_list[:1]

        return config_list


    def get_output_compute_op(self, graph, reports):
        output_computer_name = []
        if reports is None or len(reports)==0: 
            return output_computer_name
        _, ouput_compute_op = self.optimize_output_op(graph)
        for name in ouput_compute_op:
            if name in reports:
                output_computer_name.append(name)
            else:
                maca_warning(f'{name} not in error analyse report')
        return output_computer_name


    def optimize_output_op(self, graph):
        optimize_opset = set()
        optimize_computer_opset = set()

        search_engine = SearchableGraph(graph)

        for name, var in graph.outputs.items():
            if var.source_op is None:
                continue
            if var.source_op.is_computing_op:
                optimize_computer_opset.add(var.source_op.name)
                continue
            paths = search_engine.path_matching(
                                sp_expr=lambda x: x.is_computing_op, 
                                rp_expr=lambda x, y: (not y.is_computing_op), 
                                ep_expr=lambda x: x.name==var.source_op.name, 
                                direction='down')
            for path in paths:
                path = path.tolist()
                # op_name_list = [x.name for x in path]
                if len(path)<=1:
                    continue
                optimize_computer_opset.add(path[0].name)
                for op in path[1:]:
                    optimize_opset.add(op.name)

        return optimize_opset, optimize_computer_opset


    def get_hybrid_op(self, graph, op_names):
        """   
            get hybrid operator by operator name
        """
        op_list = []
        for name in op_names:
            if name not in graph.operations: continue
            op = graph.operations[name]
            if op.is_computing_op:
                op_list.append(name)
                down_ops = graph.get_downstream_operations(op)
                if len(down_ops)==1 and down_ops[0].type in ACTIVATIONS_TYPE:
                    op_list.append(down_ops[0].name)
            else:
                op_list.append(name)

        # perfine_optimize and op_morph dispath 
        for op in graph.operations.values():
            if op.extension_attrib.get(DISPATCH_FP32_ATTRIB_KEY, 0):
                op_list.append(op.name)

        return op_list


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
                inputs=next(iter(self.calib_dataloader)),
                # inputs=None,
                collate_fn=lambda x: x.to(self._executing_device), 
                platform=self._target_platform,
                device=self._executing_device
                )

        return quantized


    def check_quantized_graph(self, graph):
        interested_op = [operation for operation in graph.operations.values() 
                         if (isinstance(operation, QuantableOperation) and operation.is_computing_op)]
        if len(interested_op) == 0:
            maca_warning('Nothing ops quantized, Please check model or code')
            return False
        return True
    
    def post_op_dispatch(self, graph, config: QuantizationSetting):
        for operation in graph.operations.values():
            if operation.extension_attrib.get("PostPro", False):
                config.dispatching_table.append(operation=operation.name, platform=TargetPlatform.FP32)
        return config


    def get_operation_fusion(self) -> List[QuantizationOptimizationPass]:
        base_passes=[]
        # fuse_op_type = FUSE_OPERATION_TYPE.copy()
        fuse_env = os.environ.get(self.__OP_FUSION_ENVIRONMENT_NAME, '-openAll:0')
        fuse_env_list = fuse_env.split('-')[1:]
        if 'closeAll' in fuse_env_list:
            return base_passes

        fusion_enable = dict(map(lambda x: x.split(":"), fuse_env_list))

        for key, value in FUSE_OPERATION_TABLE.items():
            if key not in FUSE_OPERATION_TYPE or value is None: continue
            if key.lower() in fusion_enable.keys() and fusion_enable[key.lower()] == '0': continue
            base_passes.append(value())

        return base_passes

    def pre_quant_refine(self, graph: BaseGraph, passes: List[QuantizationOptimizationPass]=[]):
        # # TODO: add per_refine pass 
        # base_passes=[]
        # for key, value in FUSE_OPERATION_TABLE.items():
        #     if key not in FUSE_OPERATION_TYPE or value is None: continue
        #     base_passes.append(value())

        base_passes=self.get_operation_fusion()
        if self._optimize_output_level > 0:
            base_passes.append(MetaxDispatchPass())
        base_passes = base_passes + passes + [MetaxFormatGemmPass()]
        self.manually_optimize(graph=graph, passes=base_passes)

        mate_passes = [MetaxRemoveUselessPass(), MetaxTransposeBetweenPass()]
        self.manually_optimize(graph=graph, passes=mate_passes, trace_meta=True)

        return graph


    def post_quant_refine(self, graph: BaseGraph, passes: List[QuantizationOptimizationPass] = [] , dynamic_batch=False, dynamic_shape=False):
        if not self.check_quantized_graph(graph):
            return graph
        from .core.optim.post_refine import MetaxAutoCastPass
        passes.append(MetaxAutoCastPass())
        if self._optimize_output_level > 1:
            passes.extend([MetaxFP16Pass(self._hybrid_op_list)])
        self.manually_optimize(graph=graph, passes=passes)

        # Add for dynamic batch size
        if dynamic_batch:
            for _, in_var in graph.inputs.items():
                in_var.shape[0] = -1
            for _, out_var in graph.outputs.items():
                out_var.shape[0] = -1
        if dynamic_shape:
            for _, var in graph.inputs.items():
                for i in range(len(var.shape)):
                    if i==0: continue
                    var.shape[i] = f"indim_{i}"

            for _, var in graph.outputs.items():
                for i in range(len(var.shape)):
                    if i==0: continue
                    var.shape[i] = f"outdim_{i}"

        return graph


    def covert_specify_batchsize(self, graph: BaseGraph, batch_size: int):
        if batch_size==1: return graph
        for _, in_var in graph.inputs.items():
            in_var.shape[0] = batch_size
        for _, out_var in graph.outputs.items():
            out_var.shape[0] = batch_size
        # TODO
        dummy_input=[]
        for _, var in graph.inputs.items():
            dummy_input.append(torch.ones(size=var.shape))
        executor = TorchExecutor(graph=graph, device=self._executing_device)
        executor.tracing_operation_meta(inputs=dummy_input)
        return graph


    def baseQuantization(self, settings, mode):
        quantized_graph = []
        output_sum_error = []
        output_flag = False

        basic_settings = settings[:1] if mode in {'debug','fast','naive'} else settings
        enable = True if mode == 'debug' else False
        # enable = False

        for idx, quant_setting in enumerate(basic_settings):
            quantized = self._quantize_model(quant_setting)
            output_flag, output_error, _ = self.analyse_error(quantized, layerwise=enable)
            output_sum_error.append(np.sum(list(output_error.values())))
            maca_info("Basic config index: {} , output_error: {}".format(idx, output_error))
            if output_flag:
                maca_info('Basic Quantization Completed:')
                return quantized, idx, output_flag
            quantized_graph.append(quantized)

        best_index = np.argmin(output_sum_error)
        best_graph = quantized_graph[best_index]#.copy(copy_value=True)
        maca_info("Basic config best index {}".format(best_index))
        
        return best_graph, best_index, output_flag
    
    def advanceQuantization(self, graph: BaseGraph, qconfig: QuantizationSetting):
        training_config = get_config(self._optimize_config,'training')
        learning_rate = float(get_config(training_config, 'lr', default=1e-05))
        training_steps = int(get_config(training_config, 'steps', default=500))
        loss_function = torch_mean_square_error

        optimize_passes = {
            "lsq": [
                LearnedStepSizePass(lr=learning_rate, 
                                    steps=training_steps, 
                                    collecting_device=self._collecting_device, 
                                    loss_fn=loss_function),
                PassiveParameterQuantizePass(), ParameterBakingPass()
                ],
            "adround": [
                AdaroundPass(lr=learning_rate, 
                             steps=training_steps, 
                             collecting_device=self._collecting_device), 
                PassiveParameterQuantizePass(), ParameterBakingPass()
                ]
        }

        quant_graph = graph

        # TODO: config specilfied
        opt_passes = optimize_passes['lsq']

        # # Add BiasCorrectionPass if used equalization
        # if qconfig.equalization or qconfig.ssd_equalization:
        #     opt_passes = [BiasCorrectionPass()] + opt_passes

        self.manually_optimize(graph=quant_graph, passes=opt_passes)
  
        maca_info('Advanced Quantization Completed')

        return quant_graph


    def autoQuantization(self, mode='auto'):
        global_setting = self.default_config()
        graph_ir = self._load_maca_graph(model_path=self.import_model_file)

        # before quantization to dispatch post_op
        if self._optimize_output_level>0:
            global_setting = self.post_op_dispatch(graph_ir, global_setting)

        hybrid_config = get_config(self._optimize_config,'hybrid', None)
        if hybrid_config is not None:
            # mixture operation dispatch
            hybrid_enable = get_config(hybrid_config,'enable', default=False)
            hybrid_ops = set(get_config(hybrid_config,'ops', default=[]))
            # self._hybrid_op_list = self.get_hybrid_op(graph_ir, hybrid_ops)
            self._hybrid_op_list = self.get_hybrid_op(graph_ir, hybrid_ops)

            hybrid_path_ops = get_config(hybrid_config,'path_ops', default=[])
            hybrid_path_ops_list = get_hybrid_path_op(graph_ir, hybrid_path_ops)
            self._hybrid_op_list.extend(hybrid_path_ops_list)

            if hybrid_enable and len(self._hybrid_op_list)!=0:
                for op_name in set(self._hybrid_op_list):
                    maca_info(f'{op_name} will hybrid quantize use FP32')
                    global_setting.dispatching_table.append(operation=op_name, platform=TargetPlatform.FP32)
        del graph_ir

        basic_settings = self.init_basic_configs(global_setting)
        best_basic_graph, index, flag = self.baseQuantization(settings=basic_settings, mode=mode)

        if (flag and not self._force_advance_quant) or mode in {'debug','fast'}:
            maca_info('MacaQuantization Completed')
            return best_basic_graph

        maca_info("Start Advanced Quantization")
        quantized_graph = self.advanceQuantization(graph=best_basic_graph, qconfig=basic_settings[index])
        self.analyse_error(quantized_graph)
        maca_info('MacaQuantization Completed')
        return quantized_graph


    def hybridQuantization(self, graph, setting):
        """
        Hybrid quantization fuction 
        Args:
            graph:   BaseGrph (quantized graph)
            setting: QauntizationSetting
        """
        maca_info("Start Hybrid Quantization")
        # 算子调度(computer_op + activation)
        customer_mix_op = set(get_config(self._optimize_config,'hybrid', []))
        fp_ops_list = self.get_hybrid_op(graph, customer_mix_op)
        if len(fp_ops_list)==0:
            maca_warning('Nothing ops hybrid quantization.')

        for op_name in fp_ops_list:
            maca_info(f'{op_name} will hybrid quantize use FP32')
            setting.dispatching_table.append(operation=op_name, platform=TargetPlatform.FP32)
        quantized_graph = self._quantize_model(setting)
        #  MixPercision adjust
        for op_name in fp_ops_list:
            op = quantized_graph.operations[op_name]
            assert op.platform == TargetPlatform.FP32
            input_var = op.inputs[0] 
            flag = all([x.platform==TargetPlatform.FP32 for x in input_var.dest_ops])
            if input_var.source_op is not None and flag and \
                        isinstance(input_var.source_op, QuantableOperation):
                input_var.source_op.config.output_quantization_config[0].state = QuantizationStates.FP32

        maca_info('Computing MixPercision Graphwise Quantization SNR')
        self.analyse_error(quantized_graph)

        # maca_info('Computing MixPercision Layerwise Quantization SNR')

        maca_info('Hybrid Quantization Completed')
        return quantized_graph


    def export_quantization(self, graph, export_type, dynamic_batch=False, dynamic_shape=False, copy_graph=True):
        assert isinstance(graph, BaseGraph)
        if copy_graph: graph = graph.copy()

        exporter = EXPORTER_INSTANCE[export_type].value()
        export_cfg_file = self.export_cfg_file if export_type in {'ppl'} else None
        extra_param = {} 

        if export_type in {'onnx'}:
            extra_param['model_file'] = self.import_model_file
        if dynamic_shape:
            extra_param['dynamic_shape'] = True

        exporter.export(graph=graph, 
                        file_path=self.export_model_file, 
                        config_path=export_cfg_file,
                        **extra_param)
        if dynamic_batch:
            onnx_model = onnx.load(self.export_model_file)
            dynamic_model = modify_onnx2dynamic(onnx_model)
            maca_info("Convert dynamic batch size to \'-1\'")
            onnx.save_model(dynamic_model, self.export_model_file)

        maca_info(f'Generating Quantization Model file:{self.export_model_file}')
        if export_cfg_file is not None:
            maca_info(f'Generating Quantization Config file:{export_cfg_file}')
    

    def convert_model_form_native(self, model: Union[str, BaseGraph], dynamic_batch: bool = False, copy_graph: bool = True):
        if isinstance(model, BaseGraph):
            graph = model
        else:
            graph = load_native_graph(model)

        if copy_graph: graph_ = graph.copy()

        exporter_onnx = EXPORTER_INSTANCE['onnx'].value()
        exporter_onnx.export(graph=graph_, 
                        file_path=self.export_model_file, 
                        config_path=None,
                        model_file = self.import_model_file)
        if dynamic_batch:
            onnx_model = onnx.load(self.export_model_file)
            dynamic_model = modify_onnx2dynamic(onnx_model)
            maca_info("Convert dynamic batch size to \'-1\'")
            onnx.save_model(dynamic_model, self.export_model_file)
        maca_info(f'Generating Quantization Model file:{self.export_model_file}')


        if copy_graph: graph_ = graph.copy()
        exporter_ppl = EXPORTER_INSTANCE['ppl'].value()
        export_ppl_model_file = os.path.splitext(self.export_model_file)[0] + "_ppl.onnx"
        export_ppl_cfg_file = os.path.splitext(self.export_model_file)[0] + "_ppl.json"
        exporter_ppl.export(graph=graph_, 
                        file_path=export_ppl_model_file, 
                        config_path=export_ppl_cfg_file)

        if dynamic_batch:
            onnx_model = onnx.load(export_ppl_model_file)
            dynamic_model = modify_onnx2dynamic(onnx_model)
            maca_info("Convert dynamic batch size to \'-1\'")
            onnx.save_model(dynamic_model, export_ppl_model_file)
        maca_info(f'Generating Quantization Model file:{export_ppl_model_file}')
        maca_info(f'Generating Quantization Config file:{export_ppl_cfg_file}')
        
