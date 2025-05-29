import os
import sys
from regex import F
from sklearn.metrics import max_error

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
import onnx
import glob
import os.path as osp
from onnx import numpy_helper
from maca_quantizer.core.config import REGISTER_TABLE 
from maca_quantizer.utils.utils import maca_info, maca_warning

from example_config import *

from ppq import *
from ppq.api import *
from ppq.lib.common import *
from ppq.quantization.measure import torch_snr_error




TARGET_PLATFORM = TargetPlatform.ONNX

def _register_handler(register_type):

    for key, value in REGISTER_TABLE[register_type].items():
        if value is None: continue        
        # register quantizer
        if key=="parser":
            register_network_parser(parser=value, framework=NetworkFramework.ONNX)
        if key=="operation":  
            for op_type, op_forwad in value.items():
                register_operation_handler(handler=op_forwad, operation_type=op_type, platform=TARGET_PLATFORM)
                register_operation_handler(handler=op_forwad, operation_type=op_type, platform=TargetPlatform.UNSPECIFIED)
                register_operation_handler(handler=op_forwad, operation_type=op_type, platform=TargetPlatform.FP32)
                print(f"Register operstion: {op_type:10} --> {op_forwad}")


def load_input_data(input_path):
    print("load test data:")
    #onlu select one test case
    test_data_dirs = glob.glob(os.path.join(input_path, "test_data_set_*"))
    
    test_data_inputs = []
    test_data_outputs = []
    for test_data_dir in test_data_dirs:
        data_inputs,data_outputs = [], []
        if os.path.isdir(test_data_dir):
            input_files = glob.glob(os.path.join(test_data_dir, "input_*.pb"))
            for file in input_files:
                print('load file:', file)
                input_tensor = onnx.TensorProto()
                with open(file, 'rb') as f:
                    input_tensor.ParseFromString(f.read())
                    input_array = numpy_helper.to_array(input_tensor)
                    torch_tensor = torch.from_numpy(input_array)
                    data_inputs.append(torch_tensor)
            output_files = glob.glob(os.path.join(test_data_dir, "output_*.pb"))
            for file in output_files:
                print('load file:', file)
                # output_tensor = onnx.TensorProto()
                output_tensor = onnx.SequenceProto()
                
                with open(file, 'rb') as f:
                    output_tensor.ParseFromString(f.read())
                    output_array = [numpy_helper.to_array(t) for t in output_tensor.tensor_values]
                    torch_tensor = [torch.from_numpy(t) for t in output_array]
                    # output_array = numpy_helper.to_array(output_tensor)
                    # torch_tensor = torch.from_numpy(output_array)
                    data_outputs.append(torch_tensor)
        test_data_inputs.append(data_inputs)
        test_data_outputs.append(data_outputs)

    return test_data_inputs, test_data_outputs

def test_node(dir_path):
    model_path = osp.join(dir_path, "model.onnx")
    graph = load_onnx_graph(model_path)
    executor = TorchExecutor(graph=graph, device='cpu')

    # build an executor:
    in_data, out_data = load_input_data(dir_path)
    for i in range(len(in_data)):
        data = in_data[i]
        gold_data = out_data[i]
        tensor = executor.forward(inputs=data)
        print(f"PPQ   data: \n {tensor}")
        print(f"Gold  data: \n {gold_data}")
        # print(tensor)
        if (gold_data[0]==tensor[0]).all():
            return True, 0, 0
        else:
            snr = torch_snr_error(gold_data[0], tensor[0], reduction='mean')
            max_error = (gold_data[0]-tensor[0]).max()
            return False, snr, max_error


def test_node_2(dir_path):
    model_path = osp.join(dir_path, "model.onnx")
    graph = load_onnx_graph(model_path)
    executor = TorchExecutor(graph=graph, device='cpu')

    # build an executor:
    in_data, out_data = load_input_data(dir_path)
    for i in range(len(in_data)):
        data = in_data[i]
        gold_data = out_data[i]
        tensor = executor.forward(inputs=data)
        print(f"PPQ   data: \n {tensor}")
        print(f"Gold  data: \n {gold_data}")

# ============================================== split ==============================================

def test_base(test_name_list):
    for name in test_name_list:
        print(f"========================== {name} ==========================")
        test_dir = osp.join(onnx_test_dir, name)
        pass_flag, snr_value, max_error = test_node(test_dir)
        if pass_flag:
            maca_info(f"Test Pass: ********* {name} *********")
        else:
            maca_warning(f"Test Failed: {name} !!!!!!!!!!!!!!!!! ")
            maca_warning(f"Max error  : {max_error} !!!!!!!!!!!!!!!!! ")
            maca_warning(f"SNR value  : {snr_value} !!!!!!!!!!!!!!!!! ")


def test_base_2(test_name_list):
    for name in test_name_list:
        print(f"========================== {name} ==========================")
        test_dir = osp.join(onnx_test_dir_2, name)
        test_node_2(test_dir)


def test_argmax():
    test_name_list = argmax_name_list
    for name in test_name_list:
        print(f"========================== {name} ==========================")
        test_dir = osp.join(onnx_test_dir, name)
        test_node(test_dir)

def test_argmin():
    test_name_list = argmin_name_list
    for name in test_name_list:
        print(f"========================== {name} ==========================")
        test_dir = osp.join(onnx_test_dir, name)
        test_node(test_dir)





if __name__=="__main__":
    _register_handler('mxc_pplcuda')
    onnx_test_dir = "/workspace/inference_engine/source_code/onnx/onnx/backend/test/data/node"
    onnx_test_dir_2 = "/workspace/source/node_test/"

    # high level operations
    # test_base(argmax_name_list)
    # test_base(argmin_name_list)
    # test_base(averagepool_name_list)
    # test_base(ceil_name_list)
    # test_base(depthtospace_name_list)
    # test_base(cumsum_name_list)
    

    # middle level operations
    # test_base(einsum_name_list)
    # test_base(maxunpool_tst_list)
    # test_base(reducemin_tst_list)
    # test_base(reduceprod_tst_list)
    # test_base(scatter_elements_tst_list)
    # test_base(xor_tst_list)
    
    test_base_2(split2squence_tst_list)


    # tst_list = ["test_scatter_elements_with_duplicate_indices"]
    # test_base(tst_list)


