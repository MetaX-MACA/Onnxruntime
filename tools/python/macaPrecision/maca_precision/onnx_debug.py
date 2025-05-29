import os
import onnx
import copy
import numpy as np
import logging
import onnxruntime
import sys, getopt
import json
import subprocess

from collections import OrderedDict
from onnx import shape_inference
from onnx import numpy_helper, helper

logging.basicConfig(level=logging.INFO)

from onnx import shape_inference, TensorProto, version_converter, numpy_helper

logger = logging.getLogger("[ONNXOPTIMIZER]")

onnxfile = ''
nodefile = ''
outputfile = ''
test_data_folder=''
quant_only = False

'''
  --------------------ONNX Data Type-----------------
  enum DataType {
    UNDEFINED = 0;
    // Basic types.
    FLOAT = 1;   // float
    UINT8 = 2;   // uint8_t
    INT8 = 3;    // int8_t
    UINT16 = 4;  // uint16_t
    INT16 = 5;   // int16_t
    INT32 = 6;   // int32_t
    INT64 = 7;   // int64_t
    STRING = 8;  // string
    BOOL = 9;    // bool

    // IEEE754 half-precision floating-point format (16 bits wide).
    // This format has 1 sign bit, 5 exponent bits, and 10 mantissa bits.
    FLOAT16 = 10;

    DOUBLE = 11;
    UINT32 = 12;
    UINT64 = 13;
    COMPLEX64 = 14;     // complex with float32 real and imaginary components
    COMPLEX128 = 15;    // complex with float64 real and imaginary components

    // Non-IEEE floating-point format based on IEEE754 single-precision
    // floating-point number truncated to 16 bits.
    // This format has 1 sign bit, 8 exponent bits, and 7 mantissa bits.
    BFLOAT16 = 16;

    // Future extensions go here.
  }
'''

def convert_ort_type_2_np(ort_data_type):
    #logger.info("convert_ort_type_2_np")
    
    types = {
        1 : np.float32,
        2 : np.uint8,
        3 : np.int8,
        4 : np.uint16,
        5 : np.int16,
        6 : np.int32,
        7 : np.int64,
        8 : "",  #string
        9 : np.bool_,
        10 : np.float16,
        11 : np.float64,
        12 : np.uint32,
        13 : np.uint64,
        14 : np.complex64,
        15 : np.complex_,
        16 : ""
    }

    return types.get(ort_data_type, None)


def get_tensor_type_by_data_type(dtype):

    print('get_tensor_type_by_data_type: ', dtype.name)

    '''
    types__ = {
        np.float16 : TensorProto.FLOAT16,
        np.float32 : TensorProto.FLOAT,
        np.int8 : TensorProto.INT8,
        np.int16 : TensorProto.INT16,
        np.int32 : TensorProto.INT32,
        np.int64 : TensorProto.INT64,
        np.uint8 : TensorProto.UINT8,
        np.uint16 : TensorProto.UINT16,
        np.uint32 : TensorProto.UINT32,
        np.uint64 : TensorProto.UINT64,
        np.float64 : TensorProto.DOUBLE
    }
    '''

    types__ = {
        'float16' : TensorProto.FLOAT16,
        'float32' : TensorProto.FLOAT,
        'int8' : TensorProto.INT8,
        'int16' : TensorProto.INT16,
        'int32' : TensorProto.INT32,
        'int64' : TensorProto.INT64,
        'uint8' : TensorProto.UINT8,
        'uint16' : TensorProto.UINT16,
        'uint32' : TensorProto.UINT32,
        'uint64' : TensorProto.UINT64,
        'float64' : TensorProto.DOUBLE
    }

    t = types__.get(dtype.name, None) 
    #print('t = ', t)

    return t 

def get_optimization_level(level):
    if level == 'disable':
        return onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    if level == 'basic':
        # Constant folding and other optimizations that only use ONNX operators
        return onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
    if level == 'extended':
        # Optimizations using custom operators, excluding NCHWc and NHWC layout optimizers
        return onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    if level == 'all':
        return onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    raise ValueError('Invalid optimization level of ' + level)

def make_test_model(op_type, input_list, output_dict, attributes, init_list):
    print('make_test_model, op_type:', op_type)

    input_tensor = []
    output_tensor = []
    input_name = []
    init_list_name = [x.name for x in init_list]

    for input in input_list:
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('make_test_model, input name: ', input['name'])
        print('make_test_model, input dim: ', input['dim'])
        print('make_test_model, input shape: ', input['shape'])
        print('make_test_model, input dtype: ', input['dtype'], type(input['dtype']).__name__)

        input_name.append(input['name'])

        t = input['dtype']
        #if type(input['dtype']).__name__ == 'dtype':
        if isinstance(t, np.dtype):
            t = get_tensor_type_by_data_type(input['dtype'])

        if input['name'] not in init_list_name:
            input_tensor.append(helper.make_tensor_value_info(input['name'], t, input['shape']))

    print('-------------------------------------------------------------')
    print('make_test_model, output name: ', output_dict['name'])
    print('make_test_model, output dim: ', output_dict['dim'])
    print('make_test_model, output shape: ', output_dict['shape'])
    print('make_test_model, output dtype: ', output_dict['dtype'], type(output_dict['dtype'])) 

    t = output_dict['dtype']
    #if type(output_dict['dtype']).__name__ == 'dtype':
    if isinstance(t, np.dtype):
        t = get_tensor_type_by_data_type(output_dict['dtype'])

    output_tensor.append(helper.make_tensor_value_info(output_dict['name'], t, output_dict['shape']))

    for k, v in attributes.items():
        print('attribute:', k, ', value:', v)  

    if len(attributes) > 0 :
        node_def = helper.make_node(
                                op_type, # node name
                                input_name,  #inputs
                                [output_dict['name']], # outputs
                                **attributes
                                ) 
    else:
        node_def = helper.make_node(
                                op_type, # node name
                                input_name,  #inputs
                                [output_dict['name']], # outputs
                                ) 

    graph_def = helper.make_graph(
                                [node_def],
                                'test_model',
                                input_tensor, # graph inputs
                                output_tensor, # graph outputs
                                initializer=init_list,
                                )

    mode_def = helper.make_model(graph_def, producer_name='onnx-example', opset_imports=[helper.make_opsetid('', 11)])
    #onnx.checker.check_model(mode_def)

    save_path = './test_' + op_type
    onnxfile = 'model.onnx'
    save_path = os.path.join(save_path, onnxfile)
    onnx.save(mode_def, save_path)                                                            

def get_output(command):
    p = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    output = p.stdout.decode("ascii").strip()
    return output

def split_and_sort_output(string_list):
    string_list = string_list.split("\n")
    string_list.sort()
    return string_list

def load_onnx_test_data(path, all_inputs_shape, data_type="fp32"):
    logger.info("Parsing test data in {} ...".format(path))
    output = get_output(["find", path, "-name", "test_data*", "-type", "d"])
    test_data_set_dir = split_and_sort_output(output)
    logger.info(test_data_set_dir)

    inputs = []

    shape_flag = False
    # if not empty means input shape has been parsed before.
    if len(all_inputs_shape) > 0:
        shape_flag = True

    # find test data path
    for test_data_dir in test_data_set_dir:
        pwd = os.getcwd()
        os.chdir(test_data_dir)

        # load inputs
        output = get_output(["find", ".", "-name", "input*"])
        input_data = split_and_sort_output(output)
        logger.info(input_data)

        input_data_pb = []
        for data in input_data:
            tensor = onnx.TensorProto()
            with open(data, 'rb') as f:
                print('begin read ', data)
                tensor.ParseFromString(f.read())
                tensor_to_array = numpy_helper.to_array(tensor)
                if data_type == "fp16" and tensor_to_array.dtype == np.dtype(np.float32):
                    tensor_to_array = tensor_to_array.astype(np.float16)
                input_data_pb.append(tensor_to_array)
                if not shape_flag:
                    all_inputs_shape.append(input_data_pb[-1].shape)
                logger.info(all_inputs_shape[-1])
        inputs.append(input_data_pb)
        logger.info('Loaded {} inputs successfully.'.format(len(inputs)))

        os.chdir(pwd)

    return inputs

def get_ort_session_inputs(session, ort_input):

    sess_inputs = {}
    
    sess_inputs = {}
    for i in range(len(session.get_inputs())):
        print('get_ort_session_inputs, name', session.get_inputs()[i].name)
        sess_inputs[session.get_inputs()[i].name] = ort_input[i]

    return sess_inputs

def prepare_dir(path):  # type: (Text) -> None
    #if os.path.exists(path):
    #    shutil.rmtree(path)

    if os.path.exists(path) == False:
        os.makedirs(path)

def get_cosine(gpu_array, cpu_array):
    x = np.square(gpu_array)
    x = np.sum(x) 
    x = np.sqrt(x)

    y = np.square(cpu_array)
    y = np.sum(y) 
    y = np.sqrt(y)

    z = gpu_array * cpu_array
    z = sum(z)

    cosine_sim  = (z + 1e-7) / ((x * x) + 1e-7) # eps

    cosine = np.mean(cosine_sim)

    cosine = max(cosine, 1.0)

    cosine = 1.0 - cosine

    cosine = max(0, cosine)

    print('cosine:', cosine)

    return cosine  

def get_mse(gpu_array, cpu_array):
    diff_array = np.subtract(cpu_array, gpu_array)
    x = np.square(diff_array)
    mse = np.mean(x)

    print('mse:', mse)

    return mse  

def get_snr(gpu_array, cpu_array):
    diff_array = np.subtract(cpu_array, gpu_array)
    x = np.square(diff_array)
    x = np.sum(x)

    y = np.square(cpu_array)
    y = np.sum(y) 

    snr = (x) / (y + 1e-7)

    snr = np.mean(snr)

    print('snr:', snr)

    return snr  
    
precision_cmp_method = {
    "mse": get_mse,
    "cosine": get_cosine,
    "snr": get_snr
}

precision_cmp_str = 'snr'
precision_threshold = 0.1 

def compare_result(ort_inputs, ort_outs_cpu, ort_outs_gpu, node_list, initializer): 
    match=True 
    seq = 0

    save_path = './'
    data_set_dir = './'

    for k,v in ort_outs_gpu.items():
        #print(k, ':', v.shape)
        #print('v.type:', v.__class__)
        assert v.__class__  == np.ndarray
        #print('ndim: ', v.ndim)
        #print('dtype: ', v.dtype)
        if v.__class__  == np.ndarray :
            print('tensor name: ', k)
            c=ort_outs_cpu[k].flatten()
            print('cpu: ', c)
            assert k in ort_outs_gpu
            g=v.flatten()
            print('gpu: ', g)
            
            #over the tensor all value is 0
            b = set(g)
            if len(b) == 1 and g[0] == 0:
                continue

            diff=np.subtract(c, g, dtype=np.float64)
            index=0

            '''
            data_type = str(g.dtype)
            assert g.dtype==c.dtype
            if data_type.startswith("float"):
                error_range = [-0.1, 0.1]
            else:
                error_range = [-3, 3]
            '''
            cmp_value = precision_cmp_method[precision_cmp_str](c, g) #get_snr(diff, c)
            if cmp_value > precision_threshold:
                #for i in diff:
                #if i > error_range[1] or i < error_range[0]:
                match=False
                print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                print('WARNING: output ', k, ' is abnormal, please check it~~')
                #print('cpu val: ', c[index], 'gpu val: ', g[index])

                node_abnormal = {}
                node_abnormal_attributes = {}
                output_abnornal = {}

                for node in node_list :
                    #print('node_output: ', node['output'], ', node_name: ', node['name'])
                    if k in node['output'] :
                        node_abnormal = node
                        print('Dismatch node name: ', node['name'], ', input:',  node['input'])

                        save_path = './test_' + node['op_type']
                        prepare_dir(save_path)

                        data_set_dir = os.path.join(save_path, 'test_data_set_0')
                        prepare_dir(data_set_dir)

                        with open(os.path.join(data_set_dir, 'output_0.pb'), 'wb') as f:
                            f.write(numpy_helper.from_array(ort_outs_cpu[k], k).SerializeToString())

                        if len(node['attribute']) > 0 :
                            print('+++++++++++++++++++ got attribute:')
                            for attr  in node['attribute']:
                                #print('attr.name: ', attr.name, ', attr.type: ', attr.type)
                                value = helper.get_attribute_value(attr)
                                print('parse attr: ', attr.name, attr.type, value)
                                node_abnormal_attributes[attr.name] = value

                        output_abnornal['dim'] = v.ndim
                        output_abnornal['shape'] = v.shape
                        output_abnornal['dtype'] = v.dtype
                        output_abnornal['name'] = k

                        break

                print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')  
            
                break
                
                #index = index + 1
                    
        if match == False :
                break      
            
        seq = seq + 1            
           
    if match == True :
       print('===================================================')
       print('===================================================')
       print('Congratulations, it works well as expected')
    else:
        abnormal_input_counts = len(node_abnormal['input'])

        for k, v in node_abnormal_attributes.items():
            print('node_abnormal_attributes: ', k, ', value: ', v)

        input_abnormal = []

        j = 0
        for k,v in ort_inputs.items():
            if k in node_abnormal['input']:
                print('find ', k, 'in', node_abnormal['input'])
                with open(os.path.join(data_set_dir, 'input_{}.pb'.format(j)), 'wb') as f:
                    f.write(numpy_helper.from_array(v, k).SerializeToString())
                    j = j + 1

                d={}
                d['dim'] = v.ndim
                d['shape'] = v.shape
                d['dtype'] = v.dtype
                d['name'] = k
                input_abnormal.append(d)

        for k,v in ort_outs_cpu.items():
            if k in node_abnormal['input']:
                print('find ', k, 'in', node_abnormal['input'])
                with open(os.path.join(data_set_dir, 'input_{}.pb'.format(j)), 'wb') as f:
                    f.write(numpy_helper.from_array(v, k).SerializeToString())
                    j = j + 1

                d={}
                d['dim'] = v.ndim
                d['shape'] = v.shape
                d['dtype'] = v.dtype
                d['name'] = k
                input_abnormal.append(d)

        init_list = []  

        if j != abnormal_input_counts:
            for input in node_abnormal['input'] :  
                for init in initializer:
                    if init.name == input:
                        print('got node from initializer for ', init.name, ', data_type: ', init.data_type, ', dims: ', init.dims)
                        array = numpy_helper.to_array(init)
                        with open(os.path.join(data_set_dir, 'input_{}.pb'.format(j)), 'wb') as f:
                            f.write(numpy_helper.from_array(array, input).SerializeToString())
                            j = j + 1 

                        d={}
                        d['dim'] = len(init.dims)
                        d['shape'] = array.shape#init.dims
                        d['dtype'] = init.data_type
                        d['name'] = init.name
                        input_abnormal.append(d)
                        init_list.append(init)  

        make_test_model(node_abnormal['op_type'], input_abnormal, output_abnornal, node_abnormal_attributes, init_list)           


def generate_onnx_model_random_input(model):
    ort_inputs = {}

    initializer = []
    for init in model.graph.initializer:
        print('got init: ', init.name)
        initializer.append(init.name)

    for input in model.graph.input:
        if input.name not in initializer :
            input_shape = input.type.tensor_type.shape.dim
            type = input.type.tensor_type.elem_type

            print('-----input name is', input.name)
            #print('-----input raw_data is', input.type.tensor_type.raw_data)

            data_shape = [x.dim_value for x in input_shape]
            data_shape_new = []

            for x in data_shape:
                if x == 0:
                    data_shape_new.append(1)
                elif x == -1:
                    n = np.random.randint(1,5)
                    data_shape_new.append(n)    
                else:
                    data_shape_new.append(x)

            data_shape = data_shape_new
            data_array = np.array(np.random.random(data_shape), dtype = convert_ort_type_2_np(type))

            ort_inputs[input.name] = data_array

    return ort_inputs

def test_model_by_onnxruntime(model):

    ori_outputs = [x.name for x in model.graph.output]
    ori_outputs_backup=model.graph.output[:]
    #print('ori:', ori_outputs)
    
    node_list=[]
    
    del model.graph.output[:]

    for node in model.graph.node:
        dict={"name":node.name, "input":node.input, "output":node.output, "op_type": node.op_type, "attribute":node.attribute}
        node_list.append(dict)
        for output in node.output:
            if output not in ori_outputs:
                model.graph.output.extend([onnx.ValueInfoProto(name=output)])
                
    model.graph.output.extend(ori_outputs_backup)
                
    #for i, val in enumerate(node_list):
    #    print('num:', i, ', node:', val) 

    if nodefile != '' :    
      with open(nodefile, 'w') as f:
         f.write(str(node_list))

    EP_list = ['CPUExecutionProvider']
    ort_session = onnxruntime.InferenceSession(model.SerializeToString(), providers=EP_list)

    ort_inputs = generate_onnx_model_random_input(model)

    outputs = [x.name for x in ort_session.get_outputs()]

    print('output list:')
    print(outputs)

    if test_data_folder != '':
        inputs = load_onnx_test_data(test_data_folder, [])
        for input in inputs:
            ort_inputs = get_ort_session_inputs(ort_session, input)
            break #only once

    print('begin run cpu......')

    ort_outs = ort_session.run(outputs, ort_inputs)
    
    #print(ort_outs)

    ort_outs = OrderedDict(zip(outputs, ort_outs))

    #np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(threshold=10)

    out_list=[]
    out_dict={}

    for k,v in ort_outs.items():
       #print('cpu---- ', k, ':', v)
       #print('v.type:', v.__class__)
       if True == hasattr(v, 'tolist') :       
           dict={"output":k, "value":v.tolist()}
       else:
           dict={"output":k, "value":v} 
           
       out_list.append(dict)
       
    #print('list:', out_list)
    
    if outputfile != '' : 
      with open(outputfile, 'w') as f:
         f.write(str(out_list))

    print('-----------------------------------------------------------')
    print('-----------------------------------------------------------')
    
    print('begin run gpu......')
    
    #just for test,if you want to generate dismatch msg, uncomment belowing lines
    #img_array = np.array(np.random.random(image_shape), dtype = convert_ort_type_2_np(type))
    #img = img_array
    #for i, input_ele in enumerate(ort_session.get_inputs()):
    #    ort_inputs[input_ele.name] = img
    
    #just for test, you should use MACAExecutionProvider
    #EP_list = ['CPUExecutionProvider']
    EP_list = ['MACAExecutionProvider']
    
    ort_session_gpu = onnxruntime.InferenceSession(model.SerializeToString(), providers=EP_list)
    
    ort_outs_gpu = ort_session_gpu.run(outputs, ort_inputs)
    
    ort_outs_gpu = OrderedDict(zip(outputs, ort_outs_gpu))

    out_list_gpu=[]

    for k,v in ort_outs_gpu.items():
       #print('gpu--- ', k, ':', v)
       #print('v.type:', v.__class__)
       if True == hasattr(v, 'tolist') :       
           dict={"output":k, "value":v.tolist()}
       else:
           dict={"output":k, "value":v} 
           
       out_list_gpu.append(dict)
    
    #print('list:', out_list)
    
    logger.info("Test model by onnxruntime finish")
    
    compare_result(ort_inputs, ort_outs, ort_outs_gpu, node_list, model.graph.initializer)

    #del model.graph.output[:]

    #model.graph.output.extend(ori_output)

    return ort_outs


def make_acuity_input(input_src, data_path, input_name):    
    with open(data_path, 'w') as f:
        f.write('{}\n'.format(input_name))
    f.close()
    input_data = []
    for k, v in input_src.items():
        input_data.append(v)
    
    input_data = np.array(input_data)
    np.save(input_name[:-4], input_data)
    return True

def load_acuity_tensor(tensor_path):
    with open(tensor_path, 'r') as f:
        tensor_strs = f.readlines()
    f.close()

    tensor_float = []
    for t in tensor_strs:
        tensor_float.append(float(t))
    tensor_float = np.array(tensor_float)
    return tensor_float

def match_output_with_node(node_list):
    node_output_dict = {}
    quant_output_dict = {}
    for node in node_list:
        outputs = node['output']
        if node['op_type'] == 'QuantizeLinear':
            quant_output_dict[node['input'][0]] = outputs
        elif node['op_type'] != 'DequantizeLinear':
            node_output_dict[node['name']] = outputs

    all_keys = list(quant_output_dict.keys())
    for key in all_keys:
        for key_, value_ in node_output_dict.items():
            if key in value_:
                if key_ not in quant_output_dict:
                    quant_output_dict[key_] = quant_output_dict.pop(key)
                    break
    return node_output_dict, quant_output_dict

def run_onnx(onnx_path, compare_mode):
    model = onnx.load(onnx_path)
    ori_outputs = [x.name for x in model.graph.output]
    ori_outputs_backup=model.graph.output[:]
    
    node_list=[]
    
    del model.graph.output[:]

    for node in model.graph.node:
        dict={"name":node.name, "input":node.input, "output":node.output, "op_type": node.op_type, "attribute":node.attribute}
        node_list.append(dict)
        for output in node.output:
            if output not in ori_outputs:
                model.graph.output.extend([onnx.ValueInfoProto(name=output)])
                
    model.graph.output.extend(ori_outputs_backup)

    if compare_mode == 'MACA-Acuity':
        EP_list = ['MACAExecutionProvider']
    else:
        EP_list = ['CPUExecutionProvider']
    ort_session = onnxruntime.InferenceSession(model.SerializeToString(), providers=EP_list)

    ort_inputs = generate_onnx_model_random_input(model)

    outputs = [x.name for x in ort_session.get_outputs()]
    
    ort_outs = ort_session.run(outputs, ort_inputs)
    
    ort_outs = OrderedDict(zip(outputs, ort_outs))
    node_output_dict, quant_output_dict = match_output_with_node(node_list)

    return ort_inputs, ort_outs, node_output_dict, quant_output_dict

def maca_info(info: str):
    print(f'\033[32m{info}\033[0m')

def maca_warning(info: str):
    print(f'\033[33m{info}\033[0m')

def maca_error(info: str):
    print(f'\033[31m{info}\033[0m')

def test_model_by_acuity(onnx_path, compare_mode):
    assert compare_mode in ['CPU-Acuity', 'MACA-Acuity'], 'compare_mode error, please check!'

    ort_inputs, ort_outputs, node_output_dict, quant_output_dict = run_onnx(onnx_path, compare_mode)
    try:
        from acuitylib.vsi_nn import VSInn
        nn = VSInn(project_dir=os.getcwd())
    except:
        maca_error('\nYou must install acuity-tool before run this python file!\n')
        sys.exit()
    inputs = ''
    input_size_list = ''

    for input in ort_inputs.keys():
        inputs += '{},'.format(input)
        input_shape = str(ort_inputs[input].shape)
        input_shape = input_shape.replace(', ', ',')
        input_shape = input_shape.replace('(', '')
        input_shape = input_shape.replace(')', '')
        input_size_list += '{}#'.format(input_shape)
    inputs = inputs[:-1]
    input_size_list = input_size_list[:-1]

    outputs = ''
    for output in ort_outputs.keys():
        outputs += '{},'.format(output)
    outputs = outputs[:-1]
    outputs = outputs.split(',')[-1]

    acu_net = nn.load_onnx(onnx_path, inputs=inputs, outputs=outputs,
                           input_size_list=input_size_list, size_with_batch='True')
    
    # make input info with acuity format
    data_path = './tmp_onnx.txt'
    input_file = './acuity_input.npy'
    dump_path = './test_dump'
    feat_path = './output_path'
    if os.path.exists(dump_path):
        os.system('rm -rf {}/*'.format(dump_path))
    else:
        os.mkdir(dump_path)
    make_acuity_input(ort_inputs, data_path, input_file)
    nn.set_database(acu_net, dataset_files=data_path, dataset_type='TEXT')
    preprocess_dict = {}
    for input_name, input_value in ort_inputs.items():
        tmp_dict = {}
        tmp_dict['shape'] = list(input_value.shape)
        tmp_dict['mean'] = [0.0 for i in range(input_value.shape[1])]
        tmp_dict['scale'] = 1.0
        preprocess_dict[input_name] = tmp_dict
    nn.set_preprocess(acu_net, preprocess_dict)

    result = nn.inference(acu_net, output_path=feat_path, iterators=1, device='CPU')
    
    if len(quant_output_dict) > 0:
        nn.dump(acu_net, save_quantize=True, output_path=dump_path)
    nn.dump(acu_net, output_path=dump_path, save_file_type='tensor')

    # compare acuity output with onnxruntime
    acuity_result_list = os.listdir(dump_path)
    layer_result_dict = {}
    for acuity_tensor_name in acuity_result_list:
        quant_tensor = False
        if acuity_tensor_name.split('.')[1] == 'qnt':
            quant_tensor = True
        if quant_only and not quant_tensor and len(quant_output_dict) > 0:
            continue
        if len(acuity_tensor_name.split('.')) > 2:
            continue
        tensor_infos = acuity_tensor_name.split('.')[0].split('_')
        node_name = '{}_{}'.format(tensor_infos[1], tensor_infos[2])
        if tensor_infos[4][:3] != 'out' or tensor_infos[0] == 'attach':
            continue
        output_idx = int(tensor_infos[4][3:])
        tensor_shape = []
        for info in tensor_infos[6:]:
            tensor_shape.append(int(info))

        if quant_tensor:
            if node_name not in quant_output_dict:
                continue
            ort_output_name = quant_output_dict[node_name][output_idx]
        else:
            if node_name not in node_output_dict:
                continue
            ort_output_name = node_output_dict[node_name][output_idx]
        ort_output = ort_outputs[ort_output_name]
        ort_output_shape = list(ort_output.shape)

        mismatch = False
        for s1, s2 in zip(tensor_shape, ort_output_shape):
            if s1 != s2:
                print('Output shape mismatch!')
                mismatch = True
                break
        if mismatch:
            continue
                
        ort_output = ort_output.reshape(-1)
        tensor_path = os.path.join(dump_path, acuity_tensor_name)
        acuity_output = load_acuity_tensor(tensor_path)
        precision = precision_cmp_method[precision_cmp_str](ort_output, acuity_output)
        # print('Ort output: {} <--------> Acuity output: {} | Similarity: {}'.format(ort_output_name, acuity_tensor_name, precision))
        ort_output_name = '{}->{}'.format(node_name, ort_output_name)
        result_dict = {'ort_name' : ort_output_name,
                       'precision' : precision }
                       
        layer_result_dict[acuity_tensor_name] = result_dict

    all_tensor_list = list(layer_result_dict.keys())
    all_tensor_list.sort()
    print('\n\n\n')
    if compare_mode == 'MACA-Acuity':
        logger.info('Acuity compare with ORT MACAEP:')
    else:
        logger.info('Acuity compare with ORT CPUEP:')
    for tr_name in all_tensor_list:
        if compare_mode == 'MACA-Acuity':
            info = 'Acuity: {: <80} ORT-MACA: {: <30} {}:{}'.format(
                        tr_name, layer_result_dict[tr_name]['ort_name'], precision_cmp_str, layer_result_dict[tr_name]['precision'])
        else:            
            info = 'Acuity: {: <80} ORT-CPU: {: <30} {}:{}'.format(
                        tr_name, layer_result_dict[tr_name]['ort_name'], precision_cmp_str, layer_result_dict[tr_name]['precision'])
        precision = layer_result_dict[tr_name]['precision']
        if precision > precision_threshold:
            maca_warning(info)
        else:
            maca_info(info)

    # clear tmp file
    os.system('rm -rf {}/*'.format(feat_path))
    os.system('rm -rf {}/*'.format(dump_path))
    os.system('rm -r {}'.format(feat_path))
    os.system('rm -r {}'.format(dump_path))    
    os.system('rm {}'.format(data_path))
    os.system('rm {}'.format(input_file))   

def usage():
    print('python -m maca_precision -i <onnxfile> -m <CPU-MACA>')
    print('or') 
    print('python -m maca_precision -f <test_data_folder> -m <CPU-MACA>')
    print('If you only need quant info when compare quantized model with acuity mode, you can use like this')    
    print('python onnx_debug.py -i <onnxfile> -m <MACA-Acuity> -q')
   #print('python onnx_debug.py -i <onnxfile> -n <nodefile> -o <outputfile>', '({})'.format('Not yet support for now'))    

def main(argv):
   global onnxfile
   global nodefile
   global outputfile
   global test_data_folder
   global precision_cmp_str
   global precision_threshold
   global optimization_level
   global quant_only
   compare_mode = ''
   
   optimization_level  = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

   try:
      opts, args = getopt.getopt(argv,"hi:n:o:f:p:m:q",["onnx=", "nfile=", "ofile=", "folder=", "opt_level="])
   except getopt.GetoptError:
      usage()
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         usage()
         sys.exit()
      elif opt in ("-i", "--onnx"):
         onnxfile = arg
      elif opt in ("-n", "--nfile"):
         nodefile = arg   
      elif opt in ("-o", "--ofile"):
         outputfile = arg
      elif opt in ("-f", "--folder"):
         test_data_folder = arg
      elif opt in ("-c", "--compare_method"):
         precision_cmp_str = arg         
      elif opt in ("-p", "--opt_level"):
         print(arg)
         optimization_level = get_optimization_level(arg)
      elif opt in ("-m", "--compare_mode"):
         compare_mode = arg
      elif opt in ("-q", "--quant_only"):
          quant_only = True

   print('onnx file path: ', onnxruntime.__file__)    
         
   if onnxfile == '' and test_data_folder == '':
      print('Warning: you should specify onnxfile(-i) or test_data_folder(-f)') 
      usage()
      sys.exit()

   if onnxfile != '' and  test_data_folder != '' :
       print('Warning: you should not specify both onnxfile and test_data_folder(use -i or -f only)')
       sys.exit()

   if compare_mode == '' :
       compare_mode = 'CPU-MACA'
       
   print('model file: ', onnxfile)
   print('node file: ', nodefile)
   print('output file: ', outputfile)
   print('test_data_folder: ', test_data_folder)
   print('compare method: ', precision_cmp_str)
   print('compare mode: ', compare_mode)
   print('quant node only: ', str(quant_only))

   if precision_cmp_str == 'snr':
        precision_threshold = 0.00001
   elif precision_cmp_str == 'mse':
        precision_threshold = 0.0
   elif precision_cmp_str == 'cosine':
        precision_threshold = 0.03
   else:
        print('precision_cmp_str can only be one of [\'snr\', \'mse\', \'cosine\']')
        sys.exit()

   if compare_mode not in ['MACA-Acuity', 'CPU-Acuity', 'CPU-MACA']:
        print('check_mode can only be one of [\'CPU-Acuity\', \'MACA-Acuity\', \'CPU-MACA\']')
        sys.exit()

   if test_data_folder != '' :
       if os.path.exists(test_data_folder) == False or os.path.isfile(test_data_folder):
           print('ERROR: ', test_data_folder, ' is not exist or is not a folder')
           sys.exit()

       if test_data_folder.endswith('/'):
           onnxfile = test_data_folder + 'model.onnx'
       else:
           onnxfile = test_data_folder + '/' + 'model.onnx'

       if os.path.exists(onnxfile) == False :
           print('ERROR: onnx file: ', onnxfile, ' is not exist')
           sys.exit()    
   
   if optimization_level != onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL:
       sess_options = onnxruntime.SessionOptions()
       # Set graph optimization level
       sess_options.graph_optimization_level = optimization_level
       sess_options.optimized_model_filepath = "./opt_model.onnx"
       session_cpu = onnxruntime.InferenceSession(onnxfile, sess_options=sess_options, providers= ["CPUExecutionProvider"])
       onnxfile = sess_options.optimized_model_filepath

   if compare_mode == 'CPU-MACA':
       onnx_model = onnx.load(onnxfile)   
       test_model_by_onnxruntime(onnx_model)
   else:
       test_model_by_acuity(onnxfile, compare_mode)

if __name__ == "__main__":
   main(sys.argv[1:])
