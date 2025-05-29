import os, time
import onnx
import copy
import numpy as np
import logging
import onnxruntime
import sys, getopt
import json
import subprocess
import pickle
#import cv2

from collections import OrderedDict
from onnx import shape_inference
from onnx import numpy_helper, helper

logging.basicConfig(level=logging.INFO)

from onnx import shape_inference, TensorProto, version_converter, numpy_helper

np.random.seed(5)

logger = logging.getLogger("[ONNXOPTIMIZER]")

onnxfile = ''
test_data_folder=''

output_batch = 5
input_batch = 1

#target_ep = 'CPUExecutionProvider'
target_ep = 'MACAExecutionProvider'

mismatch_list = []

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

def check_input(gpu_array, cpu_array, method):
    is_nan1 = np.isnan(gpu_array)
    is_nan2 = np.isnan(cpu_array)

    #print('is_nan1:', is_nan1, type(is_nan1))
    #print('is_nan2:', is_nan2, type(is_nan2))

    if isinstance(is_nan1, np.ndarray):
        r1 = True in is_nan1
        if r1 == True:
            print('Warnning: your gpu data has NaN, please check it!')
            print('+++++{}: 9999'.format(method))
            return 9999

    if isinstance(is_nan1, np.bool_):
        if is_nan1 == True:
            print('Warnning: your gpu data has NaN, please check it!')
            print('+++++{}: 9999'.format(method))
            return 9999

    if isinstance(is_nan2, np.ndarray):
        r2 = True in is_nan2
        if r2 == True:
            print('Warnning: your cpu data has NaN, please check it!')
            print('+++++{}: 8888'.format(method))
            return 8888    

    if isinstance(is_nan2, np.bool_):
        if is_nan2 == True:
            print('Warnning: your cpu data has NaN, please check it!')
            print('+++++{}: 8888'.format(method))
            return 8888   

    is_neg_inf1 = np.isneginf(gpu_array)
    is_neg_inf2 = np.isneginf(cpu_array)

    if isinstance(is_neg_inf1, np.ndarray):
        r1 = True in is_neg_inf1
        if r1 == True:
            print('Warnning: your gpu data has -inf, please check it!')
            print('+++++{}: 9999'.format(method))
            return 9999

    if isinstance(is_neg_inf1, np.bool_):
        if is_neg_inf1 == True:
            print('Warnning: your gpu data has -inf, please check it!')
            print('+++++{}: 9999'.format(method))
            return 9999

    if isinstance(is_neg_inf2, np.ndarray):
        r2 = True in is_neg_inf2
        if r2 == True:
            print('Warnning: your cpu data has -inf, please check it!')
            print('+++++{}: 8888'.format(method))
            return 8888  

    if isinstance(is_neg_inf2, np.bool_):
        if is_neg_inf2 == True:
            print('Warnning: your cpu data has -inf, please check it!')
            print('+++++{}: 8888'.format(method))
            return 8888  

    is_pos_inf1 = np.isposinf(gpu_array)
    is_pos_inf2 = np.isposinf(cpu_array)

    if isinstance(is_pos_inf1, np.ndarray):
        r1 = True in is_pos_inf1
        if r1 == True:
            print('Warnning: your gpu data has +inf, please check it!')
            print('+++++{}: 9999'.format(method))
            return 9999

    if isinstance(is_pos_inf1, np.bool_):
        if is_pos_inf1 == True:
            print('Warnning: your gpu data has +inf, please check it!')
            print('+++++{}: 9999'.format(method))
            return 9999

    if isinstance(is_pos_inf2, np.ndarray):
        r2 = True in is_pos_inf2
        if r2 == True:
            print('Warnning: your cpu data has +inf, please check it!')
            print('+++++{}: 8888'.format(method))
            return 8888

    if isinstance(is_pos_inf2, np.bool_):
        if is_pos_inf2 == True:
            print('Warnning: your cpu data has +inf, please check it!')
            print('+++++{}: 8888'.format(method))
            return 8888

    return 0   

def get_cosine(gpu_array, cpu_array):
    gpu_array = np.float64(gpu_array)
    cpu_array = np.float64(cpu_array)

    r = check_input(gpu_array, cpu_array, 'cosine')
    if r != 0:
        return r

    x = np.square(gpu_array)
    x = np.sum(x) 
    x = np.sqrt(x)

    y = np.square(cpu_array)
    y = np.sum(y) 
    y = np.sqrt(y)

    z = gpu_array * cpu_array
    z = sum(z)

    print('x y z:', x, y, z)

    cosine_sim  = (z + 1e-7) / ((x * y) + 1e-7) # eps

    cosine_sim = max(cosine_sim, 1.0)

    cosine = np.mean(cosine_sim)

    print('-----cosine:', cosine)

    #cosine = max(cosine, 1.0)

    cosine = 1.0 - cosine

    #cosine = max(0, cosine)

    print('+++++cosine:', cosine)

    return cosine  

def get_mse(gpu_array, cpu_array):
    gpu_array = np.float64(gpu_array)
    cpu_array = np.float64(cpu_array)

    r = check_input(gpu_array, cpu_array, 'mse')
    if r != 0:
        return r

    diff_array = np.subtract(cpu_array, gpu_array)
    x = np.square(diff_array)
    mse = np.mean(x)

    print('mse:', mse)

    return mse  

def get_snr(gpu_array, cpu_array):
    gpu_array = np.float64(gpu_array)
    cpu_array = np.float64(cpu_array)
 
    r = check_input(gpu_array, cpu_array, 'snr')
    if r != 0:
        return r

    diff_array = np.subtract(cpu_array, gpu_array)
    x = np.square(diff_array)
    x = np.sum(x)

    y = np.square(cpu_array)
    y = np.sum(y) 

    snr = (x) / (y + 1e-7)

    #snr = np.mean(snr)

    print('snr:', snr)

    return snr  
    
precision_cmp_method = {
    "mse": get_mse,
    "cosine": get_cosine,
    "snr": get_snr
}

precision_cmp_str = 'snr'
precision_threshold = 0.001
input_npy = '' 

def compare_result(ort_outs_cpu, ort_outs_gpu, node_list, initializer, only_compare=False): 
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
            c=ort_outs_cpu[k].flatten()
            #cc=c.tolist()
            #print('cpu: ', cc[:256])
            assert k in ort_outs_gpu
            g=v.flatten()
            #gg=g.tolist()
            #print('gpu: ', gg[:256])
            diff=np.subtract(c, g, dtype=np.float64)
            print('Current ouput name is :', k)
            cmp_value = precision_cmp_method[precision_cmp_str](g, c) 
            index=0
            data_type = str(g.dtype)
            assert g.dtype==c.dtype
            if data_type.startswith("float"):
                error_range = [-0.1, 0.1]
            else:
                error_range = [-3, 3]

            if cmp_value > precision_threshold:
                match=False
                print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                print('WARNING: output ', k, ' is abnormal, please check it~~')
                #print('cpu val: ', c[index], 'gpu val: ', g[index])

                if only_compare == True:
                    return False

                node_abnormal = {}
                node_abnormal_attributes = {}
                output_abnornal = {}

                for node in node_list :
                    #print('node_output: ', node['output'], ', node_name: ', node['name'])
                    if k in node['output'] :
                        node_abnormal = node
                        print('Dismatch node name: ', node['name'], ', input:',  node['input'])
                        
                        #break

                print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')  
            
                #break
                
                #index = index + 1
                    
        if match == False :
            #break
            print('Current node abnormal')      
            
        seq = seq + 1            
           
    if match == True :
       print('===================================================')
       print('===================================================')
       print('Congratulations, it works well as expected')
       return True
    else:
        return False          

def generate_onnx_model_random_input(model):
    ort_inputs = {}
    #image_path = 'ILSVRC2012_val_00000001.JPEG'

    initializer = []
    for init in model.graph.initializer:
        #print('got init: ', init.name)
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
                    n = np.random.randint(1, 6)
                    data_shape_new.append(1)    
                else:
                    data_shape_new.append(x)

            data_shape = data_shape_new
            #print('----data_shape:', data_shape)
            data_array = np.array(np.random.random(data_shape), dtype = convert_ort_type_2_np(type))

            '''
            image =cv2.imread(image_path)
            image = cv2.resize(image, (data_shape[-1], data_shape[-2]), interpolation=cv2.INTER_LINEAR)
            image = image.transpose([2,0,1])
            print('image.shape', image.shape)
            data_array[0] = image
            '''

            ort_inputs[input.name] = data_array

            if input_npy != '':
                print('using numpy data')
                ort_inputs[input.name] = np.load(input_npy)

    return ort_inputs

def compare_model_output(model):
    EP_list = ['CPUExecutionProvider']
    ort_session = onnxruntime.InferenceSession(model.SerializeToString(), providers=EP_list)

    ort_inputs = generate_onnx_model_random_input(model)

    outputs = [x.name for x in ort_session.get_outputs()]

    print('compare_model_output,output list:')
    print(outputs)

    print('compare_model_output, begin run cpu......')

    ort_outs = ort_session.run(outputs, ort_inputs)
    
    # print(ort_outs)

    ort_outs = OrderedDict(zip(outputs, ort_outs))

    # np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(threshold=10)

    print('-----------------------------------------------------------')
    print('-----------------------------------------------------------')
    
    print('compare_model_output, begin run gpu......')
    
    EP_list = [target_ep]
    #EP_list = ['MACAExecutionProvider']
    
    ort_session_gpu = onnxruntime.InferenceSession(model.SerializeToString(), providers=EP_list)
    
    ort_outs_gpu = ort_session_gpu.run(outputs, ort_inputs)
    
    ort_outs_gpu = OrderedDict(zip(outputs, ort_outs_gpu))

    logger.info("compare_model_output, compare_model_output finish")

    return compare_result(ort_outs, ort_outs_gpu, [], model.graph.initializer, True)

def handle_test(model, ort_inputs, node_list, desc='debug'):
    EP_list = ['CPUExecutionProvider']
    ort_session = onnxruntime.InferenceSession(model.SerializeToString(), providers=EP_list)
    outputs = [x.name for x in ort_session.get_outputs()]
    print(desc+', output list:')
    print(outputs)

    print(desc + ', begin run cpu......')

    ort_outs = ort_session.run(outputs, ort_inputs)
    
    ort_outs = OrderedDict(zip(outputs, ort_outs))

    #np.set_printoptions(threshold=10)
    del ort_session

    print('-----------------------------------------------------------')
    print('-----------------------------------------------------------')
    
    print(desc+', begin run gpu......')

    EP_list = [target_ep]
    #EP_list = ['MACAExecutionProvider']
    
    ort_session_gpu = onnxruntime.InferenceSession(model.SerializeToString(), providers=EP_list)
    
    ort_outs_gpu = ort_session_gpu.run(outputs, ort_inputs)
    
    ort_outs_gpu = OrderedDict(zip(outputs, ort_outs_gpu))

    out_list_gpu=[]

    logger.info(desc+"finish")
    
    r = compare_result(ort_outs, ort_outs_gpu, node_list, model.graph.initializer)

    del ort_session_gpu

    if r == False:
        mismatch_list.extend(node_list)

def test_model(model, test_node_list, model_type):
    node_list=[]
    
    del model.graph.output[:]

    ort_inputs = generate_onnx_model_random_input(model)

    output_count = 0

    desc = 'test_common_model'
    if model_type == 'quantize':
        desc = 'test_quantize_model'

    if len(test_node_list) > 0:
        for node in model.graph.node:
            if model_type == 'quantize':
                if node.op_type != 'QuantizeLinear':
                    continue

            for output in node.output:
                if node.name in test_node_list:
                    dict_ = {"name":node.name, "input":node.input, "output":node.output, "op_type": node.op_type, "attribute":node.attribute}
                    node_list.append(dict_)
                    
                    model.graph.output.extend([onnx.ValueInfoProto(name=output)])
                    handle_test(model, ort_inputs, node_list, desc)
                    del model.graph.output[:]
                    del node_list[:]
    else:
        first_quantize_node = True
        for node in model.graph.node:
            for output in node.output:
                if model_type == 'quantize':
                    if node.op_type != 'QuantizeLinear':
                        continue
                    elif first_quantize_node == True:
                        first_quantize_node = False
                        continue

                dict_ = {"name":node.name, "input":node.input, "output":node.output, "op_type": node.op_type, "attribute":node.attribute}
                node_list.append(dict_)        

                output_count = output_count + 1
                model.graph.output.extend([onnx.ValueInfoProto(name=output)])

                if output_batch == 0 or output_count % output_batch == 0:
                    handle_test(model, ort_inputs, node_list, desc)
                    del model.graph.output[:]
                    del node_list[:]

    if output_batch != 0:
        if len(test_node_list) == 0 and output_count % output_batch != 0:
            handle_test(model, ort_inputs, node_list, desc)

def parse_common_mismatch_result(model, check_all):
    input_list = []
    output_list = []
    output_list2 = []

    all_node = []
    mismatch_node = []

    for node in model.graph.node:
            all_node.append(node.name)

    for node in mismatch_list:
        mismatch_node.append(node['name'])

    all_node = all_node[::-1]  
    mismatch_node = mismatch_node[::-1] 

    print('----------------------Debug info:-----------------------')
    print('All node:', all_node)  
    print('All mismatch_node:', mismatch_node)
    print('--------------------------------------------------------') 

    if check_all == False:
        return

    found = False
    for i, q in enumerate(all_node):
        for j, m in enumerate(mismatch_node):
            if m == q:
                found = True
                break

        if found == True:
            break        

    #print('Match i, j:', i, j)
    if found == False:
        return

    all_node = all_node[i:]
    mismatch_node_ = mismatch_node[j:]

    length = len(mismatch_node_)

    for i in range(length):
        if mismatch_node_[i] != all_node[i]:
            print('search end i=', i)
            break

    mismatch_node_ = mismatch_node_[:i]
    mismatch_node_ = mismatch_node_[::-1]

    mismatch_node_print = ''
    first = True

    for m in mismatch_node_:
        if first == True:
            mismatch_node_print = m
            first = False
        else: 
            mismatch_node_print = mismatch_node_print + '-->'   
            mismatch_node_print = mismatch_node_print + m

    print('parse_common_mismatch_result, mismatch path:', mismatch_node_print)        

def parse_quantize_mismatch_result(model, check_all):
    input_list = []
    output_list = []
    output_list2 = []

    quantize_node = []
    mismatch_node = []

    for node in model.graph.node:
        if node.op_type == 'QuantizeLinear':
            quantize_node.append(node.name)

    for node in mismatch_list:
        mismatch_node.append(node['name'])

    quantize_node = quantize_node[::-1]  
    mismatch_node = mismatch_node[::-1] 

    print('----------------------Debug info:-----------------------')
    print('All quantize node:', quantize_node)  
    print('All mismatch_node:', mismatch_node)
    print('--------------------------------------------------------') 

    if check_all == False:
        return

    found = False
    for i, q in enumerate(quantize_node):
        for j, m in enumerate(mismatch_node):
            if m == q:
                found = True
                break

        if found == True:
            break        

    #print('Match i, j:', i, j)
    if found == False:
        return

    quantize_node_ = quantize_node[i:]
    mismatch_node_ = mismatch_node[j:]

    length = len(mismatch_node_)

    for i in range(length):
        if mismatch_node_[i] != quantize_node_[i]:
            print('search end i=', i)
            break

    mismatch_node_ = mismatch_node_[:i]
    mismatch_node_ = mismatch_node_[::-1]

    mismatch_node_print = ''
    first = True

    for m in mismatch_node_:
        if first == True:
            mismatch_node_print = m
            first = False
        else: 
            mismatch_node_print = mismatch_node_print + '-->'   
            mismatch_node_print = mismatch_node_print + m

    print('mismatch path:', mismatch_node_print)      

def usage():
    print('maca_debug.py is used for compare inference result between cpu and gpu')
    print('it generates random input for given model, and compare output layer by layer')
    print('cmd lines:')
    print('---------------------------------------------------------------------')
    print('python maca_debug.py -i ./test.onnx -t common -b 3 -s 256')
    print('or') 
    print('python maca_debug.py -f ./test.onnx -t quantize -m Conv_1,Sigmoid_0')
    print('---------------------------------------------------------------------')
    print('-t for model type; -b for output batch(default is 5); -s for input batch size(default is 1)')
    print('if you want to specify compare node, you can use -m to list node name which you want to compare')
    print('it supports three methods to compare:snr(default), mse, cosine')
    print('their default threshold is 0.001(snr), 0.0(mse), 0.03(cosine)')
    print('if you want to specify compare method or threshold, you can use: -c mse -v 0.05')

def get_compare_node(str_list):
    ll = str_list.split(',')
    return ll

def main(argv):
    global onnxfile
    global precision_cmp_str
    global precision_threshold
    global output_batch
    global input_batch
    global input_npy

    test_node_list = []
    model_type = 'common'
    valid_model_type = [model_type, 'quantize']

    set_threshold = False

    try:
        opts, args = getopt.getopt(argv,"hi:v:c:m:t:b:d:s:",["onnx=", "compare_threshold=", "compare_method=", "multi_node=", "model_type=", "output_batch=", "input_data=", "input_batch"])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt in ("-i", "--onnx"):
            onnxfile = arg
        elif opt in ("-v", "--compare_threshold"):
            precision_threshold = float(arg)
            set_threshold = True     
        elif opt in ("-c", "--compare_method"):
            precision_cmp_str = arg    
        elif opt in ("-m", "--multi_node"):
            test_node_list = get_compare_node(arg)  
            print('test_node_list:', test_node_list)
        elif opt in ("-b", "--output_batch"):
            output_batch = int(arg) 
            if output_batch < 0:
                output_batch = 0
            print('output_batch:', output_batch)
        elif opt in ("-d", "--input_data"):
            input_npy = arg
            print('input_npy:', input_npy)    
        elif opt in ("-t", "--model_type"):
            model_type = arg
            if model_type not in valid_model_type:
                print('Warning: valid model type are:', valid_model_type)
                usage()
                sys.exit(-1)    
        elif opt in ("-s", "--input_batch"):
            input_batch = int(arg) 
            if input_batch < 0:
                input_batch = 1
            print('input_batch:', input_batch)

    print('onnx file path: ', onnxruntime.__file__)    
            
    if onnxfile == '':
        print('Warning: you should specify onnxfile(-i)') 
        usage()
        sys.exit(-1)
      
    print('model file: ', onnxfile)

    if precision_cmp_str not in ['snr', 'mse', 'cosine']:
        print('Warning: precision_cmp_str can only be one of [\'snr\', \'mse\', \'cosine\']')
        usage()
        sys.exit(-1)

    if set_threshold == False:
        if precision_cmp_str == 'mse':
            precision_threshold = 0.0
        elif precision_cmp_str == 'cosine':
            precision_threshold = 0.03     
            
    print('compare method: ', precision_cmp_str, precision_threshold)

    model = onnx.load(onnxfile)
    del model.graph.value_info[:]
    for input in model.graph.input:
        if len(input.type.tensor_type.shape.dim) > 1:
            input.type.tensor_type.shape.dim[0].dim_value = input_batch

    for output in model.graph.output:
        if len(output.type.tensor_type.shape.dim) > 1:
            output.type.tensor_type.shape.dim[0].dim_value = input_batch

    check_all = True
    if len(test_node_list) > 0:
        check_all = False

    r = compare_model_output(model)
    if r == False or check_all == False:
        test_model(model, test_node_list, model_type)
        if model_type == 'common':
            parse_common_mismatch_result(model, check_all)
        else:
            parse_quantize_mismatch_result(model, check_all)   

if __name__ == "__main__":
    main(sys.argv[1:])
