import os
import onnx
import copy
import numpy as np
import logging
import onnxruntime
import sys, getopt
import json
from onnx import helper
import logging
from collections import OrderedDict


logger = logging.getLogger("[ONNXOPTIMIZER]")

onnxfile = ''
nodefile = ''
outputfile = ''

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
    logger.info("convert_ort_type_2_np")
    
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


def generate_sub_model(onnx_model, onnxfile):
    graph = onnx_model.graph
    nodes = graph.node
    length=len(nodes)

    print('length=', length)

    onnx_model_list=[]

    #for n in range(length):
    #   print('--------------', n, '    ',nodes[n])

    #for vv in graph.value_info:
    #    print('vv: ', vv)

    #sys.exit()    
        
    if length < 2:
        sys.exit(0)

    for i in range(length-1):
        onnx_model_tmp = onnx.load(onnxfile)
        graph_tmp = onnx_model_tmp.graph
        #print(i,nodes[i])
        #graph_tmp.node[i].output[0]=graph.node[length-1].output[0]
        del onnx_model_tmp.graph.output[:]

        for output in graph_tmp.node[i].output:
                onnx_model_tmp.graph.output.extend([onnx.ValueInfoProto(name=output)])

        #for j in range(i+1, len):
        for j in range(length-1, i, -1):
            print('j=', j)
            graph_tmp.node.remove(graph_tmp.node[j])

        model_tmp = helper.make_model(graph_tmp, producer_name='onnx-example', opset_imports=[helper.make_opsetid('', 9)])
        #onnx.checker.check_model(model_tmp)
        file="./tmp."+str(i)+".onnx"
        onnx.save(model_tmp, file) 
        onnx_model_list.append(file)

    return onnx_model_list

def compare_result(ort_outs_cpu, ort_outs_gpu): 
    match=True 
    for k,v in ort_outs_cpu.items():
       #print(k, ':', v.shape)
       #print('v.type:', v.__class__)
       assert v.__class__  == np.ndarray
       #print('shape: ', v.shape)
       #print('ndim: ', v.ndim)
       #print('dtype: ', v.dtype)
       if v.__class__  == np.ndarray :
           c=v.flatten()
           print('cpu: ', c)
           assert k in ort_outs_gpu
           g=ort_outs_gpu[k].flatten()
           print('gpu: ', g)
           diff=np.subtract(c, g)
           index=0
           for i in diff:
              if i > 0.01 or i < -0.01:
                  match=False
                  print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                  print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                  print('WARNING: output ', k, ' is abnormal, please check it~~')
                  print('cpu val: ', c[index], 'gpu val: ', g[index])
                  print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                  print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')  
                  
                  break

              index = index + 1  
                  
       if match == False :
             break    

    print('match is ', match)   

    return match

def test_model_by_onnxruntime(model):
    logger.info("Test model by onnxruntime")

    print('input list:')
    inputs_data = {}
    for input_tensor in model.graph.input:
        input_shape = input_tensor.type.tensor_type.shape.dim
        type = input_tensor.type.tensor_type.elem_type
        print("name:{} type:{}".format(input_tensor.name,type))

        image_shape = [x.dim_value for x in input_shape]
        print(image_shape)
        image_shape_new = []
        for x in image_shape:
            if x == 0:
                image_shape_new.append(1)
            else:
                image_shape_new.append(x)
        
        image_shape = image_shape_new
        img_array = np.array(np.random.random(image_shape), dtype = convert_ort_type_2_np(type))
        inputs_data[input_tensor.name]= img_array

    ori_outputs = [x.name for x in model.graph.output]
    ori_outputs_backup=model.graph.output[:]
    #print('ori:', ori_outputs)

    EP_list = ['CPUExecutionProvider']
    ort_session = onnxruntime.InferenceSession(model.SerializeToString(), providers=EP_list)

    ort_inputs = {}

    for i, input_ele in enumerate(ort_session.get_inputs()):
        ort_inputs[input_ele.name] = inputs_data[input_ele.name]

    outputs = [x.name for x in ort_session.get_outputs()]

    print('output list:')
    print(outputs)

    print('begin run cpu......')

    ort_outs = ort_session.run(outputs, ort_inputs)
    
    #print(ort_outs)

    ort_outs = OrderedDict(zip(outputs, ort_outs))

    #np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(threshold=10)

    out_list=[]
    out_dict={}

    for k,v in ort_outs.items():
       #print(k, ':', v)
       #print('v.type:', v.__class__)
       if True == hasattr(v, 'tolist') :       
           dict={"output":k, "value":v.tolist()}
       else:
           dict={"output":k, "value":v} 
           
       out_list.append(dict)
       
    #print('list:', out_list)

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
    EP_list = ['MACAExecutionProvider','CPUExecutionProvider']
    
    ort_session_gpu = onnxruntime.InferenceSession(model.SerializeToString(), providers=EP_list)
    
    ort_outs_gpu = ort_session_gpu.run(outputs, ort_inputs)
    
    ort_outs_gpu = OrderedDict(zip(outputs, ort_outs_gpu))

    out_list_gpu=[]

    for k,v in ort_outs_gpu.items():
       #print(k, ':', v)
       #print('v.type:', v.__class__)
       if True == hasattr(v, 'tolist') :       
           dict={"output":k, "value":v.tolist()}
       else:
           dict={"output":k, "value":v} 
           
       out_list_gpu.append(dict)
    
    #print('list:', out_list)
    
    logger.info("Test model by onnxruntime finish")
    
    return compare_result(ort_outs, ort_outs_gpu)

     
def usage():
    print('python onnx_debug.py -i <onnxfile>')
    print('or') 
    print('python onnx_debug.py -i <onnxfile> -n <nodefile> -o <outputfile>', '({})'.format('Not yet support for now'))    

def main(argv):
   global onnxfile
   global nodefile
   global outputfile
   
   try:
      opts, args = getopt.getopt(argv,"hi:n:o:",["onnx=", "nfile=", "ofile="])
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
         
   if onnxfile == '' :
      usage()
      sys.exit()
         
   print('model file: ', onnxfile)
   print('node file: ', nodefile)
   print('output file: ', outputfile)
    
   onnx_model = onnx.load(onnxfile)

   match=test_model_by_onnxruntime(onnx_model)

   if match == False:
        print('compare failed, will find wihch layer is wrong......')

        onnx_sub_list = generate_sub_model(onnx_model, onnxfile)
        for i, val in enumerate(onnx_sub_list):
            print('compare: ', i, 'model: ', val)
            model = onnx.load(val)
            match=test_model_by_onnxruntime(model)
            if match == False:
                print('exit compare------------------------')
                break
   else:
       print('===================================================')
       print('===================================================')
       print('Congratulations, it works well as expected')


if __name__ == "__main__":
   main(sys.argv[1:])