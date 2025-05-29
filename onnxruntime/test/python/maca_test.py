# coding=utf-8
"""
-------------------------------------------------------------------
   Copyright (c) 2021-2023 Metax Inc. All rights reserved.

   Description :
   create date :  2023/08/23 17:30
-------------------------------------------------------------------
"""
import argparse
import os
import threading
import time
from pathlib import Path

import numpy as np
import onnx

import onnxruntime as ort


def main_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', "-m", required=True, type=str, help='onnx models path, can run multiple models')
    parser.add_argument('--device_ids', "-d", nargs='*',type=int, help='device to run each model')
    parser.add_argument('--num_threads', "-t", default=1, type=int, help='number of threads to run for each model')
    parser.add_argument('--num_tests', "-n", default=1, type=int, help='number of infers per thread')
    parser.add_argument('--input_loc_str', "-i", default="cpu", type=str, help='input loc str: cpu、maca、maca_pinned')
    parser.add_argument('--output_loc_str', "-o", default="cpu", type=str, help='output loc str: cpu、maca、maca_pinned')
    parser.add_argument('--check_mode', "-c",default=0, type=int,  help='result check with CPU,0:non-check,1:check')
    arg = parser.parse_args()
    return arg

TypeMap = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(uint8)" : np.uint8,
    "tensor(int8)": np.int8,
    "tensor(int32)": np.int32,
    "tensor(int64)" : np.int64,
    "tensor(uint64)" : np.uint64,
}

def createInputData(input_nodes):
    input_dict = {}
    for i_n in input_nodes:
        shape = []
        for k in range(len(i_n.shape)):
            if i_n.shape[k] is None:
                shape.append(1)

            else:
                shape.append(i_n.shape[k])
        input_dict[i_n.name] = np.array(np.random.random(shape),
                                dtype=TypeMap[i_n.type])
    return input_dict

def gainNodeNames(nodes):
    names = []
    for n in nodes:
        names.append(n.name)
    return names


def getIobinding(sess, input_dict, input_loc_str, output_names, output_loc_str, device_id):
    io_binding = sess.io_binding()
    for key in input_dict.keys():
        io_binding.bind_ortvalue_input(key,
            ort.OrtValue.ortvalue_from_numpy(input_dict[key], input_loc_str, device_id=device_id))
    for o_n in output_names:
        io_binding.bind_output(o_n, output_loc_str,device_id=device_id)
    return io_binding

def createInputOrtValue(input_dict, input_loc_str, device_id):
    input_ort_dict = {}
    for key in input_dict.keys():
        input_ort_dict[key] = ort.OrtValue.ortvalue_from_numpy(input_dict[key], input_loc_str, device_id=0)
    return input_ort_dict

def compareResult(golden_output, maca_output):
    golden_data_array = np.reshape(golden_output, (-1,1))
    maca_data_array = np.reshape(maca_output, (-1,1))
    assert(golden_data_array.shape[0] == maca_data_array.shape[0])
    max_absolute_error = np.max(np.abs(golden_data_array - maca_data_array))
    snr = np.sum(np.square(np.abs(golden_data_array - maca_data_array))) / (np.sum(np.square(np.abs(golden_data_array))) + 1e-7)
    mse = np.sum(np.square(np.abs(golden_data_array - maca_data_array))) / maca_data_array.shape[0]
    return max_absolute_error,snr,mse

def sessionIobindingRun(sess, input_ort_dict, output_loc_str,num_tests, device_id):
    output_nodes = sess.get_outputs()
    output_names = gainNodeNames(output_nodes)
    for i in range(num_tests):
        io_binding = sess.io_binding()
        for key in input_ort_dict.keys():
            io_binding.bind_ortvalue_input(key, input_ort_dict[key])
        for o_n in output_names:
            io_binding.bind_output(o_n, output_loc_str,device_id=0)
        sess.run_with_iobinding(io_binding)

def computeFPS(session_maca_dict,input_orts_dict,output_loc_str, num_tests):
    threads = []
    total_tests = 0
    for d_id in session_maca_dict.keys():
        total_tests += (len(input_orts_dict[d_id]) * num_tests)
        for input_ort_dict in input_orts_dict[d_id]:
            threads.append(threading.Thread(target=sessionIobindingRun, args=(session_maca_dict[d_id], input_ort_dict, output_loc_str, num_tests, d_id)))
    start = time.time()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    time_cost = time.time() - start
    latency = time_cost/(total_tests)
    FPS = 1./latency
    return FPS

def computeAccuracy(session_maca_dict, session_cpu, input_datas_dict):
    output_names = gainNodeNames(session_cpu.get_outputs())
    for d_id in session_maca_dict.keys():
        for input_data_dict in input_datas_dict[d_id]:
            output_maca = session_maca_dict[d_id].run(output_names, input_data_dict)
            output_cpu = session_cpu.run(output_names, input_data_dict)
            assert(len(output_cpu) == len(output_maca))
            for i in range(len(output_maca)):
                max_absolute_error,snr,mse = compareResult(output_cpu[i], output_maca[i])
                print("Output %d, Name : %s, max_absolute_error : %.6f, snr : %.6f, mse : %.6f"%(i,output_names[i],max_absolute_error,snr,mse))

def getAvailableGpuIDs():
    device_ids = [i for i in range(ort.getDeviceCount())]
    return device_ids

def run():
    args = main_parse_args()
    device_ids = args.device_ids
    if(device_ids is None):
        device_ids = getAvailableGpuIDs()
    model_dir_obj = Path(args.model_dir)
    input_loc_str = args.input_loc_str
    output_loc_str = args.output_loc_str
    num_tests = args.num_tests
    num_threads = args.num_threads
    for m in model_dir_obj.rglob("*.onnx"):
        model_path = str(m)
        print("Init Model test : %s"%model_path)
        session_maca_dict = {}
        for d_id in device_ids:
            provider_options = [{"device_id":d_id}]
            session_maca = ort.InferenceSession(model_path, providers = ["MACAExecutionProvider"], provider_options = provider_options)
            session_maca_dict[d_id] = session_maca
            input_nodes = session_maca.get_inputs()

        # generate input data and init session
        input_datas_dict = {}
        input_orts_dict = {}
        for d_id in session_maca_dict.keys():
            input_datas = []
            input_orts = []
            for i in range(num_threads):
                input_dict = createInputData(input_nodes)
                input_datas.append(input_dict)
                input_orts.append(createInputOrtValue(input_dict,input_loc_str,d_id))
            input_datas_dict[d_id] = input_datas
            input_orts_dict[d_id] = input_orts

        if args.check_mode == 0:
            print("Begin to Model fps test : %s"%model_path)
            fps = computeFPS(session_maca_dict,input_orts_dict,output_loc_str,num_tests)
            print("FPS is %f"%fps)
            print("End to model fps test : %s"%model_path)
        else:
            print("Begin to Model accuracy test : %s"%model_path)
            session_cpu =  ort.InferenceSession(model_path, providers = ['CPUExecutionProvider'])
            accuracy = computeAccuracy(session_maca_dict,session_cpu,input_datas_dict)
            print("End to Model accuracy test : %s"%model_path)


if __name__ == '__main__':
    np.random.seed(3)
    run()
