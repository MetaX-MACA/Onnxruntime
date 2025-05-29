# -*- coding:utf-8 -*- #
"""
-------------------------------------------------------------------
   Copyright (c) 2019-2022 MetaX Inc. All rights reserved.

   Description :
   File Name：     utils.py
   create date：  2022/7/01
-------------------------------------------------------------------
"""
import os
import sys

import cv2
import onnx
import torch
import random
import os.path as osp
import numpy as np
import importlib.util
from maca_quantizer.ppq_.ppq.core import DataType
from typing import Any


def letterbox_image(image, new_shape, color=(0, 0, 0)):
    """resize image with unchanged aspect ratio using padding"""
    iw, ih = image.shape[1], image.shape[0]
    w, h = new_shape[0], new_shape[1]

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    new_unpad = (nw, nh)
    img = cv2.resize(image, new_unpad, interpolation=cv2.INTER_CUBIC)

    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    new_image = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_image


def _find_node_input_op(graph, op):
    input_op_list = []
    nodes = graph.node
    for inp in op.input:
        for node in nodes:
            if inp == node.name:
                input_op_list.append(node)

    return input_op_list

def preprocessImage(fname, img_size, img_mean, img_std,  data_format='CHW', keep_ratio=False, isreverse=False):         
    image = cv2.imread(fname, flags=1)
    if keep_ratio:
        image = letterbox_image(image, tuple(img_size))
    else:
        image = cv2.resize(image, tuple(img_size), interpolation=cv2.INTER_CUBIC)
    if not isreverse:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = image[..., ::-1]

    image_norm = np.subtract(image, img_mean)
    image_norm = np.divide(image_norm, img_std, dtype=np.float32)
    # image_norm = np.multiply(image_norm, img_scale, dtype=np.float32)
    if data_format.upper() == 'CHW':
        image_norm = np.transpose(image_norm, (2, 0, 1))  # hwc-->chw
    return image_norm


def numpy_load(file_name):
    try:
        value = np.load(file_name)
    except ValueError:
        value = np.load(file_name, allow_pickle=True)
    return value

def get_all_file(path, cnt=None, shuffle=False):
    file_list = []
    for root, dir, files in os.walk(path, topdown=False):
        for f in files:
            file_list.append(os.path.join(root, f))
    if shuffle:
        random.shuffle(file_list)
    if cnt is not None and len(file_list)>cnt:
        file_list = file_list[:cnt]
    return file_list


def read_txt(path, cnt=None, input_num=1, shuffle=False):
    file_list = []
    with open(path, 'r') as f:
        if input_num==1:
            for line in f.readlines():
                file_path = os.path.join(os.path.dirname(path), line.strip('\n').split()[0])
                file_list.append(file_path)
        else:
            for line in f.readlines():
                file_path_list = [os.path.join(os.path.dirname(path),x) for x in line.strip('\n').split()[:input_num]]
                file_list.append(file_path_list)
    if shuffle:
        random.shuffle(file_list)
    if cnt is not None and len(file_list)>cnt:
        file_list = file_list[:cnt]
    return file_list


def TransportFile(localpath, remotepath, params):
    import paramiko
    host = params["host"]
    port = params["port"]
    username, password=params["username"] ,params["password"]
    try:
        transport = paramiko.Transport((host, port))
        transport.connect(None, username=username, password=password)
        sftp = paramiko.SFTPClient.from_transport(transport)
        sftp.put(localpath, remotepath)  #上传本地文件到服务器
        transport.close()
        print("File transfer succeed")

    except Exception as e:
        print(e)


def check_mudule(name):
    if name in sys.modules:
        print(f"{name!r} already in sys.modules")
    elif (spec := importlib.util.find_spec(name)) is not None:
        # If you chose to perform the actual import ...
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        print(f"{name!r} has been imported")
    else:
        print(f"can't find the {name!r} module")


def register_module_from_file(module_name, file_path):
    try:
        assert sys.version_info.major==3 and sys.version_info.minor>5
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception as e:
        maca_error(e)




def maca_info(info: str):
    print(f'\033[32m[Info] {info}\033[0m')

def maca_warning(info: str):
    print(f'\033[33m[Warning] {info}\033[0m')

def maca_error(info: str):
    print(f'\033[31m[Error] {info}\033[0m')
    sys.exit(-1)

def get_input_info(model_path):
    onnx_model = onnx.load(model_path)
    initializer = []
    for init in onnx_model.graph.initializer:
        # maca_info('got init: ', init.name)
        initializer.append(init.name)

    input_nums = len(onnx_model.graph.input)
    input_shapes = []
    input_types = []
    for input_proto in onnx_model.graph.input:
        if input_proto.name not in initializer :
            shpape_list = []
            dim_proto_input = input_proto.type.tensor_type.shape.dim
            onnx_type = input_proto.type.tensor_type.elem_type
            for dim in dim_proto_input:
                shpape_list.append(dim.dim_value)
            input_shapes.append(shpape_list)
            input_types.append(DataType.to_torch(DataType(onnx_type)))
    return input_shapes, input_types


def check_path(path, check_dir=False):
    if check_dir:
        if osp.isdir(path):
            if not osp.exists(path):
                maca_error(f" Directory \"{path}\" is not exist !!!")
                return False
        else:
            if not osp.exists(osp.dirname(path)):
                maca_error(f" Directory \"{osp.dirname(path)}\" is not exist !!!")
                return False
    else:
        if not osp.exists(path):
            maca_error(f" Path \"{path}\" is not exist !!!")
            return False
    return True

def get_config(content: dict, key: str, default: Any=None, compulsive: bool=False):
    if content is not None and key in content:
        return content[key]
    else: 
        if compulsive:
            raise KeyError(f"Config missing \'{key}\' value from current yaml file")
        else:
            return default


def convert_tensor_to_float16(array, min_positive_val=6.10e-5, max_finite_val=65504):
    def between(a, b, c):
        return torch.logical_and(a < b, b < c)
    array = torch.where(between(0, array, min_positive_val), min_positive_val, array)
    array = torch.where(between(-min_positive_val, array, 0), -min_positive_val, array)
    array = torch.where(between(max_finite_val, array, float('inf')), max_finite_val, array)
    array = torch.where(between(float('-inf'), array, -max_finite_val), -max_finite_val, array)
    return array.type(torch.float16)

if __name__ == "__main__":
    a = convert_tensor_to_float16(torch.tensor(9.99999993922529e-9))
    b = convert_tensor_to_float16(torch.tensor(1998.32))
    c = convert_tensor_to_float16(torch.tensor(0.0))
    print(a, b, c)