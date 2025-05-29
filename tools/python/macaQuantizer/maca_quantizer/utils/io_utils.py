# -*- coding:utf-8 -*- #
"""
-------------------------------------------------------------------
   Copyright (c) 2019-2022 MetaX Inc. All rights reserved.

   Description :
   File Name：    io_utils.py
   create date：  2022/7/21
-------------------------------------------------------------------
"""
import os
import gc
import torch
import numpy as np
from torch.cuda import empty_cache
from typing import Any, Iterable, List, Union, Callable
from maca_quantizer.ppq_.ppq import DataType
from .utils import maca_error, maca_warning, maca_info, get_all_file, read_txt


def empty_maca_cache(func: Callable):
    """Using empty_maca_cache decorator to clear  memory cache, both gpu
    memory and cpu memory will be clear via this function.

    Function which get decorated by this will clear all ppq system cache BEFORE its running.
    Args:
        func (Callable): decorated function
    """
    def _wrapper(*args, **kwargs):
        empty_cache() # torch.cuda.empty_cache might requires a sync of all cuda device.
        gc.collect()  # empty memory.
        return func(*args, **kwargs)
    return _wrapper

def convert_any_to_torch_tensor(
    x: Union[torch.Tensor, np.ndarray, int, float, list, tuple],
    accepet_none: bool=True, dtype: torch.dtype=None, device='cpu') -> torch.Tensor:
    if x is None and accepet_none: return None
    if x is None and not accepet_none: raise ValueError('Trying to convert an empty value.')
    if isinstance(x, list) or isinstance(x, tuple):
        if all([type(element) == int for element in x]):
            if dtype is None: dtype=torch.int64
        return torch.tensor(x, dtype=dtype, device=device)
    elif isinstance(x, int):
        if dtype is None: dtype=torch.int64
        return torch.tensor(x, dtype=dtype, device=device)
    elif isinstance(x, float):
        if dtype is None: dtype=torch.float32
        return torch.tensor(x, dtype=dtype, device=device)
    elif isinstance(x, torch.Tensor):
        if dtype is not None: x = x.type(dtype)
        if device is not None: x = x.to(device)
        return x
    elif isinstance(x, np.ndarray):
        if dtype is None:
            dtype = DataType.convert_from_numpy(x.dtype)
            dtype = DataType.to_torch(dtype)
        return torch.tensor(x, dtype=dtype, device=device)
    else:
        raise TypeError(f'input value {x}({type(x)}) can not be converted as torch tensor.')


def load_dataset(
    directory: str, input_shape: List[int],batchsize: int, input_format: str = 'chw', 
    count = None, device='cuda', without_bs=False) -> Iterable:
    """
    Args:
        directory (str): 加载数据集的目录，目录不应包含子文件夹，所有目录中的文件将被视为数据。
        input_shape (List[int]): 图像尺寸，对于二进制输入文件而言，你必须指定图像尺寸，对于 npy文件 此项不起作用
        batchsize (int): batchsize 大小，这个函数会自动进行打包操作，但是如果你的数据本身已经有了预设的batchsize，
            则该函数不会覆盖原有batchsize
        input_format (str, optional): chw 或 hwc，指定输入图像数据排布。即使你的图像具有batch维度，它仍然将正常工作。

    Raises:
        FileNotFoundError: _description_
        ValueError: _description_

    Returns:
        Iterable: _description_
    """

    if not os.path.exists(directory):
        raise FileNotFoundError(f'无法从指定位置加载数据集 {directory}. '
                                 '目录或文件不存在，检查你的输入路径')
    if input_format not in {'chw', 'hwc'}:
        raise ValueError(f'无法理解的数据格式，对于图片数据，数据格式只能是 chw 或 hwc，而你输入了 {input_format}')
    
    file_list = []
    if os.path.isfile(directory) or directory.endswith(".txt"):
        # add read txt file for multiply input
        file_list = read_txt(directory, cnt=count, input_num=len(input_shape))
    elif os.path.isdir(directory):
        file_list = get_all_file(directory, cnt=count)
    
    if len(file_list) < 8:
        maca_warning(f'Calibrate dataset number {len(file_list)} too smail, Require more than 8 times batchsize')

    num_of_file, samples, sizes = 0, [], set()


    if len(input_shape)==1:
        input_shape_ = input_shape[0] if without_bs else input_shape[0][1:]
        for file in file_list:
            sample = None
            if file.endswith('.npy'):
                sample = np.load(file)
                num_of_file += 1
            elif file.endswith('.bin') or file.endswith('.raw'):
                sample = np.fromfile(file, dtype=np.float32)
                assert isinstance(sample, np.ndarray), f'数据应当是 numpy.ndarray，然而你输入了 {type(sample)}'
                sample = sample.reshape(input_shape_)
                num_of_file += 1
            else:
                maca_warning(f'文件格式不可读: {file}, 该文件已经被忽略.')

            sample = convert_any_to_torch_tensor(sample)
            sample = sample.float()

            if sample.ndim == 3 and not without_bs: sample = sample.unsqueeze(0)
            if input_format == 'hwc': sample = sample.permute([0, 3, 1, 2])

            # assert sample.shape[1] == 1 or sample.shape[1] == 3 or sample.shape[1] == 4, (
            #     f'你的文件 {file} 拥有 {sample.shape[1]} 个输入通道. '
            #     '这是合理的图像文件吗?(是否忘记了输入图像应当是 NCHW 的)')

            if sample.shape[0] != 1 and  batchsize != 1: 
                samples.append(sample.to(device))
                continue

            if len(sample.shape)>=2:
                sizes.add((sample.shape[-2], sample.shape[-1]))
            else:
                sizes.add((sample.shape[-1]))
            if input_shape[0][1:] == list(sample.shape): 
                sample = sample.unsqueeze(0)
            samples.append(sample.to(device))

        if len(sizes) != 1:
            maca_warning('你的输入图像似乎包含动态的尺寸，因此 CALIBRATION BATCHSIZE 被强制设置为 1')
            batchsize = 1

    elif len(input_shape) > 1:
        input_shape_ = input_shape if without_bs else [[1]+x[1:] for x in input_shape]
        for files in file_list:
            sample_list = []
            for idx, file in enumerate(files):
                sample = None
                if file.endswith('.npy'):
                    sample = np.load(file)
                    num_of_file += 1
                elif file.endswith('.bin') or file.endswith('.raw'):
                    sample = np.fromfile(file, dtype=np.float32)
                    assert isinstance(sample, np.ndarray), f'数据应当是 numpy.ndarray，然而你输入了 {type(sample)}'
                    sample = sample.reshape(input_shape_[idx])
                    num_of_file += 1
                else:
                    maca_warning(f'文件格式不可读: {file}, 该文件已经被忽略.')

                sample = convert_any_to_torch_tensor(sample)
                sample = sample.float()
                if len(sample.shape)==0: 
                    sample=sample
                elif sample.shape[0] != 1 and sample.shape[0] != batchsize and not without_bs:
                    sample = torch.unsqueeze(sample, dim=0)

                # assert sample.shape[0] == 1 or batchsize == 1 or without_bs, (
                #     f'你的输入图像似乎已经有了预设好的 batchsize, 因此我们不会再尝试对你的输入进行打包。')
                sample_list.append(sample.to(device))

            samples.append(sample_list)


    # create batches
    batches, batch = [], []
    if batchsize != 1 and not without_bs and len(input_shape)==1:
        for sample in samples:
            if len(batch) < batchsize:
                batch.append(sample)
            else:
                batches.append(torch.cat(batch, dim=0))
                batch = [sample]
        if len(batch) != 0:
            batches.append(torch.cat(batch, dim=0))
    else:
        batches = samples

    print(f'{num_of_file} File(s) Loaded.')
    for idx, tensor in enumerate(samples[: 5]):
        if isinstance(tensor, list):
            print(f'Loaded sample {idx}, shape: {[t.shape for t in tensor]}')
        else:
            print(f'Loaded sample {idx}, shape: {tensor.shape}')
            
    assert len(batches) > 0, '你送入了空的数据集'

    if isinstance(batches[0], list):
        print(f'Batch Shape: {[x.shape for x in  batches[0]]}')
    else:
        print(f'Batch Shape: {batches[0].shape}')

    return batches




def gen_random_dataloader(input_shape, without_bs, cnt=8, device='cpu'):
    data_tensor = []

    if len(input_shape)==1:
        for i in range(cnt):
            shape = input_shape[0]
            shape_ =  [1 if shape[i] in {0,-1} else shape[i] for i in range(len(shape))]
            sample = torch.rand(shape_,device=device)
            data_tensor.append(sample)
        # data_tensor = DataLoader(dataset=data_tensor, shuffle=False) 
    else:
        for i in range(cnt):
            sample_list = []
            for shape in input_shape:
                shape_ = [1 if shape[i] in {0,-1} else shape[i] for i in range(len(shape))]
                sample = torch.rand(shape_,device=device)
                sample_list.append(sample)
            data_tensor.append(sample_list)

    return data_tensor