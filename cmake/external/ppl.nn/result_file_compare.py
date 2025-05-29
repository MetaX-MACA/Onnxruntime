import numpy as np
import sys, getopt
import os

def numpy_load(file_name):
    '''
    try:
        value = np.load(file_name)
    except ValueError:
        value = np.load(file_name, allow_pickle=True)
    '''
    value = np.fromfile(file_name, np.float32)
    return value

def get_cosine(gpu_array, cpu_array):
    from math import sqrt
    gpu_array = gpu_array.astype('float64')
    cpu_array = cpu_array.astype('float64')
    x = np.square(gpu_array)
    #print(x)
    x = np.sum(x) 
    #print(x)
    x = sqrt(x)
    #print(x)

    y = np.square(cpu_array)
    #print(y)
    y = np.sum(y)
    #print(y)
    y = sqrt(y)
    #print(y)

    z = gpu_array * cpu_array
    #print(z)
    z = np.sum(z)
    #print(z)

    cosine = ((z)) / ((x * y) + 1e-12) # eps
    if cosine > 1.0:
        cosine = 1.0

    return cosine
#'''
def get_mse(diff_array):
    x = np.square(diff_array)
    mse = np.mean(x)
    return mse  

def get_snr(diff_array, cpu_array):
    x = np.square(diff_array)
    x = np.sum(x)

    y = np.square(cpu_array)
    y = np.sum(y) 

    snr = (x) / (y + 1e-7)

    snr = np.mean(snr)

    return snr

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('error !!!!!!!!!!!!!!!!!  len(sys.argv) < 3')
        exit(1)
    dir1 = sys.argv[1]
    dir2 = sys.argv[2]
    dir1_list = os.listdir(dir1)
    dir2_list = os.listdir(dir2)
    assert len(dir1_list) == len(dir2_list)
    for file in dir1_list:
        assert file in dir2_list
        file1 = f'{dir1}/{file}'
        file2 = f'{dir2}/{file}'
        file1_np = numpy_load(file1)
        file2_np = numpy_load(file2)
        print(f'----------------   {file} compare info:   ------------------')
        #print(f'get_cosine: {get_cosine(file1_np, file2_np)}')
        print("get_cosine: %3.10f" % (get_cosine(file1_np, file2_np)))
        diff = file1_np - file2_np
        #print(f'get_mse: {get_mse(diff)}')
        print("get_mse: %3.10f" % (get_mse(diff)))
        #print(f'get_snr: {get_snr(diff, file1_np)}')
        print("get_snr: %3.10f" % (get_snr(diff, file1_np)))
        print(f'diff max: {np.max(diff)} @gpu: {file1_np.reshape(-1)[np.argmax(diff.reshape(-1))]}')
