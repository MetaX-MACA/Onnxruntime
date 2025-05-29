# /usr/bin/env python
# coding=utf-8

import os
import numpy as np
import time
import argparse
import sys

# 统计百分位数
Percentile = [0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.00007, 0.00009, 0.0001, 0.005, 0.009, 0.1, 0.2, 0.3, 0.5,
              1, 3, 5, 10, 15, 20, 50, 70, 100]

COMPARE_DIR = "./comparedir"

WRONG_ADDRESS_DISPLAY_QUANTITY = 10

write = False


def stratToCompare(targetFilePath, referenceFilePath, data_type, reportName="report.txt"):
    targetSize = targetFilePath.size
    referenceSize = referenceFilePath.size
    if targetSize != referenceSize:
        print("文件大小不匹配")
        return

    dtype_dict = {
        'int8': [np.int8, 7, 8],
        'uint8': [np.uint8, 8, 9],
        'int16': [np.int16, 15, 16],
        'uint16': [np.uint16, 16, 16],
        'float16': [np.float16, 15, 16],
        'int32': [np.int32, 31, 32],
        'uint32': [np.uint32, 32, 32],
        'float32': [np.float32, 31, 32],
        'int64': [np.int64, 63, 64],
        'uint64': [np.uint64, 64, 64],
        'float64': [np.float64, 63, 64],
    }

    if data_type in dtype_dict.keys():
        dtype = dtype_dict[data_type][0]
        bits = dtype_dict[data_type][1]
        true_bits_num = dtype_dict[data_type][2]
        target_data_raw = targetFilePath
        reference_data_raw = referenceFilePath
        # 转成浮点数计算，防止丢失精度
        target_data = target_data_raw.astype(np.float64)
        reference_data = reference_data_raw.astype(np.float64)
        max_data_bits = 2 ** bits - 1
    else:
        if data_type.startswith('u'):
            data_type = data_type[1:]
        try:
            # 整数位数
            int_bit = int(data_type.split('P')[-2])
        except:
            print("数据类型不匹配")
            return
        try:
            # 小数点位
            decimal_bit = int(data_type.split('P')[-1])
        except:
            print("数据类型不匹配")
            return
        # 有效位数
        num_bit = int_bit + decimal_bit

        if data_type.startswith('u'):
            if num_bit <= 8:
                dtype = np.uint8
                bits = 8
                true_bits_num = 8
            elif num_bit <= 16:
                dtype = np.uint16
                bits = 16
                true_bits_num = 16
            elif num_bit <= 32:
                dtype = np.uint32
                bits = 32
                true_bits_num = 32
            elif num_bit <= 64:
                dtype = np.uint64
                bits = 64
                true_bits_num = 64

            max_data_bits = 2 ** int_bit - 0.5 ** decimal_bit

            target_data_int = targetFilePath
            target_data_float = target_data_int.astype(np.float64)
            target_data = target_data_float / (2 ** decimal_bit)

            reference_data_int = referenceFilePath
            reference_data_float = reference_data_int.astype(np.float64)
            reference_data = reference_data_float / (2 ** decimal_bit)

        else:
            # 有符号定点数,先按照无符号数进行左移把定点数的符号位挤到最到位，按照有符号整数读取
            # 再转成64位浮点数进行右移
            if num_bit <= 8:
                u_dtype = np.uint8
                dtype = np.int8
                bits = 8
                true_bits_num = 8
            elif num_bit <= 16:
                u_dtype = np.uint16
                dtype = np.uint16
                bits = 16
                true_bits_num = 16
            elif num_bit <= 32:
                u_dtype = np.uint32
                dtype = np.int32
                bits = 32
                true_bits_num = 32
            elif num_bit <= 64:
                u_dtype = np.uint64
                dtype = np.int64
                bits = 64
                true_bits_num = 64

            # 数据类型最大值
            max_data_bits = 2 ** (int_bit - 1) - 0.5 ** decimal_bit

            target_data = targetFilePath
            reference_data = referenceFilePath

            # 左移把0挤掉
            target_data = target_data * (2 ** (bits - num_bit))
            reference_data = reference_data * (2 ** (bits - num_bit))
            # 解码为有符号整数
            target_data.dtype = dtype
            reference_data.dtype = dtype
            # 转为小数计算,防止丢失精度
            target_data = target_data.astype(np.float64)
            reference_data = reference_data.astype(np.float64)
            # 再移回来
            target_data = target_data / (2 ** (decimal_bit + bits - num_bit))
            reference_data = reference_data / (2 ** (decimal_bit + bits - num_bit))

    '''文件加载完毕，开始统计误差'''
    # 数据数量
    data_num = target_data.shape[0]
    # 数据最大值
    if np.max(target_data) > np.max(reference_data):
        max_data = np.max(target_data)
    else:
        max_data = np.max(reference_data)
    # 差值数列
    if np.min(target_data) < np.min(reference_data):
        min_data = np.min(target_data)
    else:
        min_data = np.min(reference_data)
    interval = max_data - min_data

    dif = np.abs(target_data - reference_data)

    # 统计差值的大小,浮点数类型和统计最大值比较，其他类型和数据类型最大值比较
    if data_type.startswith("f"):
        dif_rate_array = dif / (interval / 100)

    else:
        dif_rate_array = dif / (max_data_bits / 100)
    diference_array = np.where(dif == 0, 0.0, 1.0)
    max_index = np.argmax(dif_rate_array)
    dif_num = np.sum(diference_array)
    dif_proportion = dif_num / data_num

    # key是字符串，value是[小于待比区间的数量和比例]
    dif_per_dict = {}
    for percent_num in Percentile:
        dif_num_sub = np.sum(np.where(dif_rate_array < percent_num, 1.0, 0.0)) - (data_num - dif_num)
        dif_per_sub = dif_num_sub / data_num
        dif_per_dict[percent_num] = [dif_num_sub, dif_per_sub]

    greater_1_num = np.sum(np.where(dif_rate_array >= 100, 1.0, 0.0))
    greater_1_per = greater_1_num / data_num
    dif_per_dict[">100"] = [greater_1_num, greater_1_per]

    # 统计错误比例
    print("文件名：%s\n文件总大小：%d bytes\n文件类型：%s\n数据总量：%d\n误差数据总量：%d, 总占比：%.5f%%\n误差分布："
          % ("cpu vs gpu", targetSize, data_type, data_num, dif_num, dif_proportion * 100))
    print('{0:^15}\t{1:^24}\t{2:^8}'.format("差异大小(sub/max)", "误差数", "比例", " "))
    if write:
        file_data = open(reportName, 'a+')
        file_data.write("文件名：%s\n文件总大小：%d bytes\n文件类型：%s\n数据总量：%d\n误差数据总量：%d, 总占比：%.5f%%\n误差分布：\n"
                        % (
                           "cpu vs gpu", targetSize, data_type, data_num, dif_num,
                            dif_proportion * 100))
        file_data.write('{0:^15}\t{1:^24}\t{2:^8}\n'.format("差异大小(sub/max)", "误差数", "比例", " "))
    for i in range(len(Percentile)):
        if i != 0:
            # space_str = str(Percentile[i-1]) + "--" + str(Percentile[i]) + "%"
            space_str = "{0:^5} -- {1:^5} %".format(Percentile[i - 1], Percentile[i])
            # 区间数量
            space_num = dif_per_dict[Percentile[i]][0] - dif_per_dict[Percentile[i - 1]][0]
            # 区间占比
            space_per = (dif_per_dict[Percentile[i]][1] - dif_per_dict[Percentile[i - 1]][1]) * 100
        else:
            space_str = "<" + str(Percentile[i]) + " %"
            space_num = dif_per_dict[Percentile[i]][0]
            space_per = dif_per_dict[Percentile[i]][1] * 100
        if write:
            file_data.write('{0:^20}\t{1:^20}\t{2:^8.5f}%\n'.format(space_str, int(space_num), space_per))
        print('{0:^20}\t{1:^20}\t{2:^8.5f}%'.format(space_str, int(space_num), space_per))
    if write:
        file_data.write('{0:^20}\t{1:^20}\t{2:^8.5f}%\n'.format(">100 %", int(dif_per_dict[">100"][0]),
                                                                dif_per_dict[">100"][1] * 100))
        file_data.write('{0:^20}\t{1:^20}\t{2:^8.5f}%\n'.format("total", int(dif_num), dif_proportion * 100))
        file_data.write("\n差异最大值索引：\n")
    print('{0:^20}\t{1:^20}\t{2:^8.5f}%'.format(">100 %", int(dif_per_dict[">100"][0]), dif_per_dict[">100"][1] * 100))
    print('{0:^20}\t{1:^20}\t{2:^8.5f}%\n'.format("total", int(dif_num), dif_proportion * 100))

    print("差异最大值索引：")
    if dif_num != 0:
        if dif_num <= WRONG_ADDRESS_DISPLAY_QUANTITY:
            for h in range(int(dif_num)):
                max_index = np.argmax(dif_rate_array)
                if write:
                    file_data.write("%d max index: %s\n" % (h, max_index ))
                print("{0:^10} max index: {1:10d}".format(h, max_index), target_data[max_index],"vs", reference_data[max_index])
                dif_rate_array[max_index] = 0
                if np.sum(dif_rate_array) == 0:
                    break
        else:
            display_quantity = WRONG_ADDRESS_DISPLAY_QUANTITY
            for m in range(display_quantity):
                max_index = np.argmax(dif_rate_array)
                if write:
                    file_data.write("%d max index: %s\n" % (m, max_index ))
                print("{0:^10} max index: {1:10d}".format(m, max_index), target_data[max_index], "vs", reference_data[max_index])
                dif_rate_array[max_index] = 0
                if np.sum(dif_rate_array) == 0:
                    break
    else:
        print("")
    if write:
        file_data.close()
    
    print("=============================== split ===============================")

