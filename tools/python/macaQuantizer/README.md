# Maca-C500 Quantizer Tool

## 简介

目前只支持onnx模型，如果是其他模型，可以macaConverter转换成onnx模型。再使用 macaQuantizer 进行量化。

macaQuantizer量化过程中遍历多种量化设置，通过计算量化误差找出最优的量化模型，并导出对应的QDQ格式的onnx模型。

## 运行方式

**按照如下的参数说明配置量化yaml参数文件**

```yaml
import_model: /path/to/model.onnx
export_model: /path/to/quantized.onnx

quant_algorithm: percentile
force_advance_quant: False
export_dynamic: False
output_threshold: 0.1
without_bs: False
export_batch: -1


dataset:
  calib_dir: /dataset/classification_data/val
  calib_num: 200
  batch_size: 16
  
  preprocessing:
    enable: True
    attributes:
      isreverse: True
      mean: [123.67, 116.28, 103.53]
      std: [58.4, 57.12, 57.37]
      resize:
        keep_ratio: False
        to: [3,256,256]
        centercrop: [224,224]
```

参数说明：

- import_model：导入模型
- export_model：量化导出QDQ模型
- export_type：导出的量化模型类型， 可选[onnx, ppl, native]（默认为onnx）
- quant_algorithm: 量化采用的算法，可选[None, percentile, minmax, kl]  (默认为None）
- force_advance_quant：是否强制量化优化（默认False）
- export_batch：导出模型batchsize（默认为1）
- output_threshold：判断量化成功的输出层SNR阈值（默认0.1）
- without_bs: 模型输入是否不带有batchsize维度（默认False）
- collecting_device: 量化优化时缓存数据存在位置（默认cuda，显存不够时换成cpu）
- dispatcher:调度器 （可选 [pointwise, conservative, perseus, allin, pplnn])
- dataset：校准数据集参数
  - calib_dir：校准数据集目录或者dataset.txt 格式的文本
  - calib_num：校准数据个数
  - batch_size：量化计算时的batch_size
  - preprocessing：预处理
    - enable：是否进行预处理，如果为False，calib_dir里的数据需要为.npy 格式的数据，或者 .bin/.raw 的二进制数据
    - attributes：预处理属性参数
      - isreverse：是否做RGB转BRG
      - mean：预处理的均值
      - std：预处理的方差
      - resize:预处理resize参数
        - keep_ratio：是否等尺度缩放
        - to: resize大小 [chw]格式
        - pad_value: 预处理过程中的pad的值（默认为0）
        - centercrop：centercrop 大小，如果不做centercrop可以删除该参数



**运行命令：** 

```shell
python -m  maca_quantizer -c  quantize.yaml | tee maca_quantize.log
```

**参数说明：** -c / --config : yaml 参数文件

***注意**：多输入模型支持dataset.txt格式输入



**dataset.txt 格式**

**单个输入** 只读取第一列的文件路径（空格分割）

```
val2017/000000000139.jpg 139 640 426
val2017/000000000285.jpg 285 586 640
val2017/000000000632.jpg 632 640 483
val2017/000000000724.jpg 724 375 500
val2017/000000000776.jpg 776 428 640
val2017/000000000785.jpg 785 640 425
val2017/000000000802.jpg 802 424 640
val2017/000000000872.jpg 872 621 640
val2017/000000000885.jpg 885 640 427
val2017/000000001000.jpg 1000 640 480
val2017/000000001268.jpg 1268 640 427
val2017/000000001296.jpg 1296 427 640
val2017/000000001353.jpg 1353 375 500
val2017/000000001425.jpg 1425 640 512
val2017/000000001490.jpg 1490 640 315
```

**多个输入** 一行为一个数据样本（空格分割）

```
1000000000_input_ids.npy 1000000000_input_mask.npy 1000000000_segment_ids.npy
1000000001_input_ids.npy 1000000001_input_mask.npy 1000000001_segment_ids.npy
1000000002_input_ids.npy 1000000002_input_mask.npy 1000000002_segment_ids.npy
1000000003_input_ids.npy 1000000003_input_mask.npy 1000000003_segment_ids.npy
1000000004_input_ids.npy 1000000004_input_mask.npy 1000000004_segment_ids.npy
1000000005_input_ids.npy 1000000005_input_mask.npy 1000000005_segment_ids.npy
1000000006_input_ids.npy 1000000006_input_mask.npy 1000000006_segment_ids.npy
1000000007_input_ids.npy 1000000007_input_mask.npy 1000000007_segment_ids.npy
```



## 环境配置：

whl包编译：

```shell
./build_wheel.sh
```

运行结束后在dist目录下生成对应wheel包。



安装wheel包：

```shell
pip install  maca_quantizer-xxxx.whl
```

如果requirments里面的依赖已经安装，可以在安装wheel包 加 --no-deps 忽略依赖包的安装。



## Environment Variables

**Table 1. Environment Variables for C500 MacaQuantizer**

| **Environment Variable MacaQuantizer** | **Value**               | **Default**           | **Description**                                      |
| :------------------------------------- | :---------------------- | --------------------- | ---------------------------------------------------- |
| **MACA_QUANTIZER_USING_MXGPU**         | [0, 1]                  | 1                     | 0：Disable<br />1：Enable metax gpu to accelerate calculation   |
| **MX_ENABLE_OPERATOR_FUSION**          | -closeAll               |                       | Close all of operator fusion                        |
|                                        | -swish:0/1              | -swish:1              | 0：Disable<br />1：Enable swish operator fusion      |
|                                        | -mish:0/1               | -mish:1               | 0：Disable<br />1：Enable mish operator fusion       |
|                                        | -hardswish:0/1          | -hardswish:1          | 0：Disable<br />1：Enable hardswish operator fusion  |
|                                        | -hardsigmoid:0/1        | -hardsigmoid:1        | 0：Disable<br />1：Enable hardsigmoidoperator fusion |
|                                        | -reducel2:0/1           | -reducel2:1           | 0：Disable<br />1：Enable reducel2 operator fusion   |
|                                        | -gelu:0/1               | -gelu:1               | 0：Disable<br />1：Enable gelu operator fusion       |
|                                        | -groupnormalization:0/1 | -groupnormalization:1 | 0：Disable<br />1：Enable groupnormalization operator fusion   |
|                                        | -multiheadattentionv1:0/1 | -multiheadattentionv1:1 | 0：Disable<br />1：Enable MultiHeadAttentionv1 operator fusion   |


**Example:**

关闭上述所有图优化：
```shell
export MX_ENABLE_OPERATOR_FUSION=-closeAll
```

关闭Gelu图融合：
```shell
export MX_ENABLE_OPERATOR_FUSION=-gelu:0
```

关闭Swish和Mish图融合：
```shell
export MX_ENABLE_OPERATOR_FUSION=-swish:0-mish:0
```

