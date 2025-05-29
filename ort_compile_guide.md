# 沐曦Onnxruntime编译步骤

## 1、下载onnxruntime子模块
```
git clone https://github.com/microsoft/onnxruntime.git -b v1.12.0
git submodule update --init --recursive
```
拷贝cmake/external/中所有文件夹到沐曦提供的onnxruntime同级目录下面

## 2、拉取pybind11
```
git clone https://github.com/pybind/pybind11.git -b v2.6.2
```
拷贝pybind11到沐曦提供的onnxruntime/cmake/external目录下面

## 3、下载abseil-cpp
```
https://github.com/abseil/abseil-cpp/archive/refs/tags/20211102.0.zip
```
压缩包解压，并将文件夹重命名为abseil-cpp, 将abseil-cpp拷贝到沐曦提供的onnxruntime/cmake/external目录下面

## 4、拉取hpcc
```
git clone https://github.com/openppl-public/hpcc.git
```
拷贝hpcc文件夹到onnxruntime/cmake/external/ppl.nn/deps

## 5、拉取rapidjson
```
git clone https://github.com/Tencent/rapidjson.git
git checkout 06d58b9e848c650114556a23294d0b6440078c61
```
拷贝文件夹rapidjson到onnxruntime/cmake/external/ppl.nn/deps

## 6、编译指令
```
export CUDA_PATH=cuda路径
export MACA_PATH=maca路径
export PATH=${CUDA_PATH}/bin:${MACA_PATH}/tools/cu-bridge/tools:${MACA_PATH}/mxgpu_llvm/bin:${MACA_PATH}/bin:${PATH}

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA_PATH}/lib64:${MACA_PATH}/lib
export CUCC_PATH=${MACA_PATH}/tools/cu-bridge

./build.sh --config Release --build_shared_lib --parallel=24 --skip_tests --use_ppl --enable_pybind --build_wheel --use_maca --skip_submodule_sync --use_dnn

```
