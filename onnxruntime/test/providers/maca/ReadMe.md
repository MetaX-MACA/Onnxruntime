# Sample
This project will guide you how to build and use sample
## 1. Build
Installed maca_driver & onnxruntime_maca
```
mkdir build
cd build
cmake ..
make
```
## 2. Usage
### maca_test
```
./maca_test --model_dir xxx
```
|param|detail|
|:---:|:---:|
|model_dir|dir path of onnx models|
|num_inputs|how many different input per thread. defaults to 1|
|num_tests|how many inference per input. defaults to 100|
|num_threads|how many threads to run each model. defaults to 32|
|input_data_loc| maca ep location of input data. 0-cpu, 1-maca_pinned, 2-maca, defaults to 2|
|output_data_loc|maca ep location of output data. 0-cpu, 1-maca_pinned, 2-maca, defaults to 2|
|log_level| set macart log level. defaults to 4|
|snr_threshold| snr_threshold to print log. defaults to 0.1|
|check_mode|0 fps mode, 1 accuracy mode . defaults to 0|
|save_io|save input and output|
|no_warm_up|close warm_up|
|batch_size|batch size for dynamic batch model, defaults to 1|
|unuse_iobinding|unuse iobinding api to run|
|inputshape_xxx|assign input shape ,xxx means input names|
|output_create_pre|create output tensors|
|maca_custom_op_lib|maca custom op lib path|
|cpu_custom_op_lib|cpu custom op lib path|
