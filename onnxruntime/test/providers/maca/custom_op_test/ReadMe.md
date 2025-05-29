# Sample
This project will guide you how to build and use custom op
## 1. Build
Installed maca_driver & onnxruntime_maca
```
make
```
## 2. check result
### maca_test
```
/opt/maca-ai/onnxruntime-maca/bin/maca_test --model_dir ./custom_op_test.onnx --num_tests 1 --num_threads 1 --check_mode 1 --maca_custom_op_lib libcustom_op_library_maca.so --cpu_custom_op_lib libcustom_op_library_cpu.so
```
