#!/bin/bash

rm -rf test_result
mkdir test_result
mkdir test_result/x86
mkdir test_result/cuda

echo "-----------------------------------------------------------------------------------------------------------------------------------------"
echo "****************************************************    x86  ****************************************************************************"
PYTHONPATH=./pplnn-build/install/lib python3 ./tools/pplnn.py --use-x86 \
                                                              --onnx-model tests/testdata/conv.onnx \
                                                              --inputs tests/testdata/conv_input.data \
                                                              --save-data-dir test_result/x86 \
                                                              --save-outputs
echo "-----------------------------------------------------------------------------------------------------------------------------------------"

echo "-----------------------------------------------------------------------------------------------------------------------------------------"
echo "****************************************************    cuda  ***************************************************************************"
PYTHONPATH=./pplnn-build/install/lib python3 ./tools/pplnn.py --use-cuda \
                                                              --onnx-model tests/testdata/conv.onnx \
                                                              --inputs tests/testdata/conv_input.data \
                                                              --save-data-dir test_result/cuda \
                                                              --save-outputs
echo "-----------------------------------------------------------------------------------------------------------------------------------------"
echo "****************************************************    x86 vs cuda result info *********************************************************"

python result_file_compare.py test_result/x86 test_result/cuda
