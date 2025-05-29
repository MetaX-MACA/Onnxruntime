#!/bin/bash

check_status(){
    if [ $? -ne 0 ]; then
        echo "${1} Failed!"
        exit 1
    else
        echo "${1} Successful!"
    fi
}

echo "init conda python environment...."
platform=$(uname -m)
echo "platform:  "${platform}
source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda init bash

conda env update -f .robot/resources/env/onnxruntime-c500-cp38.yaml
conda activate onnxruntime-c500-cp38
check_status "conda activate onnxruntime-c500-cp38"

./build.sh "$@"
check_status "build.sh"

echo "exit python environment..."
conda deactivate
check_status "conda deactivate"

