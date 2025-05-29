#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Get directory this script is in
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OS=$(uname -s)

if [ "$OS" = "Darwin" ]; then
    DIR_OS="MacOS"
else
    DIR_OS="Linux"
fi

if [[ "$*" == *"--ios"* ]]; then
    DIR_OS="iOS"
elif [[ "$*" == *"--android"* ]]; then
    DIR_OS="Android"
fi

#requires python3.6 or higher
check_status(){
    if [ $? -ne 0 ]; then
        echo "${1} Failed!"
        exit 1
    else
        echo "${1} Successful!"
    fi
}

python3 $DIR/tools/ci_build/build.py --build_dir $DIR/build/$DIR_OS "$@"
check_status "Build onnxruntime"
python3 $DIR/tools/python/maca_tools_build.py --build_dir $DIR/build/$DIR_OS "$@"
check_status "Build maca tools"
python3 $DIR/tools/ci_build/package.py --build_dir $DIR/build/$DIR_OS "$@"
check_status "Package release"
