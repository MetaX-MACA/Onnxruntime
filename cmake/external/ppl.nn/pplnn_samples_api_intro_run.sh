#!/bin/bash

workdir=`pwd`

export MACA_PATH=/home/hxu/docker_shared/C500/maca/maca-current
export LD_LIBRARY_PATH=$MACA_PATH/lib:$LD_LIBRARY_PATH
export PATH=${PATH}:${workdir}/pplnn-build/samples/cpp/api

pplnn_samples_api_intro
