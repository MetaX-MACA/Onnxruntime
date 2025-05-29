#!/bin/bash
CUR_PATH=$(pwd)
rm -rf build/  dist/  *.egg-info
python3 setup.py bdist_wheel clean

rm -rf  build/  *.egg-info
