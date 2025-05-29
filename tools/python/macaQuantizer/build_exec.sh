#!/bin/bash
OldPath=$(pwd)


find_file(){
  PythonPath=$(which python3)
  DirPath=$(dirname "${PythonPath%/*}")
  # echo "======== python environment direction is: ${DirPath} ========"
  ResPath=$(find "${DirPath}" -name "$1" | head -1)
  if [ -d "${ResPath}" ];
  then
    ResPath="${ResPath}"'/'
  fi
  echo "$ResPath"
}

# PythonInclude=$(find_file Python.h)
# echo $PythonInclude


build_execute(){
  echo "run_execute"

  PythonOption=$(which python3)
  EXEC_NAME="mxq_exec"

  echo "$EXEC_NAME"

  pyi-makespec -D   --specpath build/ --name "$EXEC_NAME" --python-option "$PythonOption" \
                    --paths "$OldPath:$OldPath/maca_quantizer/core/:$OldPath/maca_quantizer/ppq_/:$OldPath/maca_quantizer/preprocess/:$OldPath/maca_quantizer/utils/" \
                    --hidden-import "$OldPath/maca_quantizer/ppq_/"  \
                    --add-data  "$OldPath/maca_quantizer/ppq_/ppq/csrc:ppq/csrc/"  \
                    # --exclude-module "torch" \
                    "$1"
                    # --exclude-module "torch" \
                    # --add-data  "/opt/conda/pkgs/mkl-2020.4-h726a3e6_304/lib/:." --add-data  "$PythonInclude:."  "$1"

  sed -i '1a\import sys\nsys.setrecursionlimit(5000)' "build/$EXEC_NAME.spec"
  echo "Make spec finished"

  pyinstaller --noconfirm  --distpath release "build/$EXEC_NAME.spec"

}

build_execute maca_quantizer/__main__.py
cd "$OldPath" || exit