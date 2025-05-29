
CURRENR_PATH=$(pwd)
SCRIPT_DIR=$(cd $(dirname $0); pwd)
MACA_ROOT="${SCRIPT_DIR}/../"

echo ${SCRIPT_DIR}
echo ${MACA_ROOT}

VERSION_FILE="${MACA_ROOT}/maca_quantizer/CORE_VERSION"
if [ -e ${VERSION_FILE} ];then
    rm  ${VERSION_FILE}
fi


cd "${MACA_ROOT}/maca_quantizer/ppq_/"
COMMIT_ID=$(git rev-parse HEAD)
if [ "$?" -eq "0" ] && [ -n "${COMMIT_ID}" ]; then
    echo ${COMMIT_ID} >> "${VERSION_FILE}"
    echo "Generate core commit id."
else
    echo "NULL" >> "${VERSION_FILE}"
    echo "Can not get core commit id."
fi
cd ${CURRENR_PATH}