// Copyright 2024 metax-tech.com Inc. All Rights Reserved.
#include "common.h"

double currentTime(){
#ifdef WIN32
    SYSTEMTIME sys_time;
    GetLocalTime(&sys_time);
    return sys_time.wSecond * 1000 + sys_time.wMilliseconds;
#else
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec * 1000 + tv.tv_usec / 1000.0;
#endif
}

std::unordered_map<ONNXTensorElementDataType, int > OrtTypeMapLen = {
    {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED, 1},
    {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 4},
    {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, 1},
    {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, 1},
    {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, 2},
    {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, 2},
    {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, 4},
    {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, 8},
    {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, 1},
    {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, 2},
    {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, 8},
    {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32, 4},
    {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64, 8}
};

std::ostream& operator<<(std::ostream& os,
                         const ONNXTensorElementDataType& type)
{
    switch (type)
    {
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
            os << "undefined";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            os << "float";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            os << "uint8_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            os << "int8_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            os << "uint16_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            os << "int16_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            os << "int32_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            os << "int64_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
            os << "std::string";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            os << "bool";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            os << "float16";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            os << "double";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            os << "uint32_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            os << "uint64_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
            os << "float real + float imaginary";
            break;
        case ONNXTensorElementDataType::
            ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
            os << "double real + float imaginary";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            os << "bfloat16";
            break;
        default:
            break;
    }

    return os;
}

std::vector<float> convertDataToFloat(void* output_ptr, ONNXTensorElementDataType outputType, int len){
    std::vector<float> output_data;
    if(outputType ==  ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
    {
        for (int j = 0; j < len; j++) {
            output_data.push_back(float(((int64_t *)output_ptr)[j]));
        }
    }else if (outputType ==  ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8)
    {
        for (int j = 0; j < len; j++) {
            output_data.push_back(float(((int8_t *)output_ptr)[j]));
        }
    }else if (outputType ==  ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
    {
        for (int j = 0; j < len; j++) {
            output_data.push_back(float(Float16(((uint16_t *)output_ptr)[j])));
        }
    }else if (outputType ==  ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    {
        for (int j = 0; j < len; j++) {
            output_data.push_back(((float *)output_ptr)[j]);
        }
    }else if (outputType ==  ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)
    {
        for (int j = 0; j < len; j++) {
            output_data.push_back(float(((uint8_t *)output_ptr)[j]));
        }
    }else if (outputType ==  ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64)
    {
        for (int j = 0; j < len; j++) {
            output_data.push_back(float(((u_int64_t *)output_ptr)[j]));
        }
    }else if (outputType ==  ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
    {
        for (int j = 0; j < len; j++) {
            output_data.push_back(float(((int32_t *)output_ptr)[j]));
            //  std::cout << float(((int32_t *)output_ptr)[j]) << ",";
        }

    }else{
        std::cout << "unkown output type (If you need, Please add it in code) : " << outputType << std::endl;
        abort();
    }
    return output_data;
}

void getOutputDataToFloat(Ort::Value& output, std::vector<float>& output_data, DATALOC output_data_loc){
    auto typeInfoMaca = output.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType outputType = typeInfoMaca.GetElementType();
    int len = typeInfoMaca.GetElementCount();
    size_t data_len = size_t(len * OrtTypeMapLen[outputType]);
    void* cpu_output_ptr = malloc(data_len);
    char* output_ptr = nullptr;
    if(output_data_loc != DATALOC::CPU){
        char* ori_output_ptr = output.GetTensorMutableData<char>();
        memoryCopy(cpu_output_ptr, DATALOC::CPU, ori_output_ptr, output_data_loc, data_len);
        output_ptr = (char*)cpu_output_ptr;
    }else{
        output_ptr = output.GetTensorMutableData<char>();
    }
    output_data = convertDataToFloat(output_ptr, outputType, len);
    delete cpu_output_ptr;
}

bool createInputData(void** dst_data,uint64_t* data_len, ONNXTensorElementDataType tensor_type, uint64_t tensor_size){

    if(tensor_type ==  ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
    {
        *data_len = createFixedLenData<int64_t>(*dst_data, tensor_size);
    }else if (tensor_type ==  ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8)
    {
        *data_len = createFixedLenData<int8_t>(*dst_data, tensor_size);
    }else if (tensor_type ==  ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
    {
        *data_len = createFixedLenData<uint16_t>(*dst_data, tensor_size);
    }else if (tensor_type ==  ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    {
        *data_len = createFixedLenData<float>(*dst_data, tensor_size);
    }else if (tensor_type ==  ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)
    {
        *data_len = createFixedLenData<u_int8_t>(*dst_data, tensor_size);
    }else if (tensor_type ==  ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64)
    {
        *data_len = createFixedLenData<u_int64_t>(*dst_data, tensor_size);
    }else if (tensor_type ==  ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
    {
        *data_len = createFixedLenData<int32_t>(*dst_data, tensor_size);
    }else if (tensor_type == ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16)
    {
        *data_len = createFixedLenData<int16_t>(*dst_data, tensor_size);
    }else {
        std::cout << "Unsupport Input type(If you need, Please add it in code) : " <<  tensor_type << std::endl;
        return false;
    }
    return true;
}

char* getCmdOption(char** begin, char** end, const std::string& option) {
    char** iter = std::find(begin, end, option);
    if (iter != end && ++iter != end) {
        return *iter;
    }
    return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option) {
  return std::find(begin, end, option) != end;
}


Ort::MemoryInfo getMemoryInfo(DATALOC data_loc){
    return getMemoryInfo(data_loc, 0);
}
Ort::MemoryInfo getMemoryInfo(DATALOC data_loc,int device_id){
    if(data_loc == DATALOC::CPU){
        return Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    }else if(data_loc == DATALOC::MACA_PINNED){
        return Ort::MemoryInfo(TEST_MACA_PINNED,OrtAllocatorType::OrtDeviceAllocator, 0,  OrtMemType::OrtMemTypeCPUOutput);
    }else if(data_loc == DATALOC::MACA){
        return Ort::MemoryInfo(TEST_MACA,OrtAllocatorType::OrtDeviceAllocator, device_id,  OrtMemType::OrtMemTypeDefault);
    }else{
        std::cout << "error! unkown data loc" << std::endl;
        abort();
    }
}

std::vector<const char *> getInputNames(const Ort::Session& session,OrtAllocator* allocator){
    size_t numInputNodes = session.GetInputCount();
    std::cout << "    Number of Input Nodes: " << numInputNodes << std::endl;
    std::vector<const char*> input_names;
    for (size_t i = 0; i < numInputNodes; i++) {
        const char* inputName = session.GetInputName(i, allocator);
        std::cout << "    Input " << i << " Name: " << inputName << std::endl;
        input_names.push_back(inputName);
        Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(i);
        auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
        std::cout << "    Input " << i << " Type: " << inputType << std::endl;
    }
    return input_names;
}
std::vector<const char *> getOutputNames(const Ort::Session& session,OrtAllocator* allocator){
    size_t numOutputNodes = session.GetOutputCount();
    std::vector<const char*> output_names;
    std::cout << "    Number of Output Nodes: " << numOutputNodes << std::endl;
    for (size_t i = 0; i < numOutputNodes; i++) {
        const char* outputName = session.GetOutputName(i, allocator);
        std::cout << "    Output " << i << " Name: " << outputName << std::endl;
        output_names.push_back(outputName);
        Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(i);
        auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
        std::cout << "    Output " << i << " Type: " << outputType << std::endl;
    }

    return output_names;
}


std::vector<Ort::Value> creatInputTensors(const Ort::Session& session, const Ort::MemoryInfo& memory_info , std::map<std::string, std::pair<void*, int>> & i_o_data){
    return creatInputTensorsImpl(session, std::unordered_map<std::string, std::vector<int64_t>>(), memory_info, 1, 1,i_o_data);
}
std::vector<Ort::Value> creatInputTensors(const Ort::Session& session, const Ort::MemoryInfo& memory_info , int batch_size ,int input_groups, std::map<std::string, std::pair<void*, int>> & i_o_data){
    return creatInputTensorsImpl(session,  std::unordered_map<std::string, std::vector<int64_t>>(), memory_info, batch_size, input_groups,i_o_data);
}
std::vector<Ort::Value> creatInputTensors(const Ort::Session& session, std::unordered_map<std::string, std::vector<int64_t>> input_shape_map, const Ort::MemoryInfo& memory_info, std::map<std::string, std::pair<void*, int>> & i_o_data){
    return creatInputTensorsImpl(session,  input_shape_map, memory_info, 1, 1,i_o_data);
}

std::vector<Ort::Value> creatInputTensorsImpl(const Ort::Session& session, std::unordered_map<std::string, std::vector<int64_t>> input_shape_map ,const Ort::MemoryInfo& memory_info,int batch_size ,int input_groups, std::map<std::string, std::pair<void*, int>> & i_o_data){
        std::vector<Ort::Value> input_tensors;
        Ort::AllocatorWithDefaultOptions allocator;
        size_t numInputNodes = session.GetInputCount();

        for(size_t i_g=0; i_g<size_t(input_groups); i_g++){

        for (size_t i = 0; i < numInputNodes; i++){
            const char* inputName = session.GetInputName(i, allocator);
            Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(i);
            auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
            ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
            std::vector<int64_t> s_input_shape;
            if(input_shape_map.find(std::string(inputName))==input_shape_map.end()) {
                s_input_shape = inputTensorInfo.GetShape();
                if(s_input_shape.empty()){
                    std::cout << "Warning! your model input shape is none, we will set to [1]" << std::endl;
                    s_input_shape.push_back(1);
                }
                s_input_shape[0] = std::max(int(s_input_shape[0]), batch_size);
                for(size_t i_s = 0; i_s < s_input_shape.size(); i_s++){
                    if(s_input_shape[i_s] <1) {
                        std::cout << "Error! your model include dynamic input, you should specify input shapes" << std::endl;
                        abort();
                    }
                }
            }
            else  {
               s_input_shape = input_shape_map[std::string(inputName)];
            }
            uint64_t inputTensorSize = vectorProduct(s_input_shape);
            void* inputTensorValues = nullptr; //will release when tensor destruct
            uint64_t data_len = 0;
            Ort::Allocator allocator = Ort::Allocator(session, memory_info);
            createInputData(&inputTensorValues,&data_len, inputType, inputTensorSize);
            if(i_o_data.find(std::string(inputName)) != i_o_data.end()){
                free(inputTensorValues);
                inputTensorValues = i_o_data[std::string(inputName)].first;
            }
            if(memory_info.GetAllocatorName() != "Cpu"){
                void* temp_cpu_ptr = inputTensorValues;
                inputTensorValues = allocator.Alloc(data_len);
                memoryCopy(inputTensorValues, DATALOC::MACA, temp_cpu_ptr,DATALOC::CPU , data_len);
                free(temp_cpu_ptr);
            }
            input_tensors.push_back(Ort::Value::CreateTensor(
                memory_info, inputTensorValues, data_len, s_input_shape.data(),
                s_input_shape.size(), inputType));
        }

        }
        return input_tensors;
}


std::vector<Ort::Value> creatOutputTensors(const Ort::Session& session, const Ort::MemoryInfo& memory_info ){
    return creatOutputTensorsImpl(session, std::unordered_map<std::string, std::vector<int64_t>>(), memory_info, 1, 1);
}
std::vector<Ort::Value> creatOutputTensors(const Ort::Session& session, const Ort::MemoryInfo& memory_info , int batch_size ,int input_groups){
    return creatOutputTensorsImpl(session,  std::unordered_map<std::string, std::vector<int64_t>>(), memory_info, batch_size, input_groups);
}
std::vector<Ort::Value> creatOutputTensors(const Ort::Session& session, std::unordered_map<std::string, std::vector<int64_t>> input_shape_map, const Ort::MemoryInfo& memory_info){
    return creatOutputTensorsImpl(session,  input_shape_map, memory_info, 1, 1);
}

std::vector<Ort::Value> creatOutputTensorsImpl(const Ort::Session& session, std::unordered_map<std::string, std::vector<int64_t>> input_shape_map ,const Ort::MemoryInfo& memory_info,int batch_size ,int input_groups){
        std::vector<Ort::Value> input_tensors;
        Ort::AllocatorWithDefaultOptions allocator;
        size_t numInputNodes = session.GetOutputCount();

        for(size_t i_g=0; i_g<size_t(input_groups); i_g++){

        for (size_t i = 0; i < numInputNodes; i++){
            const char* inputName = session.GetOutputName(i, allocator);
            Ort::TypeInfo inputTypeInfo = session.GetOutputTypeInfo(i);
            auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
            ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
            std::vector<int64_t> s_input_shape;
            if(input_shape_map.find(std::string(inputName))==input_shape_map.end()) {
                s_input_shape = inputTensorInfo.GetShape();
                if(s_input_shape.empty()){
                    std::cout << "Warning! your model input shape is none, we will set to [1]" << std::endl;
                    s_input_shape.push_back(1);
                }
                s_input_shape[0] = std::max(int(s_input_shape[0]), batch_size);
                for(size_t i_s = 0; i_s < s_input_shape.size(); i_s++){
                    if(s_input_shape[i_s] <1) {
                        std::cout << "Error! your model include dynamic input, you should specify input shapes" << std::endl;
                        abort();
                    }
                }
            }
            else  {
               s_input_shape = input_shape_map[std::string(inputName)];
            }
            uint64_t inputTensorSize = vectorProduct(s_input_shape);
            void* inputTensorValues = nullptr; //will release when tensor destruct
            uint64_t data_len = 0;
            Ort::Allocator allocator = Ort::Allocator(session, memory_info);
            createInputData(&inputTensorValues,&data_len, inputType, inputTensorSize);
            if(memory_info.GetAllocatorName() != "Cpu"){
                void* temp_cpu_ptr = inputTensorValues;
                inputTensorValues = allocator.Alloc(data_len);
                memoryCopy(inputTensorValues, DATALOC::MACA, temp_cpu_ptr,DATALOC::CPU , data_len);
                free(temp_cpu_ptr);
            }
            input_tensors.push_back(Ort::Value::CreateTensor(
                memory_info, inputTensorValues, data_len, s_input_shape.data(),
                s_input_shape.size(), inputType));
        }

        }
        return input_tensors;
}

std::vector<std::tuple<void*, uint64_t, std::vector<int64_t>>> getDataFromTensorValues(std::vector<Ort::Value>& values,DATALOC value_data_loc){
    std::vector<std::tuple<void*, uint64_t, std::vector<int64_t>>> res;
    for(size_t i=0; i<values.size(); i++){
        Ort::Value* value = &values[i];
        // std::tuple<void*, uint64_t, std::vector<int64_t>> data_tuple;
        auto typeInfoMaca = value->GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType outputType = typeInfoMaca.GetElementType();
        std::vector<int64_t> tensor_shape = typeInfoMaca.GetShape();
        int len = typeInfoMaca.GetElementCount();
        uint64_t data_len = len * OrtTypeMapLen[outputType];
        void* data_ptr = malloc(data_len);
        char* value_ptr = value->GetTensorMutableData<char>();
        memoryCopy(data_ptr,DATALOC::CPU,value_ptr,value_data_loc,data_len);
        res.push_back({data_ptr, data_len, tensor_shape});
    }
    return res;
}

AccuracyInfo compareOutput(const std::vector<float>& gold_output, const std::vector<float>& test_output){
    if(gold_output.size()!=test_output.size()){
        std::cout<<"error! gold_output.size()!=test_output.size()" << std::endl;
        abort();
    }
    AccuracyInfo info;
    float err_square_sum = 0;
    float gold_square_sum = 0;
    for(size_t i=0; i<gold_output.size(); i++){
        if(i < 5){
            std::cout << "gold: " << gold_output[i] <<"; maca: "<<test_output[i] << std::endl;
        }
        float err = abs(gold_output[i] - test_output[i]);
        err_square_sum += (err*err);
        gold_square_sum += (gold_output[i] * gold_output[i]);
        info.max_error = std::max(info.max_error,err);
    }
    info.snr = err_square_sum/(gold_square_sum + 1e-7) ;
    info.mse = float(err_square_sum)/gold_output.size();
    return info;
}


void saveTensorToFile(Ort::Value& value, std::string file_path){
    std::ofstream file_w(file_path, std::ios::out | std::ios::binary);
    auto typeInfo = value.GetTensorTypeAndShapeInfo();
    size_t len = typeInfo.GetElementCount();
    ONNXTensorElementDataType outputType = typeInfo.GetElementType();
    size_t data_len = len * OrtTypeMapLen[outputType];
    char* data_ptr = value.GetTensorMutableData<char>();
    file_w.write(data_ptr, data_len);
    file_w.close();
}

void saveTensorsToDir(std::vector<const char*> names, std::vector<Ort::Value>& values, std::string dir_path, std::string prefix){
    std::string save_path = dir_path + "/" + prefix;
    if(fs::exists(save_path) == false){
        fs::create_directory(save_path);
    }
    std::string i_o_path_txt = dir_path + "/i_o_path.txt";
    std::ofstream i_o_path_file(i_o_path_txt, std::ios::app);

    for (size_t i = 0; i < values.size(); i++) {
        std::string file_path = save_path + "/" + std::to_string(i) +".dat";
        i_o_path_file << names[i] << std::endl;
        i_o_path_file << file_path << std::endl;
        saveTensorToFile(values[i],file_path);
    }
    i_o_path_file.close();
}

void memoryCopy(void* dst, DATALOC dst_loc,const void* src, DATALOC src_loc ,size_t data_len){
    if(dst_loc == DATALOC::CPU && src_loc != DATALOC::CPU){
        mcMemcpy(dst, src, data_len, mcMemcpyDeviceToHost);
    }else if(dst_loc != DATALOC::CPU && src_loc == DATALOC::CPU){
        mcMemcpy(dst, src, data_len, mcMemcpyHostToDevice);
    }else{
        memcpy(dst,src,data_len);
    }
}

std::pair<void*, int> readData(std::string data_path){
    std::ifstream file(data_path, std::ios::binary);

    if (!file) {
        std::cout << "open data_path failed !" << data_path << std::endl;
    }


    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    void* buffer = malloc(fileSize);
    file.read((char*)buffer, fileSize);

    std::cout << "Read: " <<data_path << "; data_len: " << fileSize << std::endl;
    file.close();
    return {buffer, fileSize};
}

std::vector<std::vector<float>> readOutputData(const Ort::Session& session, std::map<std::string, std::pair<void*, int>> & i_o_data){
    std::vector<std::vector<float>> output_res;
    Ort::AllocatorWithDefaultOptions allocator;
    size_t numOutputNodes = session.GetOutputCount();
    for (size_t i = 0; i < numOutputNodes; i++){
        const char* outputName = session.GetOutputName(i, allocator);
        Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(i);
        auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();

        std::vector<float> output_data = convertDataToFloat(i_o_data[std::string(outputName)].first, outputType, i_o_data[std::string(outputName)].second / OrtTypeMapLen[outputType]);
        output_res.push_back(output_data);

    }
    return output_res;
}
