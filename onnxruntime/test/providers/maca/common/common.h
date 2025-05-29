// Copyright 2024 metax-tech.com Inc. All Rights Reserved.
#pragma once
#include <onnxruntime_cxx_api.h>
#include <experimental/filesystem>
#ifdef WIN32
#include <Windows.h>
#else
#include <sys/time.h>
#endif

#include <mcr/mc_runtime_api.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <numeric>
#include <functional>
#include <unordered_map>
#include <tuple>
#include <map>

namespace fs = std::experimental::filesystem;
constexpr const char* TEST_MACA = "Maca";
constexpr const char* TEST_MACA_PINNED = "MacaPinned";
extern std::unordered_map<ONNXTensorElementDataType, int > OrtTypeMapLen;
double currentTime();
static int rand_seed = 0;

using float16_t = uint16_t;
// loss precesion , will find better way to accomplish
struct Float16
{
    float16_t _value;
    Float16(float16_t value):_value(value){}
    Float16(float value){
        uint32_t m = *(uint32_t *) &value;
        _value = ((m & 0x7fffffff) >> 13) - (0x38000000 >> 13);
        _value |= ((m & 0x80000000) >> 16);
    }
    operator float16_t(){
        return _value;
    }
    operator float(){
        uint32_t m = _value;
        m = ((m & 0x7fff) << 13) + 0x38000000;
        m |= ((_value & 0x8000) << 16);
        return *(float*) &m;
    }
};

enum class DATALOC {
    CPU = 0,
    MACA_PINNED = 1,
    MACA = 2,
};

struct AccuracyInfo{
    float snr=0;
    float mse=0;
    float max_error=0;
};

template <typename T>
void creatRandValuePoint(T* p,long int num){
    // std::srand((unsigned)time(NULL));
    std::srand(rand_seed);
    for(int i=0; i<num; i++){
        if(sizeof(T) == 2) p[i] = float16_t(Float16(float(std::rand()%255)));
        else p[i] = std::rand()%2;
        // std::cout << p[i] << ",";
    }
    rand_seed += 11;
    // std::cout << std::endl;
}

template <typename T>
uint64_t createFixedLenData(void* & src, uint64_t len){
    uint64_t data_len = len * sizeof(T);
    src = malloc(data_len);
    creatRandValuePoint((T*)src, len);
    return data_len;
}


template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

template <typename T>
std::vector<T> splitStringToNumber(char * data_string, char split_symbol){
    std::vector<T> number_vec;
    std::stringstream ss(data_string);
    T n_value = 0;
    std::string number_str;
    while(getline(ss, number_str, split_symbol)){
        n_value = T(atof(number_str.c_str()));
        number_vec.push_back(n_value);
    }
    return number_vec;
}

/**
 * @brief Operator overloading for printing vectors
 * @tparam T
 * @param os
 * @param v
 * @return std::ostream&
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (size_t i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i != v.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

/**
 * @brief Print ONNX tensor data type
 * https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L93
 * @param os
 * @param type
 * @return std::ostream&
 */
std::ostream& operator<<(std::ostream& os,
                         const ONNXTensorElementDataType& type);



// help parse argv
char* getCmdOption(char** begin, char** end, const std::string& option);
bool cmdOptionExists(char** begin, char** end, const std::string& option);

// create memory info
Ort::MemoryInfo getMemoryInfo(DATALOC data_loc);
Ort::MemoryInfo getMemoryInfo(DATALOC data_loc, int device_id);

// get input names according session
std::vector<const char *> getInputNames(const Ort::Session& session, OrtAllocator* allocator);
std::vector<const char *> getOutputNames(const Ort::Session& session, OrtAllocator* allocator);

// create input buffer, init with random value
bool createInputData(void** dst_data,uint64_t* data_len, ONNXTensorElementDataType tensor_type, uint64_t tensor_size, Ort::Allocator& malloc_fn);

// create input tensors by random value
std::vector<Ort::Value> creatInputTensorsImpl(const Ort::Session& session, std::unordered_map<std::string, std::vector<int64_t>> input_shape_map,const Ort::MemoryInfo& memory_info, int batch_size,int input_groups, std::map<std::string, std::pair<void*, int>> & i_o_data);
std::vector<Ort::Value> creatInputTensors(const Ort::Session& session, const Ort::MemoryInfo& memory_info , std::map<std::string, std::pair<void*, int>> & i_o_data);
std::vector<Ort::Value> creatInputTensors(const Ort::Session& session, const Ort::MemoryInfo& memory_info , int batch_size,int input_groups, std::map<std::string, std::pair<void*, int>> & i_o_data);
std::vector<Ort::Value> creatInputTensors(const Ort::Session& session, std::unordered_map<std::string, std::vector<int64_t>> input_shape_map, const Ort::MemoryInfo& memory_info, std::map<std::string, std::pair<void*, int>> & i_o_data);

// create output tensors by random value
std::vector<Ort::Value> creatOutputTensorsImpl(const Ort::Session& session, std::unordered_map<std::string, std::vector<int64_t>> input_shape_map,const Ort::MemoryInfo& memory_info, int batch_size,int input_groups);
std::vector<Ort::Value> creatOutputTensors(const Ort::Session& session, const Ort::MemoryInfo& memory_info );
std::vector<Ort::Value> creatOutputTensors(const Ort::Session& session, const Ort::MemoryInfo& memory_info , int batch_size,int input_groups);
std::vector<Ort::Value> creatOutputTensors(const Ort::Session& session, std::unordered_map<std::string, std::vector<int64_t>> input_shape_map, const Ort::MemoryInfo& memory_info);

// extract float output from Ort:Value
void getOutputDataToFloat(Ort::Value& output, std::vector<float>& output_data, DATALOC output_data_loc = DATALOC::CPU);

// extract buffer from std::vector<Ort::Value>
std::vector<std::tuple<void*, uint64_t, std::vector<int64_t>>> getDataFromTensorValues(std::vector<Ort::Value>& values,DATALOC value_data_loc );

AccuracyInfo compareOutput(const std::vector<float>& gold_output, const std::vector<float>& test_output);

// save Tensor to file
void saveTensorToFile(Ort::Value& value, std::string file_path);

// save Tensors to Dir
void saveTensorsToDir(std::vector<const char*> names, std::vector<Ort::Value>& values, std::string dir_path, std::string prefix);

// copy data
void memoryCopy(void* dst, DATALOC dst_loc,const void* src, DATALOC src_loc ,size_t data_len);


// read data
std::pair<void*, int> readData(std::string data_path);

// read output data
std::vector<std::vector<float>> readOutputData(const Ort::Session& session, std::map<std::string, std::pair<void*, int>> & i_o_data);
