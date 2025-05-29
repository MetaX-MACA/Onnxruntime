// Copyright 2024 metax-tech.com Inc. All Rights Reserved.
#include <onnxruntime_cxx_api.h>

#include <assert.h>
#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <dlfcn.h>
#include <string>
#include <vector>
#include <iomanip>
#include <thread>
#include <mutex>

#include "common.h"

struct ModelParallelTime
{
    double parallel_begin_time = 0;
    double parallel_end_time = 0;
};

enum class CHECKMODE
{
    FPS_MODE = 0,
    CPU_ACCURACY_COMPARE_MODE = 1,
    INT8_ACCURACY_COMPARE_MODE = 2,
    FP16_ACCURACY_COMPARE_MODE = 3,
    DATA_ACCURACY_COMPARE_MODE = 4
};


double total_begin_time = 0;
double total_end_time = 0;

struct SessionRunArgs {
    Ort::Session* session;
    void* stream = nullptr;
    std::vector<const char*> inputNames;
    Ort::Value* inputTensors;
    size_t numInputNodes;
    bool use_iobinding = true;
    std::vector<std::vector<float>> goldOutput;
    std::vector<const char*> outputNames;
    Ort::Value* outputTensors = nullptr;
    std::vector<std::vector<AccuracyInfo>> multiOutputAccuracyInfos;
    DATALOC outputLoc;
    CHECKMODE check_mode;
    int device_id = 0;
    size_t numOutputNodes;
    size_t numInputs;
    size_t numTests;
    int num_model_io_groups=1;
    double maxLantencyTime = 0;
    double minLantencyTime = INT64_MAX;
    double totalLantencyTime = 0;
    ModelParallelTime* paralleTime = nullptr;
};


void sessionRunWithOutput(struct SessionRunArgs* args){
if(args->paralleTime->parallel_begin_time == 0) args->paralleTime->parallel_begin_time =  currentTime();
    if(total_begin_time == 0) total_begin_time = currentTime();
    for(size_t i=0; i<args->numTests; i++){
        double begin_time = currentTime();
        if(args->use_iobinding){
            Ort::IoBinding ioBinding(*args->session);
            for(size_t b=0; b<size_t(args->num_model_io_groups); b++){
                for(size_t i=0; i< args->inputNames.size(); i++){
                    ioBinding.BindInput(args->inputNames[i],args->inputTensors[i+b*args->inputNames.size()]);
                }
                for(size_t j=0 ;j <args->numOutputNodes; j++){
                    ioBinding.BindOutput(args->outputNames[j], args->outputTensors[j+b*args->outputNames.size()]);
                }
            }
            args->session->Run(Ort::RunOptions{nullptr},ioBinding);
        }else{
            std::vector<const char*> input_names;
            std::vector<const char*> output_names;
            for(size_t i=0; i<size_t(args->num_model_io_groups); i++){
                for(auto p : args->inputNames) input_names.push_back(p);
                for(auto p : args->outputNames) output_names.push_back(p);
            }
            args->session->Run(Ort::RunOptions{nullptr}, input_names.data(),
                            args->inputTensors, input_names.size(), output_names.data(),args->outputTensors, output_names.size());
        }

        double end_time = currentTime();
        double inferenceTime = end_time - begin_time;
        std::vector<AccuracyInfo> a_info;
        if(args->check_mode != CHECKMODE::FPS_MODE){
            std::vector<AccuracyInfo> a_info;
            for (size_t k = 0; k <args->numOutputNodes; k++){
                std::vector<float> output_data;
                getOutputDataToFloat(args->outputTensors[k], output_data, args->outputLoc);
                a_info.push_back(compareOutput(args->goldOutput[k],output_data));
            }
            args->multiOutputAccuracyInfos.push_back(a_info);
        }
        args->maxLantencyTime = std::max(args->maxLantencyTime, inferenceTime);
        args->minLantencyTime = std::min(args->minLantencyTime, inferenceTime);
        args->totalLantencyTime += inferenceTime;
    }

    // auto outputs = args->session->Run(Ort::RunOptions{nullptr}, args->inputNames.data(),
    //             args->inputTensors, args->numInputNodes, args->outputNames.data(), args->numOutputNodes);

    args->paralleTime->parallel_end_time = currentTime();
    total_end_time = currentTime();
}

void sessionRunNoOutput(struct SessionRunArgs* args){
if(args->paralleTime->parallel_begin_time == 0) args->paralleTime->parallel_begin_time =  currentTime();
    if(total_begin_time == 0) total_begin_time = currentTime();
    for(size_t i=0; i<args->numTests; i++){
        std::vector<Ort::Value> output_tensors;
        double begin_time = currentTime();
        if(args->use_iobinding){
            Ort::IoBinding ioBinding(*args->session);
            for(size_t b=0; b<size_t(args->num_model_io_groups); b++){
                for(size_t i=0; i< args->inputNames.size(); i++){
                    ioBinding.BindInput(args->inputNames[i],args->inputTensors[i+b*args->inputNames.size()]);
                }
                for(size_t j=0 ;j <args->numOutputNodes; j++){
                    ioBinding.BindOutput(args->outputNames[j], getMemoryInfo(args->outputLoc, args->device_id));
                }
            }
            args->session->Run(Ort::RunOptions{nullptr},ioBinding);
            output_tensors = ioBinding.GetOutputValues();
        }else{
            std::vector<const char*> input_names;
            std::vector<const char*> output_names;
            for(size_t i=0; i<size_t(args->num_model_io_groups); i++){
                for(auto p : args->inputNames) input_names.push_back(p);
                for(auto p : args->outputNames) output_names.push_back(p);
            }
            output_tensors = args->session->Run(Ort::RunOptions{nullptr}, input_names.data(),
                            args->inputTensors, input_names.size(), output_names.data(), output_names.size());


        }

        double end_time = currentTime();
        double inferenceTime = end_time - begin_time;
        std::vector<AccuracyInfo> a_info;
        if(args->check_mode != CHECKMODE::FPS_MODE){
            std::vector<AccuracyInfo> a_info;
            for (size_t k = 0; k < output_tensors.size(); k++){
                std::vector<float> output_data;
                getOutputDataToFloat(output_tensors[k], output_data, args->outputLoc);
                a_info.push_back(compareOutput(args->goldOutput[k],output_data));
            }
            args->multiOutputAccuracyInfos.push_back(a_info);
        }
        args->maxLantencyTime = std::max(args->maxLantencyTime, inferenceTime);
        args->minLantencyTime = std::min(args->minLantencyTime, inferenceTime);
        args->totalLantencyTime += inferenceTime;
    }

    // auto outputs = args->session->Run(Ort::RunOptions{nullptr}, args->inputNames.data(),
    //             args->inputTensors, args->numInputNodes, args->outputNames.data(), args->numOutputNodes);

    args->paralleTime->parallel_end_time = currentTime();
    total_end_time = currentTime();
}

void sessionRun(struct SessionRunArgs* args) {
    if(args->outputTensors){
        sessionRunWithOutput(args);
    }else{
        sessionRunNoOutput(args);
    }
}


void sessionRunMulti(SessionRunArgs* args,size_t  num_test){
    for(size_t i = 0;i<num_test;i++ ){
        sessionRun(args++);
    }
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cout << "Error, program inputs not enough." << std::endl;
        std::cout << "Usage: maca_test --model_dir xxx --num_tests 1 --num_threads 1 --num_inputs 1 --device_id 0  --snr_threshold 0.1 "
                "--check_mode 0  --log_level 1"
                << std::endl;
        std::cout << "    model_dir: dir or file path of onnx models" << std::endl;
        std::cout << "    num_tests: how many inference per thread. defaults to 100" << std::endl;
        std::cout << "    num_inputs: how many different input per thread. defaults to 1" << std::endl;
        std::cout << "    num_threads: how many threads to run each model. defaults to 32" << std::endl;
        std::cout << "    device_id: defaults to 0" << std::endl;
        std::cout << "    input_data_loc: maca ep location of input data. 0-cpu, 1-maca_pinned, 2-maca, defaults to 2  " << std::endl;
        std::cout << "    output_data_loc: maca ep location of output data. 0-cpu, 1-maca_pinned, 2-maca, defaults to 2  " << std::endl;
        std::cout << "    log_level: defaults to 4" << std::endl;
        std::cout << "    snr_threshold: snr_threshold to print log. defaults to 0.1" << std::endl;
        std::cout << "    check_mode: 0 fps mode, 1 accuracy mode. defaults to 0" << std::endl;
        std::cout << "    save_io: save input and output " << std::endl;
        std::cout << "    no_warm_up: close warm up " << std::endl;
        std::cout << "    unuse_iobinding: unuse iobinding api to run " << std::endl;
        std::cout << "    batch_size: batch size for dynamic batch model, defaults to 1 " << std::endl;
        std::cout << "    inputshape_xxx: assign input shape ,xxx means input names " << std::endl;
        std::cout << "    output_create_pre: create output tensors" << std::endl;
        std::cout << "    maca_custom_op_lib: maca custom op lib path " << std::endl;
        std::cout << "    cpu_custom_op_lib: cpu custom op lib path " << std::endl;
        return 1;
    }

    if (!cmdOptionExists(argv, argv+argc, "--model_dir")) {
        std::cout << "Argument model_dir must be provided" <<std::endl;
        return 1;
    }
    bool save_io = false;
    if (cmdOptionExists(argv, argv+argc, "--save_io")){
        save_io = true;
    }
    bool no_warm_up = false;
    if (cmdOptionExists(argv, argv+argc, "--no_warm_up")){
        no_warm_up = true;
    }

    std::string modelDir{getCmdOption(argv, argv + argc, "--model_dir")};
    if(fs::exists(modelDir) == false){
        std::cout << "error! Dir or file pathis not exists! -> " << modelDir << std::endl;
        return 1;
    }

    std::vector<std::string> modelFiles;
    std::vector<std::string> modelCompareFiles;
    std::vector<std::string> save_io_paths;
    if(fs::is_directory(modelDir)){
        for (auto p : fs::recursive_directory_iterator(modelDir)) {
            if (p.path().extension().string() != ".onnx") continue;
            modelFiles.push_back(p.path().string());
            if(save_io){
                std::string save_io_path = p.path().parent_path().string() +"/" + p.path().stem().string() + "_io";
                if(fs::exists(save_io_path) == false){
                    fs::create_directory(save_io_path);
                }
                save_io_paths.push_back(save_io_path);
            }
        }
    }else{
        modelFiles.push_back(modelDir);
        fs::path p = modelDir;
        if(save_io){
            std::string save_io_path = p.parent_path().string() +"/" + p.stem().string() + "_io";
            if(fs::exists(save_io_path) == false){
                fs::create_directory(save_io_path);
            }
            save_io_paths.push_back(save_io_path);
        }
    }


    size_t numModels = modelFiles.size();
    ModelParallelTime model_parallel_time[numModels];
    size_t numThreads = 32;
    if (cmdOptionExists(argv, argv+argc, "--num_threads"))
        numThreads = atoi(getCmdOption(argv, argv+argc, "--num_threads"));
    std::vector<int> device_id_list;
    if (cmdOptionExists(argv, argv+argc, "--device_id"))
        device_id_list = splitStringToNumber<int>(getCmdOption(argv, argv+argc, "--device_id"), ',');

    DATALOC input_loc = DATALOC::MACA;
    DATALOC output_loc = DATALOC::MACA;
    if (cmdOptionExists(argv, argv+argc, "--output_data_loc"))
        output_loc = DATALOC(atoi(getCmdOption(argv, argv+argc, "--output_data_loc")) % 3);
    if (cmdOptionExists(argv, argv+argc, "--input_data_loc"))
        input_loc = DATALOC(atoi(getCmdOption(argv, argv+argc, "--input_data_loc")) % 3);
    size_t num_model_io_groups = 1;
    size_t batch_size = 1;
    if (cmdOptionExists(argv, argv+argc, "--batch_size"))
        batch_size = atoi(getCmdOption(argv, argv+argc, "--batch_size"));
    if(batch_size>1) {
        num_model_io_groups=1;
    }
    size_t numTests=100;
    size_t numInputs = 1;
    if (cmdOptionExists(argv, argv+argc, "--num_tests")){
        numTests = atoi(getCmdOption(argv, argv+argc, "--num_tests"));
    }
    if (cmdOptionExists(argv, argv+argc, "--num_inputs")){
        numInputs = atoi(getCmdOption(argv, argv+argc, "--num_inputs"));
    }
    float snr_threshold = 0.1;
    if (cmdOptionExists(argv, argv+argc, "--snr_threshold")){
        snr_threshold = atof(getCmdOption(argv, argv+argc, "--snr_threshold"));
    }
    bool use_iobinding = true;
    if (cmdOptionExists(argv, argv+argc, "--unuse_iobinding")){
        use_iobinding = false;
    }

    bool output_create_pre = false;
    if (cmdOptionExists(argv, argv+argc, "--output_create_pre")){
        output_create_pre = true;
    }

    int log_level = 4;
    if (cmdOptionExists(argv, argv+argc, "--log_level"))
        log_level = atoi(getCmdOption(argv, argv+argc, "--log_level"));
    CHECKMODE check_mode = CHECKMODE::FPS_MODE;
    if (cmdOptionExists(argv, argv+argc, "--check_mode"))
        check_mode = CHECKMODE(atoi(getCmdOption(argv, argv+argc, "--check_mode")));

    std::string model_accuracy_extend_str = "";
    if(check_mode == CHECKMODE::INT8_ACCURACY_COMPARE_MODE){
        model_accuracy_extend_str = "_int8";
    }else if(check_mode == CHECKMODE::FP16_ACCURACY_COMPARE_MODE){
        model_accuracy_extend_str = "_fp16";
    }

    std::string i_o_txt = "";
    if (cmdOptionExists(argv, argv+argc, "--i_o_path")){
        i_o_txt = std::string(getCmdOption(argv, argv+argc, "--i_o_path"));
    }

    for(auto p : modelFiles){
        fs::path fs_path_obj = p;
        std::string compare_model_path = fs_path_obj.parent_path().string() +"/" + fs_path_obj.stem().string() + model_accuracy_extend_str + ".onnx";
        if(fs::exists(compare_model_path) == false){
            std::cout << "Error! Not found model -> " << compare_model_path << std::endl;
            return 1;
        }
        modelCompareFiles.push_back(compare_model_path);
    }

    char* maca_custom_op_lib_path = nullptr;
    if(cmdOptionExists(argv, argv+argc, "--maca_custom_op_lib"))
        maca_custom_op_lib_path = getCmdOption(argv, argv+argc, "--maca_custom_op_lib");

    void* maca_library_handle = nullptr;

    char* cpu_custom_op_lib_path = nullptr;
    if(cmdOptionExists(argv, argv+argc, "--cpu_custom_op_lib"))
        cpu_custom_op_lib_path = getCmdOption(argv, argv+argc, "--cpu_custom_op_lib");

    void* cpu_library_handle = nullptr;

    std::string instanceName{"Onnxruntime Maca EP test"};
    Ort::Env env(OrtLoggingLevel(log_level),instanceName.c_str());
    std::string compare_instanceName ="Compare" + model_accuracy_extend_str +"_TEST";
    Ort::Env compare_env(OrtLoggingLevel(log_level),compare_instanceName.c_str());
    if(save_io){
        std::cout << "Warning!!! For save_io, we will set input_data_loc=0,output_data_loc=0"<<std::endl;
        input_loc = DATALOC::CPU;
        output_loc = DATALOC::CPU;
    }

    std::map<std::string, std::pair<void*, int>> i_o_data;
    if(check_mode == CHECKMODE::DATA_ACCURACY_COMPARE_MODE){
        std::cout << "Warning!!! For DATA_ACCURACY_COMPARE_MODE, we will set input_data_loc=0,output_data_loc=0,num_tests=1,num_threads=1"<< std::endl;
        input_loc = DATALOC::CPU;
        output_loc = DATALOC::CPU;
        numTests = 1;
        numThreads = 1;
        if(modelFiles.size() != 1){
            std::cout << "Error!!! For DATA_ACCURACY_COMPARE_MODE, only support single model" << std::endl;
            return -1;
        }
        std::map<std::string, std::string> i_o_map_path;
        std::ifstream i_o_txt_file(i_o_txt);
        if (!i_o_txt_file.is_open()) {
            std::cout << "fail open txt file!" << i_o_txt << std::endl;
            return -1;
        }
        std::vector<std::string> lines;
        std::string line;
        while (std::getline(i_o_txt_file, line)) {
            if(line.empty()){
                continue;
            }
            lines.push_back(line);
        }
        i_o_txt_file.close();
        if(lines.size() % 2 != 0){
            std::cout << "Error!!! parse i_o_txt fail" << std::endl;
            return -1;
        }
        for(size_t i = 0; i<(lines.size()/2); i++){
            i_o_map_path[lines[i*2]] = lines[i*2 + 1];
        }
        for (auto p : i_o_map_path) {
            std::string i_o_name = p.first;
            i_o_data[i_o_name] = readData(p.second);
        }
    }


    std::vector<std::vector<const char*>> inputNamesModels;
    std::vector<std::vector<const char*>> outputNamesModels;
    std::vector<size_t> numInputNodesModels;
    std::vector<size_t> numOutputNodesModels;
    std::vector<std::vector<std::vector<std::vector<std::tuple<void*, uint64_t, std::vector<int64_t>>>>>> inputTensorsValuesModels;
    inputTensorsValuesModels.resize(numModels);
    //sessionOptionsMaca.SetGraphOptimizationLevel(
    //    GraphOptimizationLevel::ORT_ENABLE_BASIC);

    std::vector<Ort::Session> macaSessions;

    std::vector<std::vector<std::vector<std::vector<Ort::Value>>>> inputTensorsMacaModels;
    inputTensorsMacaModels.resize(numModels);
    std::vector<std::vector<std::vector<std::vector<Ort::Value>>>> outputTensorsMacaModels;
    outputTensorsMacaModels.resize(numModels);

    for (size_t mIdx = 0; mIdx < numModels; mIdx++) {
        size_t numInputNodes;
        size_t numOutputNodes;

        // Maca session
        Ort::SessionOptions sessionOptionsMaca;
        if(maca_custom_op_lib_path){
            Ort::GetApi().RegisterCustomOpsLibrary(sessionOptionsMaca, maca_custom_op_lib_path, &maca_library_handle);
        }

        sessionOptionsMaca.SetIntraOpNumThreads(1);

        OrtMACAProviderOptions maca_options;
        if(device_id_list.empty() == false){
            maca_options.device_id = device_id_list[mIdx % device_id_list.size()];
        }
        sessionOptionsMaca.AppendExecutionProvider_MACA(maca_options);

        Ort::Session sessionMaca(env, modelFiles[mIdx].c_str(), sessionOptionsMaca);

        Ort::AllocatorWithDefaultOptions allocator;
        Ort::MemoryInfo input_memoryInfo = getMemoryInfo(input_loc, maca_options.device_id);


        numInputNodes = sessionMaca.GetInputCount();
        numOutputNodes = sessionMaca.GetOutputCount();

        numInputNodesModels.push_back(numInputNodes);
        numOutputNodesModels.push_back(numOutputNodes);

        std::cout << "Model " << mIdx << " Info: " << std::endl;
        std::unordered_map<std::string, std::vector<int64_t>> input_shape_map;
        std::vector<const char*> inputNames = getInputNames(sessionMaca, allocator);
        std::vector<const char*> outputNames = getOutputNames(sessionMaca, allocator);
        for(auto p : inputNames){
            std::vector<int64_t> input_shape;
            if (cmdOptionExists(argv, argv+argc, "--inputshape_"+ std::string(p))){
                input_shape = splitStringToNumber<int64_t>(getCmdOption(argv, argv+argc, "--inputshape_"+std::string(p)), ',');
                input_shape_map[std::string(p)] = input_shape;
            }
        }
        inputTensorsValuesModels[mIdx].resize(numThreads);
        inputTensorsMacaModels[mIdx].resize(numThreads);
        outputTensorsMacaModels[mIdx].resize(numThreads);
        for (size_t tIdx = 0; tIdx < numThreads; tIdx++) {
            inputTensorsValuesModels[mIdx][tIdx].resize(numInputs);
            inputTensorsMacaModels[mIdx][tIdx].resize(numInputs);
            outputTensorsMacaModels[mIdx][tIdx].resize(numInputs);
            for (size_t idx = 0; idx < numInputs; idx++) {

                if(input_shape_map.empty()){
                    inputTensorsMacaModels[mIdx][tIdx][idx] = creatInputTensors(sessionMaca, input_memoryInfo, batch_size, num_model_io_groups, i_o_data);
                }else{
                    inputTensorsMacaModels[mIdx][tIdx][idx] = creatInputTensors(sessionMaca, input_shape_map, input_memoryInfo, i_o_data);
                }
                if(output_create_pre){
                    outputTensorsMacaModels[mIdx][tIdx][idx] = creatOutputTensors(sessionMaca, input_memoryInfo, batch_size, num_model_io_groups);
                }

                inputTensorsValuesModels[mIdx][tIdx][idx] = getDataFromTensorValues(inputTensorsMacaModels[mIdx][tIdx][idx],input_loc);
                if(save_io && idx == 0 && tIdx==0){
                    saveTensorsToDir(inputNames, inputTensorsMacaModels[mIdx][tIdx][idx],save_io_paths[mIdx],"input");
                }
            }
        }
        macaSessions.push_back(std::move(sessionMaca));
        inputNamesModels.push_back(inputNames);
        outputNamesModels.push_back(outputNames);
        numInputNodesModels.push_back(numInputNodes);
        numOutputNodesModels.push_back(numOutputNodes);
    }

    // Run Maca session in multiple threads mode
    struct SessionRunArgs sessArgsMaca[numModels][numThreads][numInputs];
    for (size_t mIdx = 0; mIdx < numModels; mIdx++) {
        int device_id = 0;
        if(device_id_list.empty() == false){
            device_id = device_id_list[mIdx % device_id_list.size()];
        }
        for (size_t tIdx = 0; tIdx < numThreads; tIdx++) {
            for (size_t idx = 0; idx < numInputs; idx++) {
                sessArgsMaca[mIdx][tIdx][idx].session = &macaSessions[mIdx];
                sessArgsMaca[mIdx][tIdx][idx].inputNames = inputNamesModels[mIdx];
                sessArgsMaca[mIdx][tIdx][idx].inputTensors = inputTensorsMacaModels[mIdx][tIdx][idx].data();
                if(output_create_pre){
                    sessArgsMaca[mIdx][tIdx][idx].outputTensors = outputTensorsMacaModels[mIdx][tIdx][idx].data();
                }
                sessArgsMaca[mIdx][tIdx][idx].numInputNodes = numInputNodesModels[mIdx];
                sessArgsMaca[mIdx][tIdx][idx].outputNames = outputNamesModels[mIdx];
                sessArgsMaca[mIdx][tIdx][idx].numOutputNodes = numOutputNodesModels[mIdx];
                sessArgsMaca[mIdx][tIdx][idx].numInputs = numInputs;
                sessArgsMaca[mIdx][tIdx][idx].use_iobinding = use_iobinding;
                sessArgsMaca[mIdx][tIdx][idx].numTests = numTests;
                sessArgsMaca[mIdx][tIdx][idx].outputLoc= output_loc;
                sessArgsMaca[mIdx][tIdx][idx].paralleTime = &model_parallel_time[mIdx];
                sessArgsMaca[mIdx][tIdx][idx].check_mode = check_mode;
                sessArgsMaca[mIdx][tIdx][idx].device_id = device_id;
                sessArgsMaca[mIdx][tIdx][idx].num_model_io_groups = num_model_io_groups;
            }
        }
    }

    // gain gold output from CpuEP
    if (check_mode!=CHECKMODE::FPS_MODE && check_mode != CHECKMODE::DATA_ACCURACY_COMPARE_MODE) {

        std::vector<Ort::Session> compareSessions;
        std::vector<std::vector<std::vector<std::vector<std::vector<Ort::Value>>>>> inputTensorsCompareModels;
        inputTensorsCompareModels.resize(numModels);
        for (size_t mIdx = 0; mIdx < numModels; mIdx++) {
            Ort::SessionOptions sessionOptions;
            if(check_mode == CHECKMODE::CPU_ACCURACY_COMPARE_MODE){
                if(cpu_custom_op_lib_path){
                    Ort::GetApi().RegisterCustomOpsLibrary(sessionOptions, cpu_custom_op_lib_path, &cpu_library_handle);
                }
            }else{
                OrtMACAProviderOptions maca_options;
                if(device_id_list.empty() == false){
                    maca_options.device_id = device_id_list[mIdx % device_id_list.size()];
                }
                sessionOptions.AppendExecutionProvider_MACA(maca_options);
            }
            sessionOptions.SetIntraOpNumThreads(1);

            Ort::Session compareSession(compare_env, modelCompareFiles[mIdx].c_str(), sessionOptions);
            inputTensorsCompareModels[mIdx].resize(numThreads);
            for (size_t tIdx = 0; tIdx < numThreads; tIdx++) {
                inputTensorsCompareModels[mIdx][tIdx].resize(numInputs);
                for (size_t idx = 0; idx < numInputs; idx++) {
                    inputTensorsCompareModels[mIdx][tIdx][idx].resize(num_model_io_groups);
                    for (size_t k = 0; k < size_t(num_model_io_groups) ; k++){
                    for (size_t i = 0; i < numInputNodesModels[mIdx] ; i++) {
                        Ort::TypeInfo inputTypeInfo = compareSession.GetInputTypeInfo(i);
                        auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
                        ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();

                        inputTensorsCompareModels[mIdx][tIdx][idx][k].emplace_back(
                            Ort::Value::CreateTensor(
                                getMemoryInfo(DATALOC::CPU),
                                std::get<0>(inputTensorsValuesModels[mIdx][tIdx][idx][i+k*numInputNodesModels[mIdx]]),
                                std::get<1>(inputTensorsValuesModels[mIdx][tIdx][idx][i+k*numInputNodesModels[mIdx]]),
                                std::get<2>(inputTensorsValuesModels[mIdx][tIdx][idx][i+k*numInputNodesModels[mIdx]]).data(),
                                std::get<2>(inputTensorsValuesModels[mIdx][tIdx][idx][i+k*numInputNodesModels[mIdx]]).size(),
                                inputType)
                            );
                    }}
                }
            }

            compareSessions.push_back(std::move(compareSession));
        }


        // Run CPU session
        for (size_t mIdx = 0; mIdx < numModels; mIdx++) {
            for (size_t tIdx = 0; tIdx < numThreads; tIdx++) {
                for (size_t idx = 0; idx < numInputs; idx++) {
                    for (size_t b_s = 0; b_s < num_model_io_groups; b_s++){
                        auto outputTensorsCPU = compareSessions[mIdx].Run(Ort::RunOptions{nullptr}, inputNamesModels[mIdx].data(),
                                    inputTensorsCompareModels[mIdx][tIdx][idx][b_s].data(), numInputNodesModels[mIdx], outputNamesModels[mIdx].data(), numOutputNodesModels[mIdx]);
                        if(save_io && idx == 0 && tIdx==0){
                            saveTensorsToDir(sessArgsMaca[mIdx][tIdx][idx].outputNames ,outputTensorsCPU,save_io_paths[mIdx],"output_gold");
                        }
                        // assert(outputTensorsCPU.size() == numOutputNodesModels[mIdx] && outputTensorsCPU.front().IsTensor());
                        for(size_t ndx=0; ndx<numOutputNodesModels[mIdx]; ndx++){
                            std::vector<float> output_data;
                            getOutputDataToFloat(outputTensorsCPU[ndx], output_data);
                            sessArgsMaca[mIdx][tIdx][idx].goldOutput.push_back(output_data);
                        }
                    }
                }
            }
        }
    }

    if(check_mode == CHECKMODE::DATA_ACCURACY_COMPARE_MODE){
        sessArgsMaca[0][0][0].goldOutput = readOutputData(macaSessions[0], i_o_data);
    }
    // execute model with MacaEP
    std::vector<std::thread> threads;
    for (size_t mIdx = 0; mIdx < numModels; mIdx++) {
        for (size_t tIdx = 0; tIdx < numThreads; tIdx++) {
            for (size_t idx = 0; idx < numInputs; idx++) {
                if(tIdx ==0 && idx ==0 ) {
                    auto args = &sessArgsMaca[mIdx][tIdx][0];
                    if(no_warm_up == false){
                        std::vector<const char*> input_names;
                        std::vector<const char*> output_names;
                        for(size_t i=0; i<size_t(args->num_model_io_groups); i++){
                            for(auto p : args->inputNames) input_names.push_back(p);
                            for(auto p : args->outputNames) output_names.push_back(p);
                        }
                        std::cout << "warm up ......" << std::endl;
                        for(size_t w_idx=0; w_idx<10; w_idx++){
                            auto outputs = args->session->Run(Ort::RunOptions{nullptr}, input_names.data(),
                                args->inputTensors, input_names.size(), output_names.data(), output_names.size());
                        }
                        std::cout << "warm up finished!" << std::endl;
                    }
                    if(save_io){
                        auto outputs = args->session->Run(Ort::RunOptions{nullptr}, args->inputNames.data(),
                                args->inputTensors, args->numInputNodes, args->outputNames.data(), args->numOutputNodes);
                        saveTensorsToDir(sessArgsMaca[mIdx][tIdx][idx].outputNames,outputs,save_io_paths[mIdx],"output_maca");
                    }
                }
            }
            threads.push_back(std::move(std::thread(sessionRunMulti, &sessArgsMaca[mIdx][tIdx][0], numInputs)));
        }
    }
    for (auto& thread: threads) {
        thread.join();
    }

    // print result
    if(check_mode != CHECKMODE::FPS_MODE){
    bool successful_flag = true;
     for (size_t mIdx = 0; mIdx < numModels; mIdx++) {
        std::cout << "Accuracy check begin. Model path : " << modelFiles[mIdx] << std::endl;
        for (size_t tIdx = 0; tIdx < numThreads; tIdx++) {
            for (size_t idx = 0; idx < numInputs; idx++) {
                float max_snr = 0;
                float min_snr = INT32_MAX;
                float max_mse = 0;
                float min_mse = INT32_MAX;
                float max_error = 0;
                auto args = &sessArgsMaca[mIdx][tIdx][idx];
                for(size_t adx = 0; adx<args->multiOutputAccuracyInfos.size(); adx++){
                    for(size_t ndx = 0; ndx<args->multiOutputAccuracyInfos[adx].size(); ndx++){
                        AccuracyInfo a_info = args->multiOutputAccuracyInfos[adx][ndx];
                        max_snr = std::max(a_info.snr,max_snr);
                        min_snr = std::min(a_info.snr,min_snr);
                        max_mse = std::max(a_info.mse,max_mse);
                        min_mse = std::min(a_info.mse,min_mse);
                        max_error = std::max(a_info.max_error,max_error);
                    }
                }

                if( std::isnan(max_snr)  || max_snr >= snr_threshold){
                    successful_flag = false;
                    std::cout.setf(std::ios::fixed);
                    std::cout << "Model " << std::setw(2) << std::setfill('0') << std::right << mIdx
                    << " Thread " << std::setw(2) << std::setfill('0') << std::right << tIdx
                    << " Input " << std::setw(6) << std::setfill('0') << std::right << idx
                    << " RunNum " << std::setw(6) << std::setfill('0') << std::right << args->numTests
                    << " min_SNR:" << std::setprecision(6) << min_snr
                    << " max_SNR:" << std::setprecision(6) << max_snr
                    << " min_MSE:" << std::setprecision(6) << min_mse
                    << " max_MSE:" << std::setprecision(6) << max_mse
                    << "  max_AbsoluteDif:" << std::setprecision(6)<< max_error
                    << " <SNRError> "<< std::endl;
                }
            }

        }
        if(successful_flag){
            std::cout << "Accuracy check successful. Model path : " << modelFiles[mIdx] << std::endl;
        }else{
            std::cout << "Accuracy check fail. Model path : " << modelFiles[mIdx] << std::endl;
        }
    }
    }else{
        for(size_t i = 0; i<modelFiles.size(); i++){
            std::cout << "Model path : " << modelFiles[i] << std::endl;
            double min_time = INT64_MAX;
            double max_time = 0;
            double avg_time = 0;
            for(size_t j = 0; j<numThreads; j++){
                for(size_t k = 0; k<numInputs; k++){
                    min_time = std::min(sessArgsMaca[i][j][k].minLantencyTime,min_time) ;
                    max_time = std::max(sessArgsMaca[i][j][k].maxLantencyTime,max_time);
                    avg_time += sessArgsMaca[i][j][k].totalLantencyTime / (numThreads * numInputs * numTests * batch_size * num_model_io_groups );
                }
            }
            double parallel_time = model_parallel_time[i].parallel_end_time - model_parallel_time[i].parallel_begin_time;
            double throughput =  numThreads * numInputs * numTests * batch_size * num_model_io_groups / parallel_time * 1000;
            std::cout << "Inference time record : "
                        << " Min_lantency_time = " << min_time  / (batch_size * num_model_io_groups ) << " ms,"
                        << " Max_lantency_time = " << max_time  / (batch_size * num_model_io_groups ) << " ms,"
                        << " Avg_lantency_time = " << avg_time << " ms,"
                        << " Parallel_time = " <<  parallel_time << " ms,"
                        << " Throughput = " << throughput << " models/s"
                        << std::endl;
        }
         std::cout << "All Model inference time : " << total_end_time - total_begin_time << " ms"<< std::endl;
    }

    // if(maca_library_handle){
    //     dlclose(maca_library_handle);
    // }
}
