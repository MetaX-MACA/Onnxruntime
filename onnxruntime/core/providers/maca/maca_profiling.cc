// Copyright 2024 metax-tech.com Inc. All Rights Reserved.
#include "maca_profiling.h"
#include <iostream>
#include <fstream>
using namespace ppl::common;
using namespace ppl::nn;
using namespace std;
namespace onnxruntime{


int parseENV(std::string env_str, int default_value);
static uint32_t g_flag_warmup_iterations = parseENV("MACART_PROFILING_WARM_NUM",1);
static const uint32_t g_flag_run_num = parseENV("MACART_PROFILING_RUN_NUM",1);
static const uint32_t g_update_frequency = parseENV("MACART_UPDATE_PROFILING_FREQUENCY",1);

static void PrintProfilingStatistics(const ProfilingStatistics& stat, double run_dur, int32_t run_count) {
    std::map<std::string, std::pair<double, double>> type_stat;
    std::map<std::string, int> type_count;
    char float_buf_0[128];
    char float_buf_1[128];
    std::cout<< "----- OP statistics by Node -----" << std::endl;
    for (auto x = stat.prof_info.begin(); x != stat.prof_info.end(); ++x) {
        auto ext_type = (x->domain == "" ? "" : x->domain + ".") + x->type;
        double time = (double)x->exec_microseconds / 1000;
        double avg_time = time / x->exec_count;
        if (type_stat.find(ext_type) == type_stat.end()) {
            type_stat[ext_type] = std::make_pair(avg_time, time);
            type_count[ext_type] = 1;
        } else {
            std::pair<double, double>& time_pair = type_stat[ext_type];
            time_pair.first += avg_time;
            time_pair.second += time;
            type_count[ext_type]++;
        }
        sprintf(float_buf_0, "%8.4f", avg_time);
        string temp = x->name;
        temp.insert(temp.length(), temp.length() > 50 ? 0 : 50 - temp.length(), ' ');
        std::cout<< "NAME: [" << temp << "], "
                  << "AVG_TIME: [" << float_buf_0 << "], "
                  << "EXEC_COUNT: [" << x->exec_count << "]"<< std::endl;
    }
    std::cout<< "----- OP statistics by OpType -----"<< std::endl;
    double tot_kernel_time = 0;
    for (auto it = type_stat.begin(); it != type_stat.end(); ++it) {
        tot_kernel_time += it->second.second;
    }
    for (auto it = type_stat.begin(); it != type_stat.end(); ++it) {
        sprintf(float_buf_0, "%8.4f", it->second.first);
        sprintf(float_buf_1, "%8.4f", it->second.second / tot_kernel_time * 100);
        string temp = it->first;
        temp.insert(temp.length(), temp.length() > 20 ? 0 : 20 - temp.length(), ' ');
        std::cout<< "TYPE: [" << temp << "], AVG_TIME: [" << float_buf_0 << "], Percentage: [" << float_buf_1
                  << "], excute times [" << type_count[it->first] << "]"<< std::endl;
    }

    std::cout<< "----- TOTAL statistics -----"<< std::endl;
    sprintf(float_buf_0, "%8.4f", tot_kernel_time / run_count);
    sprintf(float_buf_1, "%8.4f", run_dur / run_count);
    std::cout<< "RUN_COUNT: [" << run_count << "]"<< std::endl;
    std::cout<< "AVG_KERNEL_TIME: [" << float_buf_0 << "]"<< std::endl;
    std::cout<< "AVG_RUN_TIME: [" << float_buf_1 << "]"<< std::endl;
    sprintf(float_buf_0, "%8.4f", tot_kernel_time);
    sprintf(float_buf_1, "%8.4f", run_dur);
    std::cout<< "TOT_KERNEL_TIME: [" << float_buf_0 << "]"<< std::endl;
    std::cout<< "TOT_RUN_TIME: [" << float_buf_1 << "]"<< std::endl;
    sprintf(float_buf_0, "%8.4f%%", (run_dur - tot_kernel_time) / run_dur * 100);
    std::cout<< "SCHED_LOST: [" << float_buf_0 << "]"<< std::endl;
}

bool macaProfiling(std::shared_ptr<ppl::nn::Runtime> runtime, double* run_dur, uint32_t* run_count) {
    if (g_flag_warmup_iterations > 0) {
        std::cout<< "Warm up start for " << g_flag_warmup_iterations << " times."<< std::endl;
        for (uint32_t i = 0; i < g_flag_warmup_iterations; ++i) {
            runtime->Run();
        }
        std::cout<< "Warm up end."<< std::endl;
        g_flag_warmup_iterations = 0;
    }
    if((*run_count) == 0){
        auto status = runtime->Configure(RUNTIME_CONF_SET_KERNEL_PROFILING_FLAG, true);
        if (status != RC_SUCCESS) {
            LOGS_DEFAULT(WARNING) << "enable profiling failed: " << GetRetCodeStr(status);
        }
    }

    std::cout<< "Profiling start"<< std::endl;

    for(uint32_t i=0; i< g_flag_run_num; i++){
        auto run_begin_ts = std::chrono::system_clock::now();
        runtime->Run();
        auto run_end_ts = std::chrono::system_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::microseconds>(run_end_ts - run_begin_ts);
        (*run_dur) += (double)diff.count() / 1000;
        (*run_count) += 1;
    }



    std::cout<< "Total duration: " << *run_dur << " ms"<< std::endl;


    ProfilingStatistics stat;
    auto status = runtime->GetProfilingStatistics(&stat);
    if (status != RC_SUCCESS) {
        LOGS_DEFAULT(WARNING) << "Get profiling statistics failed: " << GetRetCodeStr(status);
    }
    PrintProfilingStatistics(stat, *run_dur, *run_count);

    std::cout<< "Profiling End"<< std::endl;
    if((g_update_frequency!=0) && (g_update_frequency <= ((*run_count) / g_flag_run_num))){
        auto status = runtime->Configure(RUNTIME_CONF_SET_KERNEL_PROFILING_FLAG, false);
            if (status != RC_SUCCESS) {
                LOGS_DEFAULT(WARNING) << "Stop profiling failed: " << GetRetCodeStr(status);
        }
        (*run_count) = 0;
        (*run_dur) = 0;
    }

    return true;
}
}
