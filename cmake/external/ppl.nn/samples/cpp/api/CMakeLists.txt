if(PPLNN_USE_X86_64 AND PPLNN_ENABLE_ONNX_MODEL)
    file(GLOB PPLNN_SAMPLES_API_SRC *.cc)
    add_executable(pplnn_samples_api_intro ${PPLNN_SAMPLES_API_SRC})
    target_link_libraries(pplnn_samples_api_intro PUBLIC pplnn_static)
    target_compile_features(pplnn_samples_api_intro PRIVATE cxx_std_11)
endif()
if(PPLNN_USE_MACA)
target_compile_definitions(pplnn_samples_api_intro PUBLIC
    PPLNN_USE_MACA)
endif()

if(PPLNN_USE_CUDA)
target_compile_definitions(pplnn_samples_api_intro PUBLIC
    PPLNN_USE_CUDA)
endif()
