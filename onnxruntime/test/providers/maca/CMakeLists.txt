cmake_minimum_required(VERSION 3.10)

project(MacavxTest VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)

if(NOT ONNXRUNTIME_MACA_PATH)
    message(STATUS "You are not set ONNXRUNTIME_MACA_PATH, we will use default path /opt/maca-ai/onnxruntime-maca")
    set(ONNXRUNTIME_MACA_PATH "/opt/maca-ai/onnxruntime-maca")
endif()

if(NOT MACA_DRIVER_PATH)
    message(STATUS "You are not set MACA_DRIVER_PATH, we will use default path /opt/maca")
    set(MACA_DRIVER_PATH "/opt/maca")
endif()


list(APPEND LIBS mxkw mcruntime)
foreach(lib ${LIBS})
    find_library(driver_${lib} ${lib} ${MACA_DRIVER_PATH}/lib)
    list(APPEND DRIVER_LIBS ${driver_${lib}})
    message("--> " ${driver_${lib}})
endforeach()

aux_source_directory(${PROJECT_SOURCE_DIR}/common COMMON_S)
find_library(ONNX_RUNTIME_LIB onnxruntime ${ONNXRUNTIME_MACA_PATH}/lib)

add_executable(maca_test maca_test.cc ${COMMON_S})
target_include_directories(maca_test PRIVATE ${ONNXRUNTIME_MACA_PATH}/include ${PROJECT_SOURCE_DIR}/common ${MACA_DRIVER_PATH}/include)
target_link_libraries(maca_test PRIVATE 
        ${ONNX_RUNTIME_LIB} 
        ${DRIVER_LIBS}
        libstdc++fs.so 
        libpthread.so )



