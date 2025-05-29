file(GLOB_RECURSE __PPLNN_MODEL_ONNX_SRC__ src/ppl/nn/models/onnx/*.cc)
# if external sources are set, remove `default_register_resources.cc`
if(PPLNN_SOURCE_EXTERNAL_ONNX_MODEL_SOURCES)
    list(REMOVE_ITEM __PPLNN_MODEL_ONNX_SRC__ src/ppl/nn/models/onnx/default_register_resources.cc)
endif()
add_library(pplnn_onnx_static STATIC ${__PPLNN_MODEL_ONNX_SRC__} ${PPLNN_SOURCE_EXTERNAL_ONNX_MODEL_SOURCES})
unset(__PPLNN_MODEL_ONNX_SRC__)

target_compile_definitions(pplnn_onnx_static PUBLIC PPLNN_ENABLE_ONNX_MODEL)
target_link_libraries(pplnn_onnx_static PUBLIC pplnn_basic_static)

if (NOT TARGET protobuf::libprotobuf-lite)
  include(cmake/protobuf.cmake)
  target_include_directories(pplnn_onnx_static PRIVATE ${protobuf_SOURCE_DIR}/src)
endif()
set(PROTOBUF_LITE_LIB "protobuf::libprotobuf-lite")

if(NOT TARGET onnx_proto)
  add_definitions("-DONNX_NAMESPACE=onnx")
  add_definitions("-DONNX_ML=1")
  include(cmake/onnx.cmake)
  target_include_directories(pplnn_onnx_static PUBLIC
                    ${onnx_SOURCE_DIR})
endif()

target_link_libraries(pplnn_onnx_static PUBLIC onnx_proto ${PROTOBUF_LITE_LIB})

target_link_libraries(pplnn_static INTERFACE pplnn_onnx_static)

if(PPLNN_INSTALL)
    install(DIRECTORY include/ppl/nn/models/onnx DESTINATION include/ppl/nn/models)
    install(TARGETS pplnn_onnx_static DESTINATION lib)
endif()
