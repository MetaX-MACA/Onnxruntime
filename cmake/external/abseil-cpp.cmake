# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# 2024 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

include(FetchContent)

# Pass to build
set(ABSL_PROPAGATE_CXX_STD 1)
set(BUILD_TESTING 0)

# if(Patch_FOUND)
#   set(ABSL_PATCH_COMMAND ${Patch_EXECUTABLE} --ignore-whitespace -p1 < ${PROJECT_SOURCE_DIR}/patches/abseil/Fix_Nvidia_Build_Break.patch)
# else()
#   set(ABSL_PATCH_COMMAND git apply --ignore-space-change --ignore-whitespace ${PROJECT_SOURCE_DIR}/patches/abseil/Fix_Nvidia_Build_Break.patch)
# endif()

# FetchContent_Declare(
#     abseil_cpp
#     PREFIX "${CMAKE_CURRENT_BINARY_DIR}/abseil-cpp"
#     BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/external/abseil-cpp"
#     GIT_REPOSITORY  ssh://gerrit.metax-internal.com:29418/external/abseil-cpp.git
#     GIT_TAG lts_2021_11_02
#     PATCH_COMMAND ${ABSL_PATCH_COMMAND} || true
# )

FetchContent_Declare(
  abseil_cpp
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/abseil-cpp
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/external/abseil-cpp
)

FetchContent_MakeAvailable(abseil_cpp)
FetchContent_GetProperties(abseil_cpp SOURCE_DIR)

if (GDK_PLATFORM)
  # Abseil considers any partition that is NOT in the WINAPI_PARTITION_APP a viable platform
  # for Win32 symbolize code (which depends on dbghelp.lib); this logic should really be flipped
  # to only include partitions that are known to support it (e.g. DESKTOP). As a workaround we
  # tell Abseil to pretend we're building an APP.
  target_compile_definitions(absl_symbolize PRIVATE WINAPI_FAMILY=WINAPI_FAMILY_DESKTOP_APP)
endif()
