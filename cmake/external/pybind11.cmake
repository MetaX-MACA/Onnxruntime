
if(onnxruntime_PREFER_SYSTEM_LIB)
  find_package(pybind11)
endif()

if(NOT TARGET pybind11::module)
  include(ExternalProject)

  set(pybind11_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/external/pybind11/include)
  set(pybind11_URL ssh://gerrit.metax-internal.com:29418/external/pybind11.git)
  set(pybind11_TAG v2.6.2)

  ExternalProject_Add(pybind11
        PREFIX pybind11
       # GIT_REPOSITORY ${pybind11_URL}
       # GIT_TAG ${pybind11_TAG}
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
  )
  set(pybind11_dep pybind11)
else()
  set(pybind11_lib pybind11::module)
endif()
