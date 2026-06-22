cmake_minimum_required(VERSION 3.16)

project(pairec_brpc_gateway CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(PAIREC_ENABLE_TRTLLM_CPP "Build brpc_inference_server with TensorRT-LLM C++ Executor backend" OFF)

find_package(Protobuf REQUIRED)
find_package(Threads REQUIRED)
find_package(PkgConfig QUIET)

if(PkgConfig_FOUND)
  pkg_check_modules(BRPC_PC QUIET brpc)
endif()

find_path(
  BRPC_INCLUDE_DIR
  NAMES brpc/server.h
  HINTS ${BRPC_PC_INCLUDE_DIRS}
)

find_library(
  BRPC_LIBRARY
  NAMES brpc
  HINTS ${BRPC_PC_LIBRARY_DIRS}
)

if(NOT BRPC_INCLUDE_DIR OR NOT BRPC_LIBRARY)
  message(FATAL_ERROR
    "brpc SDK was not found. Set BRPC_INCLUDE_DIR and BRPC_LIBRARY, "
    "or build inside an image that already contains Apache bRPC.")
endif()

# Some ARM/openEuler images ship protobuf builds that depend on Abseil shared
# libraries without exporting every transitive dependency through
# protobuf::libprotobuf. Link the common Abseil runtime libraries explicitly
# when they are present. The first entry fixes the observed
# libabsl_log_internal_message DSO-missing linker failure.
set(ABSL_OPTIONAL_LIB_NAMES
  absl_log_internal_message
  absl_log_internal_check_op
  absl_log_internal_conditions
  absl_log_internal_format
  absl_log_internal_globals
  absl_log_internal_log_sink_set
  absl_log_internal_nullguard
  absl_log_internal_proto
  absl_log_sink
  absl_log_entry
  absl_log_severity
  absl_raw_logging_internal
  absl_base
  absl_spinlock_wait
  absl_strings
  absl_throw_delegate
  absl_status
  absl_statusor
  absl_symbolize
  absl_stacktrace
  absl_examine_stack
  absl_malloc_internal
  absl_time
  absl_time_zone
  absl_civil_time
  absl_int128
  absl_str_format_internal
)

set(ABSL_EXTRA_LIBS)
foreach(ABSL_LIB_NAME IN LISTS ABSL_OPTIONAL_LIB_NAMES)
  string(MAKE_C_IDENTIFIER "${ABSL_LIB_NAME}" ABSL_LIB_VAR)
  string(TOUPPER "${ABSL_LIB_VAR}" ABSL_LIB_VAR)
  find_library("${ABSL_LIB_VAR}_LIBRARY" NAMES "${ABSL_LIB_NAME}")
  set(ABSL_LIB_PATH "${${ABSL_LIB_VAR}_LIBRARY}")
  if(ABSL_LIB_PATH)
    list(APPEND ABSL_EXTRA_LIBS "${ABSL_LIB_PATH}")
  endif()
endforeach()

if(ABSL_EXTRA_LIBS)
  message(STATUS "Linking optional Abseil libraries: ${ABSL_EXTRA_LIBS}")
endif()

if(PAIREC_ENABLE_TRTLLM_CPP)
  set(
    TRTLLM_INCLUDE_DIR
    ""
    CACHE PATH
    "TensorRT-LLM C++ include directory, usually /TensorRT-LLM/cpp/include or /home/TensorRT-LLM/cpp/include"
  )
  set(
    TRTLLM_LIBRARY
    ""
    CACHE FILEPATH
    "Path to libtensorrt_llm.so"
  )
  set(
    TRTLLM_EXTRA_LIBS
    ""
    CACHE STRING
    "Extra TensorRT-LLM runtime libraries, separated by semicolons"
  )
  set(
    TRTLLM_CUDA_INCLUDE_DIR
    ""
    CACHE PATH
    "CUDA include directory that contains cuda_fp16.h"
  )
  set(
    CUDA_DRIVER_LIBRARY
    ""
    CACHE FILEPATH
    "Path to libcuda.so or CUDA driver stub library"
  )

  set(PAIREC_TRTLLM_INCLUDE_DIR "${TRTLLM_INCLUDE_DIR}")
  set(PAIREC_TRTLLM_LIBRARY "${TRTLLM_LIBRARY}")
  set(PAIREC_TRTLLM_CUDA_INCLUDE_DIR "${TRTLLM_CUDA_INCLUDE_DIR}")
  set(PAIREC_CUDA_DRIVER_LIBRARY "${CUDA_DRIVER_LIBRARY}")

  if(NOT PAIREC_TRTLLM_INCLUDE_DIR)
    find_path(
      PAIREC_TRTLLM_INCLUDE_DIR
      NAMES tensorrt_llm/executor/executor.h
      HINTS
        ENV TRTLLM_INCLUDE_DIR
        /TensorRT-LLM/cpp/include
        /home/TensorRT-LLM/cpp/include
        /home/zcx/TensorRT-LLM/cpp/include
        /usr/local/include
    )
  endif()

  if(NOT PAIREC_TRTLLM_LIBRARY)
    find_library(
      PAIREC_TRTLLM_LIBRARY
      NAMES tensorrt_llm
      HINTS
        ENV TRTLLM_LIBRARY_DIR
        /TensorRT-LLM/cpp/build/tensorrt_llm
        /home/TensorRT-LLM/cpp/build/tensorrt_llm
        /home/zcx/TensorRT-LLM/cpp/build/tensorrt_llm
        /usr/local/lib
        /usr/local/lib64
        /usr/lib64
    )
  endif()

  if(NOT PAIREC_TRTLLM_CUDA_INCLUDE_DIR)
    find_path(
      PAIREC_TRTLLM_CUDA_INCLUDE_DIR
      NAMES cuda_fp16.h
      HINTS
        ENV CUDA_HOME
        /usr/local/cuda
        /usr/local/cuda-12
        /usr/local/cuda-12.8
      PATH_SUFFIXES include
    )
  endif()

  if(NOT PAIREC_CUDA_DRIVER_LIBRARY)
    find_library(
      PAIREC_CUDA_DRIVER_LIBRARY
      NAMES cuda
      HINTS
        ENV CUDA_DRIVER_LIBRARY_DIR
        /usr/local/cuda/lib64/stubs
        /usr/local/cuda/targets/aarch64-linux/lib/stubs
        /usr/local/cuda/targets/x86_64-linux/lib/stubs
        /usr/local/cuda/lib64
        /usr/local/nvidia/lib64
        /usr/lib64
    )
  endif()

  if(NOT PAIREC_TRTLLM_INCLUDE_DIR OR NOT EXISTS "${PAIREC_TRTLLM_INCLUDE_DIR}/tensorrt_llm/executor/executor.h")
    message(FATAL_ERROR
      "TensorRT-LLM C++ headers were not found. Set TRTLLM_INCLUDE_DIR.")
  endif()
  if(NOT PAIREC_TRTLLM_LIBRARY OR NOT EXISTS "${PAIREC_TRTLLM_LIBRARY}")
    message(FATAL_ERROR
      "libtensorrt_llm.so was not found. Set TRTLLM_LIBRARY.")
  endif()
  if(NOT PAIREC_TRTLLM_CUDA_INCLUDE_DIR OR NOT EXISTS "${PAIREC_TRTLLM_CUDA_INCLUDE_DIR}/cuda_fp16.h")
    message(FATAL_ERROR
      "CUDA headers were not found. Set TRTLLM_CUDA_INCLUDE_DIR.")
  endif()
  if(NOT PAIREC_CUDA_DRIVER_LIBRARY OR NOT EXISTS "${PAIREC_CUDA_DRIVER_LIBRARY}")
    message(FATAL_ERROR
      "CUDA driver library was not found. Set CUDA_DRIVER_LIBRARY to libcuda.so or the CUDA stub library.")
  endif()

  message(STATUS "TensorRT-LLM C++ backend enabled")
  message(STATUS "  TRTLLM_INCLUDE_DIR=${PAIREC_TRTLLM_INCLUDE_DIR}")
  message(STATUS "  TRTLLM_LIBRARY=${PAIREC_TRTLLM_LIBRARY}")
  message(STATUS "  TRTLLM_CUDA_INCLUDE_DIR=${PAIREC_TRTLLM_CUDA_INCLUDE_DIR}")
  message(STATUS "  CUDA_DRIVER_LIBRARY=${PAIREC_CUDA_DRIVER_LIBRARY}")
endif()

set(PROTO_FILE ${CMAKE_CURRENT_SOURCE_DIR}/../../proto/recommend.proto)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILE})

add_library(pairec_recommend_proto ${PROTO_SRCS} ${PROTO_HDRS})
target_include_directories(
  pairec_recommend_proto
  PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
    ${Protobuf_INCLUDE_DIRS}
)
target_link_libraries(pairec_recommend_proto PUBLIC protobuf::libprotobuf)

set(
  BRPC_EXTRA_LIBS
  "gflags;ssl;crypto;z;dl;rt"
  CACHE STRING
  "Extra libraries required by the local brpc build, separated by semicolons"
)

add_executable(brpc_gateway recommend_gateway.cpp)
target_include_directories(brpc_gateway PRIVATE ${BRPC_INCLUDE_DIR})
target_link_libraries(
  brpc_gateway
  PRIVATE
    pairec_recommend_proto
    ${BRPC_LIBRARY}
    ${ABSL_EXTRA_LIBS}
    ${BRPC_EXTRA_LIBS}
    Threads::Threads
)

add_executable(brpc_recommend_client recommend_client.cpp)
target_include_directories(brpc_recommend_client PRIVATE ${BRPC_INCLUDE_DIR})
target_link_libraries(
  brpc_recommend_client
  PRIVATE
    pairec_recommend_proto
    ${BRPC_LIBRARY}
    ${ABSL_EXTRA_LIBS}
    ${BRPC_EXTRA_LIBS}
    Threads::Threads
)

add_executable(brpc_inference_server brpc_inference_server.cpp)
target_include_directories(brpc_inference_server PRIVATE ${BRPC_INCLUDE_DIR})
target_link_libraries(
  brpc_inference_server
  PRIVATE
    pairec_recommend_proto
    ${BRPC_LIBRARY}
    ${ABSL_EXTRA_LIBS}
    ${BRPC_EXTRA_LIBS}
    Threads::Threads
)

if(PAIREC_ENABLE_TRTLLM_CPP)
  get_filename_component(PAIREC_TRTLLM_LIBRARY_DIR "${PAIREC_TRTLLM_LIBRARY}" DIRECTORY)
  target_compile_definitions(brpc_inference_server PRIVATE PAIREC_ENABLE_TRTLLM_CPP=1)
  target_include_directories(
    brpc_inference_server
    PRIVATE
      ${PAIREC_TRTLLM_INCLUDE_DIR}
      ${PAIREC_TRTLLM_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(
    brpc_inference_server
    PRIVATE
      ${PAIREC_TRTLLM_LIBRARY}
      ${PAIREC_CUDA_DRIVER_LIBRARY}
      ${TRTLLM_EXTRA_LIBS}
  )
  set_property(
    TARGET brpc_inference_server
    PROPERTY INSTALL_RPATH
      "$ORIGIN/../lib;${PAIREC_TRTLLM_LIBRARY_DIR}"
  )
endif()

install(TARGETS brpc_gateway brpc_recommend_client brpc_inference_server RUNTIME DESTINATION bin)
