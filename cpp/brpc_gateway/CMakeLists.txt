cmake_minimum_required(VERSION 3.16)

project(pairec_brpc_gateway CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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

add_executable(brpc_http_proxy brpc_http_proxy.cpp)
target_include_directories(brpc_http_proxy PRIVATE ${BRPC_INCLUDE_DIR})
target_link_libraries(
  brpc_http_proxy
  PRIVATE
    pairec_recommend_proto
    ${BRPC_LIBRARY}
    ${ABSL_EXTRA_LIBS}
    ${BRPC_EXTRA_LIBS}
    Threads::Threads
)

install(TARGETS brpc_gateway brpc_recommend_client brpc_http_proxy RUNTIME DESTINATION bin)
