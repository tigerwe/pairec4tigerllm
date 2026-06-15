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
    ${BRPC_EXTRA_LIBS}
    Threads::Threads
)

install(TARGETS brpc_gateway brpc_recommend_client RUNTIME DESTINATION bin)
