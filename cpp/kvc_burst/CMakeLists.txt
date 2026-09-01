cmake_minimum_required(VERSION 3.16)

project(pairec_kvc_burst CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)

set(DATASYSTEM_INCLUDE_DIR "" CACHE PATH "Directory containing datasystem/kv_client.h")
set(DATASYSTEM_LIBRARY "" CACHE FILEPATH "Path to libdatasystem.so")

if(NOT DATASYSTEM_INCLUDE_DIR)
  find_path(
    DATASYSTEM_INCLUDE_DIR
    NAMES datasystem/kv_client.h
    HINTS
      ENV DATASYSTEM_INCLUDE_DIR
      /usr/local/lib/python3.11/site-packages/yr/datasystem/include
      /usr/local/lib64/python3.11/site-packages/yr/datasystem/include
      /usr/local/include
      /opt/datasystem/include
  )
endif()

if(NOT DATASYSTEM_LIBRARY)
  find_library(
    DATASYSTEM_LIBRARY
    NAMES datasystem
    HINTS
      ENV DATASYSTEM_LIBRARY_DIR
      /usr/local/lib/python3.11/site-packages/yr/datasystem/lib
      /usr/local/lib64/python3.11/site-packages/yr/datasystem/lib
      /usr/local/lib
      /usr/local/lib64
      /opt/datasystem/lib
  )
endif()

if(NOT DATASYSTEM_INCLUDE_DIR OR NOT EXISTS "${DATASYSTEM_INCLUDE_DIR}/datasystem/kv_client.h")
  message(FATAL_ERROR "DataSystem headers not found; set DATASYSTEM_INCLUDE_DIR")
endif()
if(NOT DATASYSTEM_LIBRARY OR NOT EXISTS "${DATASYSTEM_LIBRARY}")
  message(FATAL_ERROR "libdatasystem.so not found; set DATASYSTEM_LIBRARY")
endif()

add_executable(kvc_burst_wrapper kvc_burst_wrapper.cpp)
add_executable(kvc_burst_business_probe kvc_burst_business_probe.cpp)
add_executable(kvc_ub_integrity_probe kvc_ub_integrity_probe.cpp)
add_executable(kvc_operation_proxy_test kvc_operation_proxy.cpp kvc_operation_proxy_test.cpp)
add_executable(
  kvc_operation_proxy_rank_test
  kvc_operation_proxy.cpp
  kvc_operation_proxy_rank_test.cpp
)
target_include_directories(kvc_operation_proxy_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(kvc_operation_proxy_test PRIVATE Threads::Threads)
target_include_directories(kvc_operation_proxy_rank_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(kvc_operation_proxy_rank_test PRIVATE Threads::Threads)

foreach(target kvc_burst_wrapper kvc_burst_business_probe kvc_ub_integrity_probe)
  target_include_directories(${target} PRIVATE ${DATASYSTEM_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${target} PRIVATE ${DATASYSTEM_LIBRARY} Threads::Threads)
  get_filename_component(DATASYSTEM_LIBRARY_DIR "${DATASYSTEM_LIBRARY}" DIRECTORY)
  set_property(TARGET ${target} PROPERTY INSTALL_RPATH "$ORIGIN/../lib;${DATASYSTEM_LIBRARY_DIR}")
endforeach()

target_link_libraries(kvc_ub_integrity_probe PRIVATE OpenSSL::Crypto)

install(
  TARGETS kvc_burst_wrapper kvc_burst_business_probe kvc_ub_integrity_probe
  RUNTIME DESTINATION bin
)
