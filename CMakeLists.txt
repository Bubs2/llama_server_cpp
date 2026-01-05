cmake_minimum_required(VERSION 3.24)
project(llama_server VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)

option(LLAMA_SERVER_BUILD_TESTS "Build tests" OFF)

set(LLAMA_BUILD_TOOLS    ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "" FORCE)

if (NOT DEFINED LLAMA_BUILD_EXAMPLES)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
endif()
if (NOT DEFINED LLAMA_BUILD_SERVER)
    set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
endif()
if (NOT DEFINED LLAMA_BUILD_TESTS)
    set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
endif()
if (NOT DEFINED LLAMA_CURL)
    set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)
endif()
if (NOT DEFINED BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)
endif()

if (NOT DEFINED GGML_CUDA)
    set(GGML_CUDA            ON  CACHE BOOL "" FORCE)
endif()

include(FetchContent)
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.16.0
)
FetchContent_MakeAvailable(spdlog)

include(FetchContent)
FetchContent_Declare(
    re2
    GIT_REPOSITORY https://github.com/google/re2.git
    GIT_TAG 2025-11-05
)
FetchContent_MakeAvailable(re2)

add_subdirectory(third_party/llama.cpp)

if(LLAMA_SERVER_BUILD_TESTS)
    add_subdirectory(test/test_generate)
    add_subdirectory(test/test_mtmd)
    add_subdirectory(test/test_cache)
endif()

set(LIB_SOURCES
    "src/utils/utils.cpp"
    "src/utils/llama_strategy.cpp"

    "src/llama_wrapper/llama_model.cpp"
    "src/llama_wrapper/llama_context.cpp"

    "src/session_component/tokenizer.cpp"
    "src/session_component/templater.cpp"

    "src/session_component/input_encoder.cpp"
    "src/session_component/kv_scheduler.cpp"

    "src/session_component/sampler.cpp"
    "src/session_component/streamer.cpp"
    
    "src/llama_session.cpp"
    "src/model_server.cpp"
)

set(LIB_HEADERS
    "include/llama_exception.h"
    "include/llama_configs.h"
    "include/llama_inputs.h"
    "include/llama_strategy.h"
    "include/model_server.h"
    "include/llama_session.h"

    "src/internal/utils.h"
    "src/internal/llama_log.h"
    "src/internal/id_chunk.h"

    "src/internal/llama_model.h"
    "src/internal/llama_context.h"

    "src/internal/tokenizer.h"
    "src/internal/templater.h"

    "src/internal/input_encoder.h"
    "src/internal/kv_scheduler.h"

    "src/internal/sampler.h"
    "src/internal/streamer.h"
)

add_library(${PROJECT_NAME}
    ${LIB_SOURCES}
    ${LIB_HEADERS}
)

if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /utf-8)
    target_compile_options(llama PRIVATE /Zc:char8_t-) #兼容char8_t、char隐式转换行为
endif()

target_include_directories(${PROJECT_NAME}
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include> 

    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/internal
    "third_party/llama.cpp"
    "third_party/llama.cpp/common"
    "third_party/llama.cpp/tools/mtmd"
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
    llama
    common
    mtmd
    spdlog::spdlog
    re2::re2
)