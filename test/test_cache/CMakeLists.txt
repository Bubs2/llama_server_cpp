set(TEST_TARGET test_cache)

include(FetchContent)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.12.0
)
FetchContent_MakeAvailable(json)

add_executable(${TEST_TARGET} main.cpp)

target_link_libraries(${TEST_TARGET} PRIVATE llama_server nlohmann_json::nlohmann_json)

if(MSVC)
    target_compile_options(${TEST_TARGET} PRIVATE /utf-8)
endif()
