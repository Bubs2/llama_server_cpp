set(TEST_TARGET test_mtmd)

include(FetchContent)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.12.0
)
FetchContent_MakeAvailable(json)

add_executable(${TEST_TARGET} main.cpp)
add_custom_command(TARGET test_mtmd POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/image/prac_back.png"
    "$<TARGET_FILE_DIR:test_mtmd>/image/prac_back.png"
)

target_link_libraries(${TEST_TARGET} PRIVATE llama_server nlohmann_json::nlohmann_json)

if(MSVC)
    target_compile_options(${TEST_TARGET} PRIVATE /utf-8)
endif()
