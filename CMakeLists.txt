cmake_minimum_required(VERSION 3.24)
project(marlin-serve LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MLX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

include(FetchContent)

FetchContent_Declare(mlx
    GIT_REPOSITORY https://github.com/ml-explore/mlx.git
    GIT_TAG        v0.25.2
    GIT_SHALLOW    TRUE
)

FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)

FetchContent_Declare(fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG        10.2.1
    GIT_SHALLOW    TRUE
)

FetchContent_Declare(httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG        v0.15.3
    GIT_SHALLOW    TRUE
)

FetchContent_Declare(catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.2
    GIT_SHALLOW    TRUE
)

FetchContent_Declare(tokenizers_cpp
    GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp.git
    GIT_TAG        acbdc5a27ae01ba74cda756f94da698d40f11dfe
    PATCH_COMMAND  sh -c "sed -i.bak '/fmacro-prefix-map=.*CMAKE_SOURCE_DIR/d' sentencepiece/CMakeLists.txt || true"
)
set(TOKENIZERS_CPP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(mlx json fmt httplib catch2 tokenizers_cpp)

find_package(PkgConfig REQUIRED)
set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:/opt/homebrew/opt/ffmpeg/lib/pkgconfig")
pkg_check_modules(AVFORMAT REQUIRED libavformat)
pkg_check_modules(AVCODEC REQUIRED libavcodec)
pkg_check_modules(SWSCALE REQUIRED libswscale)
pkg_check_modules(AVUTIL REQUIRED libavutil)

set(CORE_SOURCES
    src/core/engine.cpp
    src/core/config.cpp
    src/core/types.cpp
)

set(MODEL_SOURCES
    src/model/marlin.cpp
    src/model/weights.cpp
    src/model/gated_deltanet.cpp
    src/model/attention.cpp
    src/model/streaming_detokenizer.cpp
)

set(SAMPLING_SOURCES
    src/sampling/sampler.cpp
)

set(TOKENIZER_SOURCES
    src/tokenizer/tokenizer.cpp
    src/tokenizer/chat_template.cpp
)

set(VISION_SOURCES
    src/vision/video_decoder.cpp
    src/vision/frame_processor.cpp
)

set(SERVER_SOURCES
    src/server/http_server.cpp
    src/server/openai_protocol.cpp
    src/server/session_manager.cpp
    src/server/sse_writer.cpp
)

set(ALL_LIB_SOURCES
    ${CORE_SOURCES}
    ${MODEL_SOURCES}
    ${SAMPLING_SOURCES}
    ${TOKENIZER_SOURCES}
    ${VISION_SOURCES}
)

add_library(marlin_lib STATIC ${ALL_LIB_SOURCES})
target_include_directories(marlin_lib PUBLIC src)
target_link_libraries(marlin_lib PUBLIC
    mlx
    nlohmann_json::nlohmann_json
    fmt::fmt
    tokenizers_cpp
)
target_include_directories(marlin_lib SYSTEM PUBLIC
    ${AVFORMAT_INCLUDE_DIRS}
    ${AVCODEC_INCLUDE_DIRS}
    ${SWSCALE_INCLUDE_DIRS}
    ${AVUTIL_INCLUDE_DIRS}
)
target_link_directories(marlin_lib PUBLIC
    ${AVFORMAT_LIBRARY_DIRS}
    ${AVCODEC_LIBRARY_DIRS}
    ${SWSCALE_LIBRARY_DIRS}
    ${AVUTIL_LIBRARY_DIRS}
)
target_link_libraries(marlin_lib PUBLIC
    ${AVFORMAT_LIBRARIES}
    ${AVCODEC_LIBRARIES}
    ${SWSCALE_LIBRARIES}
    ${AVUTIL_LIBRARIES}
    "-framework Foundation"
    "-framework Metal"
    "-framework MetalKit"
)

add_executable(marlin-serve src/main_serve.cpp ${SERVER_SOURCES})
target_link_libraries(marlin-serve PRIVATE marlin_lib httplib::httplib)

add_executable(marlin-caption src/main_caption.cpp)
target_link_libraries(marlin-caption PRIVATE marlin_lib)

add_executable(marlin-bench src/bench/bench.cpp)
target_link_libraries(marlin-bench PRIVATE marlin_lib)
