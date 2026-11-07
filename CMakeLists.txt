cmake_minimum_required(VERSION 3.25)
project(bomberman CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# All executables go directly into the build directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Build type default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG   "-g -O0 -fsanitize=address,undefined")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# libtorch (CUDA build: its CMake config calls enable_language(CUDA), so nvcc
# must be locatable even though this project has no .cu sources)
set(TORCH_PATH "$ENV{HOME}/.local/libtorch"
    CACHE PATH "Path to libtorch installation")
if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX}
   AND EXISTS "/usr/local/cuda/bin/nvcc")
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "CUDA compiler")
endif()
set(TORCH_CUDA_ARCH_LIST "12.0" CACHE STRING "CUDA arch list for libtorch")
find_package(Torch REQUIRED PATHS ${TORCH_PATH} NO_DEFAULT_PATH)

include(FetchContent)

# CLI11 for training app argument parsing
FetchContent_Declare(CLI11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG        v2.4.1
)
FetchContent_MakeAvailable(CLI11)

# FTXUI
FetchContent_Declare(ftxui
  GIT_REPOSITORY https://github.com/ArthurSonzogni/FTXUI
  GIT_TAG        v7.0.0
)
FetchContent_MakeAvailable(ftxui)

# Catch2 for unit tests
FetchContent_Declare(Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.7.0
)
FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

enable_testing()

add_subdirectory(src)
add_subdirectory(test)
