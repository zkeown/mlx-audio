cmake_minimum_required(VERSION 3.25)
project(mlx_audio_ext LANGUAGES CXX)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python (required for nanobind)
find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

# Find MLX
find_package(MLX CONFIG REQUIRED)

# Include the csrc subdirectory
add_subdirectory(csrc)
