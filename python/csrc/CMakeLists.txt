# Core C++ library
add_library(mlx_audio_ext)

target_sources(
    mlx_audio_ext
    PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/primitives/overlap_add.cpp
    ${CMAKE_CURRENT_LIST_DIR}/primitives/frame_signal.cpp
    ${CMAKE_CURRENT_LIST_DIR}/primitives/pad_signal.cpp
    ${CMAKE_CURRENT_LIST_DIR}/primitives/windows.cpp
    ${CMAKE_CURRENT_LIST_DIR}/primitives/mel_filterbank.cpp
    ${CMAKE_CURRENT_LIST_DIR}/primitives/autocorrelation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/primitives/resample.cpp
    ${CMAKE_CURRENT_LIST_DIR}/primitives/dct.cpp
    ${CMAKE_CURRENT_LIST_DIR}/primitives/spectral.cpp
    ${CMAKE_CURRENT_LIST_DIR}/primitives/spectral_contrast.cpp
)

target_include_directories(
    mlx_audio_ext PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(mlx_audio_ext PUBLIC mlx)

# Enable warnings for better code quality
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
    target_compile_options(mlx_audio_ext PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter  # MLX callbacks often have unused params
    )
endif()

# Metal library compilation
if(MLX_BUILD_METAL)
    mlx_build_metallib(
        TARGET mlx_audio_ext_metallib
        TITLE mlx_audio_ext
        SOURCES
            ${CMAKE_CURRENT_LIST_DIR}/metal/overlap_add.metal
            ${CMAKE_CURRENT_LIST_DIR}/metal/frame_signal.metal
            ${CMAKE_CURRENT_LIST_DIR}/metal/pad_signal.metal
            ${CMAKE_CURRENT_LIST_DIR}/metal/windows.metal
            ${CMAKE_CURRENT_LIST_DIR}/metal/mel_filterbank.metal
            ${CMAKE_CURRENT_LIST_DIR}/metal/spectral_contrast.metal
        INCLUDE_DIRS
            ${PROJECT_SOURCE_DIR}
            ${MLX_INCLUDE_DIRS}
        OUTPUT_DIRECTORY
            ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    )
    add_dependencies(mlx_audio_ext mlx_audio_ext_metallib)
endif()

# Find nanobind
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT
    RESULT_VARIABLE NANOBIND_RESULT
    ERROR_VARIABLE NANOBIND_ERROR
)
if(NOT NANOBIND_RESULT EQUAL 0)
    message(FATAL_ERROR
        "Failed to find nanobind cmake directory. Is nanobind installed?\n"
        "Install with: pip install nanobind>=2.0.0\n"
        "Error: ${NANOBIND_ERROR}"
    )
endif()
find_package(nanobind CONFIG REQUIRED)

# Python bindings with nanobind
# Must match MLX's nanobind configuration exactly for type caster sharing:
# - STABLE_ABI: MLX uses this
# - NB_DOMAIN mlx: Enables cross-module type visibility
nanobind_add_module(
    _ext
    NB_STATIC
    STABLE_ABI
    LTO
    NOMINSIZE
    NB_DOMAIN
    mlx
    ${CMAKE_CURRENT_LIST_DIR}/bindings.cpp
)

target_link_libraries(_ext PRIVATE mlx_audio_ext)

if(BUILD_SHARED_LIBS)
    target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path)
endif()

# Install the extension module to the package directory
install(TARGETS _ext LIBRARY DESTINATION mlx_audio/primitives)
