cmake_minimum_required(VERSION 3.24)

project(llm_infer_reduce LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures")

add_library(reduce_options INTERFACE)
target_include_directories(reduce_options INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)
target_compile_options(reduce_options INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

file(GLOB REDUCE_CUDA_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/*/*.cu
)

if(NOT REDUCE_CUDA_SOURCES)
    message(STATUS "No .cu files found yet. Add sum/reduce_sum.cu, then rebuild.")
endif()

foreach(source_file IN LISTS REDUCE_CUDA_SOURCES)
    file(RELATIVE_PATH relative_source ${CMAKE_CURRENT_SOURCE_DIR} ${source_file})
    file(READ ${source_file} source_contents)

    if(NOT source_contents MATCHES "(^|[ \t\r\n])int[ \t\r\n]+main[ \t\r\n]*\\(")
        message(STATUS "Skipping ${relative_source}: no main() yet.")
        continue()
    endif()

    get_filename_component(source_dir ${relative_source} DIRECTORY)
    get_filename_component(source_name ${source_file} NAME_WE)
    string(REPLACE "/" "_" source_dir_name "${source_dir}")

    if(source_dir_name STREQUAL "")
        set(target_name ${source_name})
    else()
        set(target_name ${source_dir_name}_${source_name})
    endif()

    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} PRIVATE reduce_options)
    set_target_properties(${target_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_RUNTIME_LIBRARY Shared
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endforeach()
