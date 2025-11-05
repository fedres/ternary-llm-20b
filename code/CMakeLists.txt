cmake_minimum_required(VERSION 3.20)

# ============================================================================
# PROJECT SETUP AND VERSION MANAGEMENT
# ============================================================================
project(
    TernaryLLM-20B
    VERSION 1.0.0
    DESCRIPTION "Ultra-efficient 20B parameter ternary language model"
    LANGUAGES CXX
)

# Set C++20 standard requirement
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# BUILD CONFIGURATION
# ============================================================================
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ============================================================================
# COMPILER FLAGS AND OPTIMIZATION
# ============================================================================
# Base optimization flags
set(CMAKE_CXX_FLAGS_INIT "")
set(CMAKE_CXX_FLAGS_DEBUG_INIT "-g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO_INIT "-O3 -g -DNDEBUG")
set(CMAKE_CXX_FLAGS_MINSIZEREL_INIT "-Os -DNDEBUG")

# Architecture-specific optimizations
set(ARCH_FLAGS "")
if(USE_NVCPP)
    # NVIDIA nvc++ compiler with GPU parallelization
    list(APPEND ARCH_FLAGS "-stdpar=multicore" "-acc=gpu")
    message(STATUS "Using NVIDIA nvc++ with GPU parallelization (-stdpar=multicore)")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Enable AVX-512F for maximum performance on supported hardware
    list(APPEND ARCH_FLAGS "-march=native" "-mavx512f")
    
    # General optimization flags for Release builds
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -ffast-math -funroll-loops")
    
    # Additional optimization flags
    list(APPEND CMAKE_CXX_FLAGS_RELEASE "-fno-exceptions" "-fno-rtti")
    list(APPEND CMAKE_CXX_FLAGS_RELEASE "-fvectorize" "-ftree-vectorize")
    list(APPEND CMAKE_CXX_FLAGS_RELEASE "-flto" "-fmerge-constants")
    
    # Debug flags
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fsanitize=undefined")
endif()

# Link-time optimization for Release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Compiler version: ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Architecture flags: ${ARCH_FLAGS}")

# ============================================================================
# DEPENDENCY DETECTION
# ============================================================================
# Find OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_VERSION}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# pthread is usually part of the standard library, but we'll try to find it explicitly
find_package(Threads REQUIRED)
set(LIBRARIES Threads::Threads)

# Math libraries (m, dl, pthread, etc.)
set(MATH_LIBRARIES m)
if(UNIX AND NOT APPLE)
    list(APPEND MATH_LIBRARIES dl)
endif()

# Optional: Find MKL for Intel Math Kernel Library (high-performance BLAS)
find_package(MKL QUIET)
if(MKL_FOUND)
    message(STATUS "Intel MKL found - using for optimized math operations")
    list(APPEND LIBRARIES ${MKL_LIBRARIES})
else()
    # Fallback to standard BLAS/LAPACK
    find_package(BLAS QUIET)
    find_package(LAPACK QUIET)
    if(BLAS_FOUND AND LAPACK_FOUND)
        message(STATUS "BLAS/LAPACK found - using standard implementations")
        list(APPEND LIBRARIES ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
    endif()
endif()

# Optional: Find Eigen3 for linear algebra operations
find_package(Eigen3 QUIET)
if(Eigen3_FOUND)
    message(STATUS "Eigen3 found - using for linear algebra")
    set(HAVE_EIGEN TRUE)
endif()

# Optional: Find SIMDjson for high-performance JSON parsing
find_package(simdjson QUIET)
if(simdjson_FOUND)
    message(STATUS "SIMDjson found - using for JSON operations")
    set(HAVE_SIMDJSON TRUE)
endif()

# ============================================================================
# HEADER FILES CONFIGURATION
# ============================================================================
# Collect all header files
file(GLOB_RECURSE HEADER_FILES
    "include/*.hpp"
    "include/*.h"
)

# Set up include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ============================================================================
# LIBRARY TARGET CREATION
# ============================================================================
# Main ternary computation library
add_library(ternary_core STATIC ${HEADER_FILES})
target_compile_definitions(ternary_core PRIVATE TERNARY_CORE_BUILD)
target_compile_features(ternary_core PUBLIC cxx_std_20)
target_link_libraries(ternary_core ${LIBRARIES} ${MATH_LIBRARIES})

# Apply architecture-specific flags
if(ARCH_FLAGS)
    target_compile_options(ternary_core PRIVATE ${ARCH_FLAGS})
endif()

# Set library properties
set_target_properties(ternary_core PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    PUBLIC_HEADER "${HEADER_FILES}"
)

# Install library headers and library
install(TARGETS ternary_core
    EXPORT TernaryLLMTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    PUBLIC_HEADER DESTINATION include
    INCLUDES DESTINATION include
)

# ============================================================================
# EXECUTABLE TARGETS
# ============================================================================
# Main application executable
set(MAIN_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../user_input_files/run.cpp
)

add_executable(ternary_llm ${MAIN_SOURCES})
target_link_libraries(ternary_llm ternary_core)
target_compile_definitions(ternary_llm PRIVATE TERNARY_LLM_MAIN)

# Apply architecture-specific flags
if(ARCH_FLAGS)
    target_compile_options(ternary_llm PRIVATE ${ARCH_FLAGS})
endif()

# Examples directory
option(BUILD_EXAMPLES "Build example programs" ON)
if(BUILD_EXAMPLES)
    file(GLOB_RECURSE EXAMPLE_FILES "examples/*.cpp")
    foreach(EXAMPLE_FILE ${EXAMPLE_FILES})
        get_filename_component(EXAMPLE_NAME ${EXAMPLE_FILE} NAME_WE)
        add_executable(${EXAMPLE_NAME} ${EXAMPLE_FILE})
        target_link_libraries(${EXAMPLE_NAME} ternary_core)
        install(TARGETS ${EXAMPLE_NAME} DESTINATION bin/examples)
    endforeach()
endif()

# ============================================================================
# TESTING FRAMEWORK INTEGRATION
# ============================================================================
option(BUILD_TESTS "Build unit tests" ON)

if(BUILD_TESTS)
    # Find Google Test
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        # Download and build Google Test if not found
        include(FetchContent)
        FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
        )
        FetchContent_MakeAvailable(googletest)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        include(GoogleTest)
    endif()

    # Test sources
    file(GLOB_RECURSE TEST_FILES "tests/*.cpp")
    
    if(TEST_FILES)
        enable_testing()
        
        # Create test executable
        add_executable(ternary_tests ${TEST_FILES})
        target_link_libraries(ternary_tests 
            ternary_core 
            GTest::gtest 
            GTest::gtest_main
        )
        
        # Add custom test command
        add_test(NAME ternary_core_tests COMMAND ternary_tests)
        
        # Install test executable
        install(TARGETS ternary_tests DESTINATION bin/tests)
    endif()
endif()

# ============================================================================
# DOCUMENTATION GENERATION SETUP
# ============================================================================
option(BUILD_DOCS "Build documentation with Doxygen" OFF)

if(BUILD_DOCS)
    # Find Doxygen
    find_package(Doxygen QUIET)
    if(DOXYGEN_FOUND)
        # Set Doxyfile options
        set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
        set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
        
        # Create Doxyfile from template
        configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
        
        # Add documentation build target
        add_custom_target(docs
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM
        )
        
        # Install documentation
        install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs/html/
            DESTINATION share/doc/ternary-llm
            FILES_MATCHING PATTERN "*.html" PATTERN "*.css" PATTERN "*.js"
        )
    else()
        message(WARNING "Doxygen not found - documentation will not be built")
    endif()
endif()

# ============================================================================
# BENCHMARKING
# ============================================================================
option(BUILD_BENCHMARKS "Build benchmarking executables" OFF)

if(BUILD_BENCHMARKS)
    # Find Google Benchmark
    find_package(benchmark QUIET)
    if(NOT benchmark_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            googlebenchmark
            URL https://github.com/google/benchmark/archive/refs/tags/v1.8.3.tar.gz
        )
        FetchContent_MakeAvailable(googlebenchmark)
    endif()
    
    # Benchmark sources
    file(GLOB_RECURSE BENCHMARK_FILES "benchmarks/*.cpp")
    
    if(BENCHMARK_FILES)
        # Create benchmark executable
        add_executable(ternary_benchmarks ${BENCHMARK_FILES})
        target_link_libraries(ternary_benchmarks 
            ternary_core 
            benchmark::benchmark
        )
        
        # Install benchmark executable
        install(TARGETS ternary_benchmarks DESTINATION bin/benchmarks)
    endif()
endif()

# ============================================================================
# INSTALLATION RULES
# ============================================================================
# Install CMake package configuration files
include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "TernaryLLMConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Create export set
install(EXPORT TernaryLLMTargets
    FILE TernaryLLMTargets.cmake
    NAMESPACE TernaryLLM::
    DESTINATION lib/cmake/TernaryLLM
)

# Install configuration files
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/TernaryLLMConfigVersion.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/TernaryLLMConfig.cmake"
    DESTINATION lib/cmake/TernaryLLM
)

# Install pkg-config file for external projects
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ternary-llm.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/ternary-llm.pc"
    @ONLY
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/ternary-llm.pc"
    DESTINATION lib/pkgconfig
)

# ============================================================================
# PROFILING AND DEBUGGING SUPPORT
# ============================================================================
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Add profiling flags for Debug builds
    target_compile_definitions(ternary_core PRIVATE TERNARY_PROFILE)
    
    # Find common profiling tools
    find_package(Valgrind QUIET)
    if(Valgrind_FOUND)
        message(STATUS "Valgrind found - memory debugging available")
    endif()
endif()

# ============================================================================
# CODE COVERAGE (GCC/Clang only)
# ============================================================================
option(ENABLE_COVERAGE "Enable code coverage reporting" OFF)

if(ENABLE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Enable coverage flags
    set(CMAKE_CXX_FLAGS_COVERAGE "${CMAKE_CXX_FLAGS_RELEASE} --coverage -fprofile-arcs -ftest-coverage")
    set(CMAKE_EXE_LINKER_FLAGS_COVERAGE "--coverage")
    
    # Add coverage target
    add_custom_target(coverage
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target ternary_tests
        COMMAND gcov ternary_core.cpp
        COMMAND gcovr --html --output coverage.html .
        COMMENT "Generating code coverage report"
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()

# ============================================================================
# EXPORT COMPILATION DATABASE (for clangd, etc.)
# ============================================================================
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# SUMMARY AND FINAL STATUS
# ============================================================================
message(STATUS "==========================================================================")
message(STATUS "TernaryLLM-20B Configuration Summary")
message(STATUS "==========================================================================")
message(STATUS "Version:                ${PROJECT_VERSION}")
message(STATUS "C++ Standard:           ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type:             ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler:               ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Install Prefix:         ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
message(STATUS "Dependencies:")
message(STATUS "  OpenMP:                ${OpenMP_FOUND}")
message(STATUS "  Threads:               Threads::Threads")
message(STATUS "  Math Libraries:        ${MATH_LIBRARIES}")
message(STATUS "  Intel MKL:             ${MKL_FOUND}")
message(STATUS "  Eigen3:                ${Eigen3_FOUND}")
message(STATUS "  SIMDjson:              ${simdjson_FOUND}")
message(STATUS "")
message(STATUS "Build Options:")
message(STATUS "  Build Tests:           ${BUILD_TESTS}")
message(STATUS "  Build Documentation:   ${BUILD_DOCS}")
message(STATUS "  Build Benchmarks:      ${BUILD_BENCHMARKS}")
message(STATUS "  Build Examples:        ${BUILD_EXAMPLES}")
message(STATUS "  Code Coverage:         ${ENABLE_COVERAGE}")
message(STATUS "  LTO:                   ${CMAKE_INTERPROCEDURAL_OPTIMIZATION}")
message(STATUS "")
message(STATUS "Targets:")
message(STATUS "  Library:               ternary_core (STATIC)")
message(STATUS "  Executable:            ternary_llm")
if(BUILD_TESTS)
    message(STATUS "  Tests:                 ternary_tests")
endif()
if(BUILD_BENCHMARKS)
    message(STATUS "  Benchmarks:            ternary_benchmarks")
endif()
if(BUILD_EXAMPLES)
    message(STATUS "  Examples:              simple_example")
endif()
if(BUILD_DOCS AND DOXYGEN_FOUND)
    message(STATUS "  Documentation:         docs target")
endif()
message(STATUS "==========================================================================")

# ============================================================================
# CUSTOM TARGETS AND COMMANDS
# ============================================================================

# Target for running all tests
add_custom_target(test
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS ternary_tests
    COMMENT "Running all tests"
)

# Clean target enhancements
add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/docs
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/coverage
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/CMakeCache.txt
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/CMakeFiles
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/cmake_install.cmake
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/Makefile
    COMMENT "Deep clean build directory"
)

# Performance verification target
add_custom_target(verify-performance
    COMMAND ternary_benchmarks
    DEPENDS ternary_benchmarks
    COMMENT "Running performance verification benchmarks"
)

# ============================================================================
# END OF CMAKE CONFIGURATION
# ============================================================================