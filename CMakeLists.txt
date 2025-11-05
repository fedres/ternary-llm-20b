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
    PUBLIC_HEADER "include/ternary_core.hpp"
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    OUTPUT_NAME "ternary_core"
)

# Install rules for headers and library
install(TARGETS ternary_core
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
# Testing framework
option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
    enable_testing()
    
    # Google Test framework
    find_package(GTest QUIET)
    if(GTest_FOUND)
        message(STATUS "Google Test framework found")
        
        # Add a test for ternary_core
        set(TEST_SOURCES test/test_ternary_core.cpp)
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test/test_ternary_core.cpp)
            add_executable(test_ternary_core ${TEST_SOURCES})
            target_link_libraries(test_ternary_core ternary_core GTest::gtest GTest::gtest_main)
            target_include_directories(test_ternary_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
            target_compile_definitions(test_ternary_core PRIVATE TERNARY_CORE_TEST)
            
            # Apply architecture flags to tests
            if(ARCH_FLAGS)
                target_compile_options(test_ternary_core PRIVATE ${ARCH_FLAGS})
            endif()
            
            add_test(NAME TernaryCoreUnitTests COMMAND test_ternary_core)
        endif()
    else()
        message(STATUS "Google Test not found - skipping unit tests")
    endif()
endif()

# ============================================================================
# BENCHMARKING
# ============================================================================
# Benchmark executables
option(BUILD_BENCHMARKS "Build benchmarking executables" OFF)
if(BUILD_BENCHMARKS)
    # Performance benchmark for ternary operations
    set(BENCHMARK_SOURCES 
        benchmark/benchmark_ternary_operations.cpp
    )
    
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/benchmark/benchmark_ternary_operations.cpp)
        add_executable(benchmark_ternary_operations ${BENCHMARK_SOURCES})
        target_link_libraries(benchmark_ternary_operations ternary_core)
        target_compile_definitions(benchmark_ternary_operations PRIVATE TERNARY_BENCHMARK)
        
        # Apply architecture flags to benchmarks
        if(ARCH_FLAGS)
            target_compile_options(benchmark_ternary_operations PRIVATE ${ARCH_FLAGS})
        endif()
        
        # Copy benchmark results directory structure
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/benchmark_results)
        install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/benchmark_results/
                DESTINATION bin/)
    endif()
endif()

# ============================================================================
# PROFILING AND ANALYSIS
# ============================================================================
# Profiling support
set(PROFILING_FLAGS "")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Add profiling flags for debug builds
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        list(APPEND PROFILING_FLAGS "-g" "-fno-omit-frame-pointer")
    endif()
endif()

# Apply profiling flags to all targets
if(PROFILING_FLAGS)
    target_compile_options(ternary_core PRIVATE ${PROFILING_FLAGS})
    target_compile_options(ternary_llm PRIVATE ${PROFILING_FLAGS})
endif()

# ============================================================================
# CODE COVERAGE
# ============================================================================
option(ENABLE_COVERAGE "Enable code coverage reporting" OFF)
if(ENABLE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Find required tools
    find_program(GCNCMAN NAMES gcovm)
    find_program(LLVMCOVR NAMES llvm-cov)
    find_program(LLVMCMAN NAMES llvm-profdata)
    
    if(NOT GCNCMAN OR NOT LLVMCOVR OR NOT LLVMCMAN)
        message(WARNING "Code coverage tools not found - coverage disabled")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
        message(STATUS "Code coverage reporting enabled")
        
        # Add coverage target
        add_custom_target(coverage
            COMMAND ${CMAKE_CTEST_COMMAND} || true
            COMMAND gcov ${CMAKE_SOURCE_DIR}/src/ternary_core.cpp -o ${CMAKE_BINARY_DIR}/CMakeFiles/ternary_core.dir/ternary_core.cpp.gcno || true
            COMMAND llvm-cov report --object-directory ${CMAKE_BINARY_DIR}/CMakeFiles/ternary_core.dir/ ${CMAKE_BINARY_DIR}/ternary_core.a || true
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        )
    endif()
endif()

# ============================================================================
# INSTALLATION
# ============================================================================
# Install rules for headers
install(DIRECTORY include/ DESTINATION include
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)

# Install library
install(TARGETS ternary_core
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)

# Install main executable
install(TARGETS ternary_llm
    RUNTIME DESTINATION bin
)

# Install configuration files
install(FILES CMakeLists.txt DESTINATION share/ternary-llm-20b
    PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
)

# Install README and documentation
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/README.md)
    install(FILES README.md DESTINATION share/ternary-llm-20b)
endif()

# ============================================================================
# PACKAGING
# ============================================================================
# Create package configuration
set(CPACK_PACKAGE_NAME "ternary-llm-20b")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Ultra-efficient 20B parameter ternary language model")
set(CPACK_PACKAGE_VENDOR "Zombie")
set(CPACK_PACKAGE_CONTACT "ternary-llm@example.com")
set(CPACK_GENERATOR "TGZ;ZIP")

# Platform-specific packaging
if(UNIX)
    set(CPACK_PACKAGE_INSTALL_DIRECTORY "ternary-llm-20b")
    set(CPACK_STRIP_FILES true)
endif()

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

include(CPack)

# ============================================================================
# DOCUMENTATION
# ============================================================================
# Documentation generation with Doxygen
option(BUILD_DOCS "Build documentation with Doxygen" OFF)
if(BUILD_DOCS)
    find_package(Doxygen REQUIRED)
    
    # Configure Doxyfile
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    
    # Replace variables in Doxyfile
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    
    # Set Doxyfile options
    set(DOXYGEN_GENERATE_HTML YES)
    set(DOXYGEN_GENERATE_MAN YES)
    set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs)
    
    # Build documentation
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
endif()

# ============================================================================
# SUMMARY
# ============================================================================
message(STATUS "======================= Build Configuration Summary =======================")
message(STATUS "Project:                   ${PROJECT_NAME} v${PROJECT_VERSION}")
message(STATUS "Build Type:                ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard:              ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler:                  ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Architecture flags:        ${ARCH_FLAGS}")
message(STATUS "Install Prefix:            ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
message(STATUS "Options:")
message(STATUS "  USE_NVCPP:                ${USE_NVCPP}")
message(STATUS "  BUILD_EXAMPLES:           ${BUILD_EXAMPLES}")
message(STATUS "  BUILD_TESTS:              ${BUILD_TESTS}")
message(STATUS "  BUILD_BENCHMARKS:         ${BUILD_BENCHMARKS}")
message(STATUS "  BUILD_DOCS:               ${BUILD_DOCS}")
message(STATUS "  ENABLE_COVERAGE:          ${ENABLE_COVERAGE}")
message(STATUS "")
message(STATUS "Dependencies:")
message(STATUS "  OpenMP:                   ${OpenMP_FOUND} (${OpenMP_VERSION})")
message(STATUS "  Intel MKL:                ${MKL_FOUND}")
message(STATUS "  Google Test:              ${GTest_FOUND}")
message(STATUS "")
message(STATUS "Output:")
message(STATUS "  Library:                  ${CMAKE_BINARY_DIR}/libternary_core.a")
message(STATUS "  Executable:               ${CMAKE_BINARY_DIR}/ternary_llm")
message(STATUS "  Install path:             ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Documentation:            ${CMAKE_BINARY_DIR}/docs (if enabled)")
message(STATUS "==========================================================================")

# ============================================================================
# POST-BUILD SUMMARY
# ============================================================================
add_custom_command(TARGET ternary_core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "==================================="
    COMMAND ${CMAKE_COMMAND} -E echo "TernaryLLM-20B Build Complete!"
    COMMAND ${CMAKE_COMMAND} -E echo "==================================="
    COMMAND ${CMAKE_COMMAND} -E echo "Library: ${CMAKE_BINARY_DIR}/libternary_core.a"
    COMMAND ${CMAKE_COMMAND} -E echo "Install: sudo make install"
    COMMAND ${CMAKE_COMMAND} -E echo "========================================================================"
)