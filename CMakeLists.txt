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
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Enable AVX-512F for maximum performance on supported hardware
    list(APPEND ARCH_FLAGS "-march=native" "-mavx512f")
    
    # General optimization flags for Release builds
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -ffast-math -funroll-loops")
    
    # Additional optimization flags
    list(APPEND CMAKE_CXX_FLAGS_RELEASE "-fno-exceptions" "-fno-rtti")
    list(APPEND CMAKE_CXX_FLAGS_RELEASE "-fvectorize" "-ftree-vectorize")
    list(APPEND CMAKE_CXX_FLAGS_RELEASE "-flto" "-fmerge-constants")
    
endif()

# Apply architecture flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARCH_FLAGS}")

# ============================================================================
# FIND REQUIRED PACKAGES
# ============================================================================
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

# Math library (required for mathematical operations)
find_library(MATH_LIBRARY m)
if(MATH_LIBRARY)
    set(LIBS ${LIBS} ${MATH_LIBRARY})
endif()

# ============================================================================
# INCLUDE DIRECTORIES
# ============================================================================
include_directories(include)

# ============================================================================
# LIBRARY TARGETS
# ============================================================================

# Main ternary core library
add_library(ternary_core STATIC
    src/main.cpp
)

# Set library properties
set_target_properties(ternary_core PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    CXX_STANDARD 20
)

# Link libraries
target_link_libraries(ternary_core 
    Threads::Threads
    OpenMP::OpenMP_CXX
    ${LIBS}
)

# ============================================================================
# EXECUTABLE TARGETS
# ============================================================================

# Main executable
add_executable(ternary_llm src/main.cpp)

# Set executable properties
set_target_properties(ternary_llm PROPERTIES
    CXX_STANDARD 20
    OUTPUT_NAME "ternary-llm"
)

# Link libraries to main executable
target_link_libraries(ternary_llm
    ternary_core
    Threads::Threads
    OpenMP::OpenMP_CXX
    ${LIBS}
)

# ============================================================================
# TESTING
# ============================================================================
option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
    enable_testing()
    
    # Create test executable
    add_executable(ternary_tests tests/test_ternary_core.cpp)
    target_link_libraries(ternary_tests ternary_core)
    
    # Add test
    add_test(NAME TernaryCoreTests COMMAND ternary_tests)
endif()

# ============================================================================
# BENCHMARKING
# ============================================================================
option(BUILD_BENCHMARKS "Build performance benchmarks" ON)
if(BUILD_BENCHMARKS)
    # Create benchmark executable
    add_executable(ternary_bench benchmarks/ternary_benchmarks.cpp)
    target_link_libraries(ternary_bench ternary_core)
endif()

# ============================================================================
# INSTALLATION
# ============================================================================
install(TARGETS ternary_llm ternary_core
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

# ============================================================================
# SUMMARY
# ============================================================================
message(STATUS "")
message(STATUS "========================================")
message(STATUS "TernaryLLM-20B Build Configuration")
message(STATUS "========================================")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "AVX-512 Support: ${ARCH_FLAGS}")
message(STATUS "Tests: ${BUILD_TESTS}")
message(STATUS "Benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "========================================")
message(STATUS "")