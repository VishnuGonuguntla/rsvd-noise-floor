cmake_minimum_required(VERSION 3.20)
project(rsvd_noise_floor
        VERSION 0.1.0
        DESCRIPTION "Randomized SVD of financial correlation matrices, validated against Marchenko-Pastur"
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------- options
option(RNF_NATIVE_ARCH     "Compile with -march=native (AVX2/FMA micro-kernels)" ON)
option(RNF_USE_BOOST_MATH  "Cross-check our quadrature/statistics against Boost.Math" OFF)
option(RNF_USE_OPENBLAS    "Benchmark-only comparison against vendor BLAS" OFF)

include(cmake/CompilerWarnings.cmake)

# ---------------------------------------------------------------- library
add_library(rnf
  src/gemm.cpp
  src/qr.cpp
  src/syev.cpp
  src/stats.cpp
  src/mp.cpp
  src/rsvd.cpp
  src/spiked.cpp
  src/diagnostics.cpp
  src/csv.cpp
)
add_library(rnf::rnf ALIAS rnf)

target_include_directories(rnf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
rnf_set_warnings(rnf)

find_package(Threads REQUIRED)
target_link_libraries(rnf PUBLIC Threads::Threads)

if(RNF_NATIVE_ARCH)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-march=native" RNF_HAS_MARCH_NATIVE)
  if(RNF_HAS_MARCH_NATIVE)
    target_compile_options(rnf PUBLIC -march=native)
  endif()
endif()

# Never -ffast-math: FP reassociation and broken NaN handling would undercut
# the whole numerical-accuracy story this project is built to demonstrate.
target_compile_options(rnf PUBLIC $<$<CONFIG:Release>:-O3;-fno-math-errno>)

if(RNF_USE_BOOST_MATH)
  include(cmake/BoostMath.cmake)
  rnf_enable_boost_math(rnf)
endif()

if(RNF_USE_OPENBLAS)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(OPENBLAS REQUIRED IMPORTED_TARGET openblas)
  target_link_libraries(rnf PUBLIC PkgConfig::OPENBLAS)
  target_compile_definitions(rnf PUBLIC RNF_HAVE_OPENBLAS=1)
endif()

# ---------------------------------------------------------------- apps
foreach(app rnf_mp_null rnf_spiked rnf_bench rnf_analyze)
  add_executable(${app} apps/${app}.cpp)
  target_link_libraries(${app} PRIVATE rnf::rnf)
  rnf_set_warnings(${app})
endforeach()

# ---------------------------------------------------------------- tests
include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

message(STATUS "rnf: build=${CMAKE_BUILD_TYPE} native=${RNF_NATIVE_ARCH} boost_math=${RNF_USE_BOOST_MATH} openblas=${RNF_USE_OPENBLAS}")
