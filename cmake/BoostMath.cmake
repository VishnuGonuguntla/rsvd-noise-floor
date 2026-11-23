# Boost.Math is OPTIONAL and used only to cross-check our own Gauss-Legendre
# quadrature and KS statistics in the test suite. The library builds and the
# apps run identically without it; nothing in src/ requires it.
#
# Resolution order: an installed Boost first, then a shallow FetchContent of
# boostorg/math in standalone mode (header-only, pulls in no other Boost).
function(rnf_enable_boost_math target)
  find_package(Boost 1.75 QUIET COMPONENTS headers)
  if(Boost_FOUND)
    message(STATUS "rnf: using system Boost.Math (${Boost_VERSION})")
    target_link_libraries(${target} PUBLIC Boost::headers)
  else()
    message(STATUS "rnf: fetching boostorg/math (standalone mode)")
    include(FetchContent)
    FetchContent_Declare(boost_math
      GIT_REPOSITORY https://github.com/boostorg/math.git
      GIT_TAG        boost-1.86.0
      GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(boost_math)
    target_include_directories(${target} PUBLIC ${boost_math_SOURCE_DIR}/include)
    target_compile_definitions(${target} PUBLIC BOOST_MATH_STANDALONE=1)
  endif()
  target_compile_definitions(${target} PUBLIC RNF_HAVE_BOOST_MATH=1)
endfunction()
