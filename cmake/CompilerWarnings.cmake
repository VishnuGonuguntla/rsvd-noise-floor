# Shared warning set. Deliberately strict: numerical code hides bugs in
# implicit conversions and shadowed loop variables.
function(rnf_set_warnings target)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE
      -Wall -Wextra -Wpedantic
      -Wconversion -Wsign-conversion
      -Wshadow -Wnon-virtual-dtor -Wold-style-cast
      -Wcast-align -Wunused -Woverloaded-virtual
      -Wdouble-promotion -Wformat=2
    )
  endif()
endfunction()
