cmake_minimum_required(VERSION 3.16)
project(student_t_srukf C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# MKL configuration
set(MKL_INTERFACE lp64)
set(MKL_THREADING sequential)  # Or intel_thread for parallel
set(MKL_MPI intelmpi)

# Find MKL
find_package(MKL CONFIG REQUIRED)

#───────────────────────────────────────────────────────────────────────────────
# Main library (static)
#───────────────────────────────────────────────────────────────────────────────

add_library(student_t_srukf STATIC
    MKL/student_t_srukf.c
)

target_include_directories(student_t_srukf PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/MKL
)

target_link_libraries(student_t_srukf PUBLIC
    MKL::MKL
)

#───────────────────────────────────────────────────────────────────────────────
# Shared library (for Python bindings)
#───────────────────────────────────────────────────────────────────────────────

add_library(student_t_srukf_shared SHARED
    MKL/student_t_srukf.c
)

target_include_directories(student_t_srukf_shared PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/MKL
)

target_link_libraries(student_t_srukf_shared PUBLIC
    MKL::MKL
)

# Set output name to match what Python expects
set_target_properties(student_t_srukf_shared PROPERTIES
    OUTPUT_NAME "student_t_srukf"
)

if(WIN32)
    # Use .def file for explicit symbol exports (more reliable than WINDOWS_EXPORT_ALL_SYMBOLS)
    target_sources(student_t_srukf_shared PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/MKL/student_t_srukf.def
    )
else()
    # Remove 'lib' prefix on Linux/Mac
    set_target_properties(student_t_srukf_shared PROPERTIES
        PREFIX ""
    )
endif()

# Compiler-specific flags
if(MSVC)
    target_compile_options(student_t_srukf PRIVATE
        /W4
        $<$<CONFIG:Release>:/O2 /fp:fast>
        $<$<CONFIG:Debug>:/Od>
    )
    target_compile_definitions(student_t_srukf PRIVATE
        _CRT_SECURE_NO_WARNINGS
    )
    target_compile_options(student_t_srukf_shared PRIVATE
        /W4
        $<$<CONFIG:Release>:/O2 /fp:fast>
        $<$<CONFIG:Debug>:/Od>
    )
    target_compile_definitions(student_t_srukf_shared PRIVATE
        _CRT_SECURE_NO_WARNINGS
    )
elseif(CMAKE_C_COMPILER_ID MATCHES "Intel")
    target_compile_options(student_t_srukf PRIVATE
        -qopt-report=5
        -qopt-report-phase=vec
    )
    target_compile_options(student_t_srukf_shared PRIVATE
        -qopt-report=5
        -qopt-report-phase=vec
    )
elseif(CMAKE_C_COMPILER_ID MATCHES "GNU")
    target_compile_options(student_t_srukf PRIVATE
        -Wall -Wextra -Wpedantic
        -O3 -march=native
        -ffast-math
        -funroll-loops
    )
    target_compile_options(student_t_srukf_shared PRIVATE
        -Wall -Wextra -Wpedantic
        -O3 -march=native
        -ffast-math
        -funroll-loops
        -fPIC
    )
elseif(CMAKE_C_COMPILER_ID MATCHES "Clang")
    target_compile_options(student_t_srukf PRIVATE
        -Wall -Wextra -Wpedantic
        -O3 -march=native
        -ffast-math
        -funroll-loops
    )
    target_compile_options(student_t_srukf_shared PRIVATE
        -Wall -Wextra -Wpedantic
        -O3 -march=native
        -ffast-math
        -funroll-loops
        -fPIC
    )
endif()

#───────────────────────────────────────────────────────────────────────────────
# Tests
#───────────────────────────────────────────────────────────────────────────────

option(BUILD_TESTS "Build test executables" ON)

if(BUILD_TESTS)
    enable_testing()
    
    # Unit tests for UKF
    add_executable(test_srukf
        test/test_srukf.c
    )
    target_link_libraries(test_srukf PRIVATE
        student_t_srukf
    )
    # Link math library only on Unix
    if(UNIX)
        target_link_libraries(test_srukf PRIVATE m)
    endif()
    add_test(NAME test_srukf COMMAND test_srukf)
    
    # Unit tests for Kelly
    add_executable(test_kelly
        test/test_kelly.c
    )
    target_link_libraries(test_kelly PRIVATE
        student_t_srukf
    )
    if(UNIX)
        target_link_libraries(test_kelly PRIVATE m)
    endif()
    add_test(NAME test_kelly COMMAND test_kelly)
    
    # Benchmark
    add_executable(bench_srukf
        test/bench_srukf.c
    )
    target_link_libraries(bench_srukf PRIVATE
        student_t_srukf
    )
    if(UNIX)
        target_link_libraries(bench_srukf PRIVATE m)
    endif()
endif()

#───────────────────────────────────────────────────────────────────────────────
# Install
#───────────────────────────────────────────────────────────────────────────────

install(TARGETS student_t_srukf student_t_srukf_shared
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES MKL/student_t_srukf.h MKL/kelly.h MKL/mkl_config_14900k.h MKL/student_t_srukf.def
    DESTINATION include
)

install(FILES python/srukf.py python/compare_ukf.py
    DESTINATION python
    OPTIONAL
)