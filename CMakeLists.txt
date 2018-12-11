# CMakeLists.txt
#
#     Author: Fabian Meyer
# Created On: 23 May 2018

cmake_minimum_required(VERSION 3.5)
project(lsqcpp)

SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "-Wall -Wextra")
endif(CMAKE_COMPILER_IS_GNUCXX)

add_subdirectory(dep)

include_directories(
    include
    ${EIGEN3_INCLUDE_DIR}
    ${CATCH2_INCLUDE_DIR}
    ${TPCPP_INCLUDE_DIR}
)

install(DIRECTORY include/lsq DESTINATION include)

if(${BUILD_TESTS})
    # Compile tests
    SET(TEST_SRC
        "test/main.cpp"
        "test/test_armijo_backtracking.cpp"
        "test/test_error_function.cpp"
        "test/test_gauss_newton.cpp"
        "test/test_gradient_descent.cpp"
        "test/test_increasing_line_search.cpp"
        "test/test_levenberg_marquardt.cpp"
        "test/test_solver_dense_cholesky.cpp"
        "test/test_solver_dense_svd.cpp"
        "test/test_solver_sparse_cholesky.cpp"
    )
    add_executable(run_tests ${TEST_SRC})
endif(${BUILD_TESTS})

if(${BUILD_EXAMPLES})
    add_executable(my_error_function "example/my_error_function.cpp")
endif(${BUILD_EXAMPLES})
