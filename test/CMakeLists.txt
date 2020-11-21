# CMakeLists.txt
#
#     Author: Fabian Meyer
# Created On: 12 Jul 2019

include_directories(
    ${CMAKE_CURRENT_LIST_DIR}
    ${CATCH2_INCLUDE_DIR}
)

set(TEST_SRC
    "main.cpp"
    "backward_differences.double.cpp"
    "backward_differences.float.cpp"
    "forward_differences.double.cpp"
    "forward_differences.float.cpp"
    "central_differences.double.cpp"
    "central_differences.float.cpp"
    "dense_cholesky_solver.cpp"
    "dense_svd_solver.cpp"
    "dogleg_method.cpp"
    "gauss_newton.cpp"
    "gradient_descent.cpp"
    "levenberg_marquardt.cpp"
)

add_executable(run_tests ${TEST_SRC})
