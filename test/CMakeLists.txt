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
    "dense_cholesky_solver.cpp"
    "dense_svd_solver.cpp"
    "backward_differences.cpp"
    "dogleg_method.cpp"
    "forward_differences.cpp"
    "central_differences.cpp"
    "gauss_newton.cpp"
    "gradient_descent.cpp"
    "levenberg_marquardt.cpp"
)

add_executable(run_tests ${TEST_SRC})
