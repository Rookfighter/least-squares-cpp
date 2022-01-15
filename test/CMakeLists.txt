# CMakeLists.txt
#
#     Author: Fabian Meyer
# Created On: 12 Jul 2019

include_directories(
    ${CMAKE_CURRENT_LIST_DIR}/include
)

set(TEST_SRC
    "main.cpp"
    "src/backward_differences.double.cpp"
    "src/backward_differences.float.cpp"
    "src/forward_differences.double.cpp"
    "src/forward_differences.float.cpp"
    "src/central_differences.double.cpp"
    "src/central_differences.float.cpp"
    "src/dense_cholesky_solver.double.cpp"
    "src/dense_cholesky_solver.float.cpp"
    "src/dense_svd_solver.double.cpp"
    "src/dense_svd_solver.float.cpp"
    "src/dogleg_method.double.cpp"
    "src/dogleg_method.float.cpp"
    "src/gauss_newton.cpp"
    "src/gradient_descent.cpp"
    "src/levenberg_marquardt.cpp"
)

add_executable(run_tests ${TEST_SRC})
