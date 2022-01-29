# CMakeLists.txt
#
#     Author: Fabian Meyer
# Created On: 12 Jul 2019

include_directories(
    ${CMAKE_CURRENT_LIST_DIR}/include
)

set(TEST_SRC
    "main.cpp"
    "src/finite_differences_backward_test.cpp"
    "src/finite_differences_central_test.cpp"
    "src/finite_differences_forward_test.cpp"

    "src/step_refiner_constant_factor_test.cpp"
    "src/step_refiner_barzilai_borwein_test.cpp"
    "src/step_refiner_armijo_backtracking_test.cpp"
    "src/step_refiner_wolfe_backtracking_test.cpp"
    "src/step_refiner_dogleg_method_test.cpp"
    # "src/dense_cholesky_solver.double.cpp"
    # "src/dense_cholesky_solver.float.cpp"
    # "src/dense_svd_solver.double.cpp"
    # "src/dense_svd_solver.float.cpp"
    # "src/dogleg_method.double.cpp"
    # "src/dogleg_method.float.cpp"
    # "src/gauss_newton.cpp"
    # "src/gradient_descent.cpp"
    # "src/levenberg_marquardt.cpp"
)

add_executable(run_tests ${TEST_SRC})
