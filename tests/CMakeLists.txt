# CMakeLists.txt
#
# Author:     Fabian Meyer
# Created On: 12 Jul 2019

include_directories(
    ${CMAKE_CURRENT_LIST_DIR}/include
)

set(TEST_SRC
    "main.cpp"
    "src/finite_differences_backward.test.cpp"
    "src/finite_differences_central.test.cpp"
    "src/finite_differences_forward.test.cpp"
    "src/finite_differences_parameter.test.cpp"

    "src/step_refiner_constant_factor.test.cpp"
    "src/step_refiner_barzilai_borwein.test.cpp"
    "src/step_refiner_armijo_backtracking.test.cpp"
    "src/step_refiner_wolfe_backtracking.test.cpp"
    "src/step_refiner_dogleg_method.test.cpp"

    "src/dense_cholesky_solver.test.cpp"
    "src/dense_svd_solver.test.cpp"

    "src/gauss_newton.test.cpp"
    "src/levenberg_marquardt.test.cpp"
)

add_executable(unittests ${TEST_SRC})
