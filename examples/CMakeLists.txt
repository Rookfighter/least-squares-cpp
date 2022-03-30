# CMakeLists.txt
#
#     Author: Fabian Meyer
# Created On: 13 Jul 2019
#    License: MIT

add_executable(custom_termination_criterion "custom_termination_criterion.cpp")
target_link_libraries(custom_termination_criterion lsqcpp::lsqcpp)

add_executable(gauss_newton_armijo_backtracking "gauss_newton_armijo_backtracking.cpp")
target_link_libraries(gauss_newton_armijo_backtracking lsqcpp::lsqcpp)

add_executable(gauss_newton_dogleg_method "gauss_newton_dogleg_method.cpp")
target_link_libraries(gauss_newton_dogleg_method lsqcpp::lsqcpp)

add_executable(gauss_newton_wolfe_backtracking "gauss_newton_wolfe_backtracking.cpp")
target_link_libraries(gauss_newton_wolfe_backtracking lsqcpp::lsqcpp)

add_executable(gradient_descent_barzilai_borwein "gradient_descent_barzilai_borwein.cpp")
target_link_libraries(gradient_descent_barzilai_borwein lsqcpp::lsqcpp)

add_executable(levenberg_marquardt "levenberg_marquardt.cpp")
target_link_libraries(levenberg_marquardt lsqcpp::lsqcpp)

add_executable(provide_explicit_jacobian "provide_explicit_jacobian.cpp")
target_link_libraries(provide_explicit_jacobian lsqcpp::lsqcpp)

add_executable(pointcloud_registration "pointcloud_registration.cpp")
target_link_libraries(pointcloud_registration lsqcpp::lsqcpp)
