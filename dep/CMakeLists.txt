# CMakeLists.txt
#
#     Author: Fabian Meyer
# Created On: 20 Apr 2018
#    License: MIT

set(EIGEN3_ROOT "${CMAKE_CURRENT_LIST_DIR}/eigen")
set(EIGEN3_INCLUDE_DIR "${EIGEN3_ROOT}")
add_library(Eigen3 INTERFACE)
target_include_directories(Eigen3 INTERFACE "${EIGEN3_INCLUDE_DIR}")
add_library(Eigen3::Eigen3 ALIAS Eigen3)

set(CATCH2_ROOT "${CMAKE_CURRENT_LIST_DIR}/Catch2" CACHE INTERNAL "")
set(CATCH2_INCLUDE_DIR "${CATCH2_ROOT}/single_include" CACHE INTERNAL "")
add_library(Catch2 INTERFACE)
target_include_directories(Catch2 INTERFACE "${CATCH2_INCLUDE_DIR}")
add_library(Catch2::Catch2 ALIAS Catch2)
