# CMakeLists.txt
#
#     Author: Fabian Meyer
# Created On: 03 Apr 2022
#    License: MIT

cmake_minimum_required(VERSION 3.5)
project(findlsqcpp)

find_package(Eigen3 REQUIRED)
find_package(lsqcpp REQUIRED)

add_executable(findtest main.cpp)
target_link_libraries(findtest lsqcpp::lsqcpp)