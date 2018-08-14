# CMakeLists.txt
#
#     Author: Fabian Meyer
# Created On: 23 May 2018

cmake_minimum_required(VERSION 3.5)
project(optcpp)

SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")

set(OPTCPP_LIBRARY "optcpp")

add_subdirectory(dep)

include_directories(
    include
    ${EIGEN3_INCLUDE_DIR}
    ${CATCH2_INCLUDE_DIR}
)

install(DIRECTORY include/optcpp
        DESTINATION include
        FILES_MATCHING PATTERN "*.h")

# Compile tests
file(GLOB TEST_SRC test/*.cpp)
add_executable(run_tests ${TEST_SRC})
