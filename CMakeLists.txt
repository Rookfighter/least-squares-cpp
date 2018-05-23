# CMakeLists.txt
#
#     Author: Fabian Meyer
# Created On: 23 May 2018

cmake_minimum_required(VERSION 3.5)
project(optcpp)

SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall")
set(DEP_DIR "${CMAKE_SOURCE_DIR}/dep")

add_subdirectory(dep)

include_directories(
	${EIGEN_INCLUDE_DIR}
    ${CATCH2_INCLUDE_DIR}
	"include"
)

file(GLOB SRC src/optcpp/*.cpp)

add_library(optcpp ${SRC})
