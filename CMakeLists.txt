# CMakeLists.txt
#
#     Author: Fabian Meyer
# Created On: 23 May 2018

cmake_minimum_required(VERSION 3.5)
project(optcpp)

SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall")

set(OPTCPP_LIBRARY "optcpp")

add_subdirectory(dep)

include_directories(
    include
    ${EIGEN3_INCLUDE_DIR}
    ${CATCH2_INCLUDE_DIR}
)

file(GLOB SRC src/optcpp/*.cpp)

add_library(${OPTCPP_LIBRARY} ${SRC})

install(TARGETS ${OPTCPP_LIBRARY}
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
install(DIRECTORY include/optcpp DESTINATION include)
