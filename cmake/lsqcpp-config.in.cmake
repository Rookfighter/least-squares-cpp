# lsqcpp-config.cmake
#
#     Author: Fabian Meyer
# Created On: 04 Apr 2022
#
# Defines
#   lsqcpp::lsqcpp
#   lsqcpp_INCLUDE_DIR
#   lsqcpp_FOUND

find_path(lsqcpp_INCLUDE_DIR
          lsqcpp/lsqcpp.hpp
          HINTS "@CMAKE_INSTALL_PREFIX@/include")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(lsqcpp DEFAULT_MSG lsqcpp_INCLUDE_DIR)

if(${lsqcpp_FOUND})
    add_library(lsqcpp INTERFACE)
    target_include_directories(lsqcpp INTERFACE "${lsqcpp_INCLUDE_DIR}")
    target_link_libraries(lsqcpp INTERFACE Eigen3::Eigen)
    add_library(lsqcpp::lsqcpp ALIAS lsqcpp)
endif()