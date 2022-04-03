# Findlsqcpp.cmake
#
#     Author: Fabian Meyer
# Created On: 05 Oct 2018
#
# Defines
#   lsqcpp::lsqcpp
#   lsqcpp_INCLUDE_DIR
#   lsqcpp_FOUND

find_path(lsqcpp_INCLUDE_DIR
    lsqcpp/lsqcpp.hpp
    HINTS
    ${lsqcpp_ROOT}
    ENV lsqcpp_ROOT
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(lsqcpp DEFAULT_MSG lsqcpp_INCLUDE_DIR)

if(${lsqcpp_FOUND})
    add_library(lsqcpp INTERFACE)
    target_include_directories(lsqcpp INTERFACE "${lsqcpp_INCLUDE_DIR}")
    target_link_libraries(lsqcpp INTERFACE Eigen3::Eigen3)
    add_library(lsqcpp::lsqcpp ALIAS lsqcpp)
endif()