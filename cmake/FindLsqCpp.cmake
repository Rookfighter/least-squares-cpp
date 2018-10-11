# FindLsqCpp.txt
#
#     Author: Fabian Meyer
# Created On: 05 Oct 2018
#
# Defines
#   LSQCPP_INCLUDE_DIR
#   LSQCPP_FOUND

find_path(LSQCPP_INCLUDE_DIR
    HINTS
    ${LSQCPP_ROOT}
    ENV LSQCPP_ROOT
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LSQCPP DEFAULT_MSG LSQCPP_INCLUDE_DIR)
