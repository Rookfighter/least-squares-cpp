# FindOptimizationCpp.txt
#
#     Author: Fabian Meyer
# Created On: 05 Oct 2018
#
# Defines
#   OPTCPP_INCLUDE_DIR
#   OPTCPP_FOUND

find_path(OPTCPP_INCLUDE_DIR
    HINTS
    ${OPTCPP_ROOT}
    ENV OPTCPP_ROOT
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OPTCPP DEFAULT_MSG OPTCPP_INCLUDE_DIR)
