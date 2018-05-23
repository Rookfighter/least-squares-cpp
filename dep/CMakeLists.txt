# CMakeLists.txt
#
#     Author: Fabian Meyer
# Created On: 20 Apr 2018

set(EIGEN_ROOT "${DEP_DIR}/eigen" CACHE INTERNAL "")
set(EIGEN_INCLUDE_DIR "${EIGEN_ROOT}" CACHE INTERNAL "")

set(CATCH2_ROOT "${DEP_DIR}/Catch2" CACHE INTERNAL "")
set(CATCH2_INCLUDE_DIR "${CATCH2_ROOT}/single_include" CACHE INTERNAL "")
