# CMakeLists.txt
#
#     Author: Fabian Meyer
# Created On: 03 Apr 2022
#    License: MIT

configure_file(lsqcpp-config.in.cmake lsqcpp-config.cmake @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/lsqcpp-config.cmake" DESTINATION lib/lsqcpp)
