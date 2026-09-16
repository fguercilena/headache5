cmake_minimum_required(VERSION 3.10)

project(HEADACHE5 VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# HDF5 library
##############
find_package(HDF5 REQUIRED)

# Test suite
############
add_executable(test test.cpp)
target_link_libraries(test hdf5)

# Install options
#################
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set_property(CACHE CMAKE_INSTALL_PREFIX PROPERTY VALUE "./")
endif()

install(TARGETS test DESTINATION .)
