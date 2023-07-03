cmake_minimum_required(VERSION 3.20.0)

# define project:
project(lesstimate 
        VERSION 0.0.2
        LANGUAGES CXX)
include(GNUInstallDirs)

# specify dependencies:
find_package(Armadillo REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# define target
add_library(lesstimate INTERFACE)

target_compile_features(lesstimate INTERFACE cxx_std_17)
# we don't want to use R by default
target_compile_definitions(lesstimate INTERFACE -DUSE_R=0)

target_include_directories(lesstimate INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# also include the include directory. This allows users to just specify #include "lesstimate.h"
target_link_directories(lesstimate INTERFACE include)

# link to dependencies
target_link_libraries(lesstimate INTERFACE armadillo::armadillo BLAS::BLAS LAPACK::LAPACK)

