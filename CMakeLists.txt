# This file was partially generated with cmake-init by friendlyanon at https://github.com/friendlyanon/cmake-init

cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(
    lesstimate
    VERSION 0.1.0
    DESCRIPTION "Optimize your models with lasso, scad, mcp, and other penalty functions"
    HOMEPAGE_URL "https://jhorzek.github.io/lesstimate/"
    LANGUAGES NONE
)


# add dependencies
# First: add CPM to download dependencies if necessary
include(cmake/CPM.cmake)
# add armadillo
CPMAddPackage(
	URL "https://sourceforge.net/projects/arma/files/armadillo-12.6.3.tar.xz"
	NAME armadillo
	VERSION 12.6.3
)

# ---- Declare library ----

add_library(lesstimate_lesstimate INTERFACE)
add_library(lesstimate::lesstimate ALIAS lesstimate_lesstimate)

set_property(
    TARGET lesstimate_lesstimate PROPERTY
    EXPORT_NAME lesstimate
)

target_include_directories(
    lesstimate_lesstimate ${warning_guard}
    INTERFACE
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
)

# we don't want to use R -> set USE_R to 0
target_compile_definitions(lesstimate_lesstimate INTERFACE -DUSE_R=0)
target_compile_features(lesstimate_lesstimate INTERFACE cxx_std_17)

target_link_libraries(lesstimate_lesstimate INTERFACE 
                      armadillo)
