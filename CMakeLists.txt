cmake_minimum_required(VERSION 3.20.0)

# define project:
project(lessOptimizers)

# specify dependencies:
find_package(Armadillo REQUIRED)

# create library
add_library(lessOptimizers INTERFACE)

# we want to use lessOptimizers without Rcpp -> set USE_R=0
add_definitions(-DUSE_R=0)

# include all directories
target_include_directories(lessOptimizers INTERFACE .)

# include armadillo
target_include_directories(${PROJECT_NAME} INTERFACE ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} INTERFACE ${ARMADILLO_LIBRARIES})

