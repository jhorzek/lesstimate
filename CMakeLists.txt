# The following is adapted from https://github.com/pr0g/cmake-examples/blob/main/examples/core/header-only/library/CMakeLists.txt
cmake_minimum_required(VERSION 3.20.0)

# define project:
project(lessOptimizers 
        VERSION 1.0.0 
        LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # this ensures that the project is built without
                              # compiler specific arguments (e.g., uint when using g++)
                              # to make the project more portable 

# Include directories
include_directories(include)

# find dependencies:
find_package(Armadillo REQUIRED)

# create library
add_library(lessOptimizers INTERFACE)

# we want to use lessOptimizers without Rcpp -> set USE_R=0
target_compile_definitions(${PROJECT_NAME} INTERFACE MY_VAR=0)

# include required directories
target_include_directories(${PROJECT_NAME} INTERFACE include)

# link to armadillo
target_include_directories(${PROJECT_NAME} INTERFACE ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} INTERFACE ${ARMADILLO_LIBRARIES})

# install target and export
install(
        # source directory with files to be installed
        DIRECTORY include/${PROJECT_NAME}
        # install them to the include folder.
        # users can now include the library with #include include/lessOptimizers.h
        DESTINATION include)