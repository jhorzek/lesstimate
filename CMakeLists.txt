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

# we want to use lessOptimizers without Rcpp -> set USE_R=0
add_definitions(-DUSE_R=0)

# use GNU standard install directories
include(GNUInstallDirs)

# find dependencies:
find_package(Armadillo REQUIRED)

# create library
add_library(lessOptimizers INTERFACE)

# include required directories to allow users to include the project
# with #include
target_include_directories(${PROJECT_NAME} 
 INTERFACE
    # ensure that the include directory is found during build
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>

    # ensure that the include directory is found during install
    # and specifies that the headers of lessOptimizers are found in the 
    # include directory within the project that uses lessOptimizers 
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# link to armadillo
target_include_directories(${PROJECT_NAME} INTERFACE ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} INTERFACE ${ARMADILLO_LIBRARIES})

# install target and export
install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets)

install(
    EXPORT ${PROJECT_NAME}-targets
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})





