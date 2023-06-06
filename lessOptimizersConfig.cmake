# Adapted from Jefferson Amstutz
# at https://www.youtube.com/watch?v=GmcMct7LJuE

# Check if the library has already been loaded
if(TARGET lessOptimizers::lessOptimizers)
    # has already been loaded -> just return
    return()
endif()

find_package(Armadillo REQUIRED)

# define library
add_library(lessOptimizers::lessOptimizers INTERFACE IMPORTED)
# we want to use lessOptimizers without Rcpp -> set USE_R=0
target_compile_definitions(lessOptimizers::lessOptimizers INTERFACE -DUSE_R=0)
target_compile_features(lessOptimizers::lessOptimizers INTERFACE cxx_std_14)

# include current directory
target_include_directories(lessOptimizers::lessOptimizers INTERFACE ${CMAKE_CURRENT_LIST_DIR})

# include armadillo directory 
target_include_directories(lessOptimizers::lessOptimizers INTERFACE ${ARMADILLO_INCLUDE_DIRS})
# and link to armadillo
target_link_libraries(lessOptimizers::lessOptimizers INTERFACE ${ARMADILLO_LIBRARIES})

if(NOT DEFINED lessOptimizers_FIND_QUIETLY)
    message(STATUS "Found lessOptimizers in ${CMAKE_CURRENT_LIST_DIR}.")
endif()
