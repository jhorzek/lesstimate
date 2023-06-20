# Adapted from Jefferson Amstutz
# at https://www.youtube.com/watch?v=GmcMct7LJuE

# Check if the library has already been loaded
if(TARGET lesstimate::lesstimate)
    # has already been loaded -> just return
    return()
endif()

find_package(Armadillo REQUIRED)

# define library
add_library(lesstimate::lesstimate INTERFACE IMPORTED)
# we want to use lesstimate without Rcpp -> set USE_R=0
target_compile_definitions(lesstimate::lesstimate INTERFACE -DUSE_R=0)
target_compile_features(lesstimate::lesstimate INTERFACE cxx_std_14)

# include current directory
target_include_directories(lesstimate::lesstimate INTERFACE ${CMAKE_CURRENT_LIST_DIR})

# include armadillo directory 
target_include_directories(lesstimate::lesstimate INTERFACE ${ARMADILLO_INCLUDE_DIRS})
# and link to armadillo
target_link_libraries(lesstimate::lesstimate INTERFACE ${ARMADILLO_LIBRARIES})

if(NOT DEFINED lesstimate_FIND_QUIETLY)
    message(STATUS "Found lesstimate in ${CMAKE_CURRENT_LIST_DIR}.")
endif()
