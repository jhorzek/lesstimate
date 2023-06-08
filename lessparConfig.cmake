# Adapted from Jefferson Amstutz
# at https://www.youtube.com/watch?v=GmcMct7LJuE

# Check if the library has already been loaded
if(TARGET lesspar::lesspar)
    # has already been loaded -> just return
    return()
endif()

find_package(Armadillo REQUIRED)

# define library
add_library(lesspar::lesspar INTERFACE IMPORTED)
# we want to use lesspar without Rcpp -> set USE_R=0
target_compile_definitions(lesspar::lesspar INTERFACE -DUSE_R=0)
target_compile_features(lesspar::lesspar INTERFACE cxx_std_14)

# include current directory
target_include_directories(lesspar::lesspar INTERFACE ${CMAKE_CURRENT_LIST_DIR})

# include armadillo directory 
target_include_directories(lesspar::lesspar INTERFACE ${ARMADILLO_INCLUDE_DIRS})
# and link to armadillo
target_link_libraries(lesspar::lesspar INTERFACE ${ARMADILLO_LIBRARIES})

if(NOT DEFINED lesspar_FIND_QUIETLY)
    message(STATUS "Found lesspar in ${CMAKE_CURRENT_LIST_DIR}.")
endif()
